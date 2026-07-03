import os
from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import dask
import dask_jobqueue
import lsdb
import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
import pyarrow.compute
from astra_infer import Infer, preprocess_many
from dask.distributed import Client
#from frisky import Client
from lsdb.core.crossmatch import crossmatch_args
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from nested_pandas.utils import count_nested
from upath import UPath


dask.config.set({
    "dataframe.convert-string": False,
    "distributed.nanny.environ.ARROW_DEFAULT_MEMORY_POOL": "jemalloc",
})


MIN_NOBS = 2000
NDIMS_EMBEDDING = 512
MODEL_PATH = UPath("./best_contrastive.onnx")
NUMPY_TYPE = np.float16
ARROW_TYPE = pa.list_(pa.float16(), list_size=NDIMS_EMBEDDING)
PANDAS_DTYPE = pd.ArrowDtype(ARROW_TYPE)


def tmp_folder() -> UPath | None:
    if "LOCAL" not in os.environ:
        return None
    
    local_dir = UPath(os.environ["LOCAL"])
    if "USER" in os.environ:
        local_dir / os.environ["USER"]
    return local_dir


@dataclass
class GPUNode:
    name: str
    cores_per_gpu: int
    memory_per_gpu: int


GPU_NODES = {
    "h100-80": GPUNode("h100-80", 12, 256),
    "l40s-48": GPUNode("l40s-48", 24, 128),
    "v100-32": GPUNode("v100-32", 5, 64),
}


def parse_cli_args(argv=None):
    parser = ArgumentParser(description="Self match ZTF DR and do stuff")
    parser.add_argument("action", choices=["embed", "count", "none"])
    parser.add_argument("--region", default=None, type=parse_region,
                        help="'order,pixel' for healpix, 'start:end' for partition range, 'full' or omit for all")
    parser.add_argument("--head", action="store_true",
                        help="run with .head(5) on each partition")
    parser.add_argument("--cluster", default="local-cpu", choices=["local-cpu", "slurm-gpu"],
                        help="type of the Dask cluster")
    parser.add_argument("--gpu-node", default=GPU_NODES["l40s-48"], type=GPU_NODES.get, choices=list(GPU_NODES.values()),
                        help="type of PSC Bridges2 GPU nodes")
    parser.add_argument("--output", default=None, type=UPath,
                        help="Output path, if not provided, compute into memory and print the head")
    parser.add_argument("--debug", action="store_true",
                        help="run in debug mode: no Dask Cluster, single partition only, ignores --cluster")
    return parser.parse_args(argv)


def prepare_lc_catalog(df):
    df = df.query("lightcurve.hmjd.notna() and lightcurve.catflags == 0 and lightcurve.magerr > 0")
    df = df.dropna(subset=["lightcurve"])
    df = df.drop(columns=["lightcurve.catflags"])
    df["lightcurve.filterid"] = df["filterid"]
    df = df.drop(columns=["filterid"])
    return df


class Job(dask_jobqueue.slurm.SLURMJob):
    # Rewrite the default, which is a property equal to cores/processes
    worker_process_threads = 2

class Cluster(dask_jobqueue.SLURMCluster):
    job_cls = Job


def parse_region(s):
    if s == "full":
        return None
    if ":" in s:
        start, end = s.split(":", 1)
        return slice(int(start), int(end))
    if "," in s:
        order, pixel = s.split(",", 1)
        return lsdb.PixelSearch([(int(order), int(pixel))], fine=False)
    raise ValueError(
        f"Invalid region {s!r}: expected 'order,pixel', 'start:end', or 'full'"
    )


def embedding_into_series(a, index):
    list_array = pa.FixedSizeListArray.from_arrays(
        a.reshape(-1).astype(NUMPY_TYPE),
        type=ARROW_TYPE,
    )
    return pd.Series(list_array, dtype=PANDAS_DTYPE, index=index)


def embed(df, *, gpu: bool, cores_per_gpu: int):
    so = ort.SessionOptions()
    providers = None
    if gpu:
        if len(df) == 0:
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
        else:
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    f"GPU requested but CUDAExecutionProvider is not available. "
                    f"Available providers: {available_providers}"
                )
            so.intra_op_num_threads = cores_per_gpu
            so.inter_op_num_threads = 1
            providers = ['CUDAExecutionProvider']
    model = Infer(
        MODEL_PATH,
        providers=providers,
        sess_options=so,
    )
    if gpu and len(df) > 0:
        actual = model._session.get_providers()
        if "CUDAExecutionProvider" not in actual:
            raise RuntimeError(
                f"GPU requested but onnxruntime session fell back to CPU. "
                f"Actual providers: {actual}"
            )
    
    lc_array = pa.array(df["lc"])
    inputs = preprocess_many(
        lc_array,
        field_names={"mjd": "hmjd", "mag": "mag", "magerr": "magerr", "band": "band"},
        subsampling=["beginning", "middle", "end"],
    )
    outputs = model.predict(inputs, batch_size=128)
    df = df.assign(
        embedding_beggining=embedding_into_series(outputs[:,0,:], df.index),
        embedding_middle=embedding_into_series(outputs[:,1,:], df.index),
        embedding_end=embedding_into_series(outputs[:,2,:], df.index),
    )
    return df


class SelfmatchAlgo(AbstractCrossmatchAlgorithm):
    def __init__(self, n_neighbors, radius_arcsec, id_col: str = "objectid"):
        self.kdtree_algo = KdTreeCrossmatch(n_neighbors=n_neighbors, radius_arcsec=radius_arcsec, min_radius_arcsec=0.0)
        self.id_col = id_col

    extra_columns = pd.DataFrame({
        "_dist_arcsec": pd.Series(dtype=pd.ArrowDtype(pa.float64())),
        "koid": pd.Series(dtype=pd.ArrowDtype(pa.int64())),
    })
    
    def perform_crossmatch(self, crossmatch_args):
        left_idx, right_idx, kdtree_extra_columns = self.kdtree_algo.perform_crossmatch(crossmatch_args)
        
        partition_oids = frozenset(crossmatch_args.left_df[self.id_col])

        left_id_df = crossmatch_args.left_df[[self.id_col]].assign(
            __left_index=np.arange(len(crossmatch_args.left_df)),
        )
        right_id_df = crossmatch_args.right_df[[self.id_col]].assign(
            __right_index=np.arange(len(crossmatch_args.right_df)),
        )
        
        id_ndf = self.kdtree_algo._create_nested_crossmatch_df(
            left_df=left_id_df,
            right_df=right_id_df,
            left_idx=left_idx,
            right_idx=right_idx,
            extra_cols=kdtree_extra_columns,
            nested_column_name="__right",
            how="inner",
        )

        koid_dtype = self.extra_columns.dtypes["koid"]
        if len(id_ndf) == 0:
            id_ndf = id_ndf.assign(koid=np.array([], dtype=koid_dtype), n_matches=np.array([], dtype=np.int64))
        else:    
            id_ndf = id_ndf.map_rows(
                lambda right_oids: (np.min(right_oids), np.size(right_oids)),
                columns=[f"__right.{self.id_col}"],
                row_container="args",
                output_names=["koid", "n_matches"],
                append_columns=True,
            )
            id_ndf["koid"] = id_ndf["koid"].astype(koid_dtype)
        id_ndf = id_ndf.loc[id_ndf["koid"].isin(id_ndf[self.id_col])]
        id_ndf = id_ndf.loc[id_ndf[["koid", "n_matches"]].groupby("koid")["n_matches"].idxmax()]

        # Filter out `koid`s pointing out of the partition
        id_ndf = id_ndf.loc[id_ndf["koid"].isin(partition_oids)]

        id_flat_df = id_ndf.drop(
            columns=[self.id_col, "n_matches"],
        ).explode(
            "__right",
        )

        return (
            np.asarray(id_flat_df["__left_index"]),
            np.asarray(id_flat_df["__right_index"]),
            id_flat_df[["_dist_arcsec", "koid"]],
        )


def finalize_self_match(df):
    if len(df) == 0:
        df = df.assign(koid=np.array([], dtype=np.int64))
    else:
        df = df.map_rows(
            lambda koids: koids[0],
            columns=["lc.koid"],
            output_names=["koid"],
            append_columns=True,
            row_container="args",
        )
    df = df.drop(columns=["lc.koid"])

    # Split "lc" into "matches" and actual lightcurves
    df["matches"] = df["lc"]
    df["matches.oid"] = df["matches.objectid"]
    df = df.drop(
        columns=["matches.lightcurve", "matches.objectid"],
    ).drop(
        columns=["lc._dist_arcsec", "lc.objectid"],
    )
    df["matches._dist_arcsec"] = df["matches._dist_arcsec"].astype(pd.ArrowDtype(pa.float16()))
    df = df.sort_values("matches.oid")
    
    df["lc"] = df["lc"].nest.to_flatten_inner("lightcurve")
    df = df.sort_values("lc.hmjd")
    assert df["lc.filterid"].notna().all()
    if len(df) == 0:
        df["lc.band"] = pd.Series([], dtype=pd.ArrowDtype(pa.string()))
    else:
        filterid = pa.array(df["lc.filterid"])
        filterid_minus_1 = pa.compute.subtract(filterid, 1)
        ztf_bands = pa.array(["g", "r", "i"])
        band = pa.compute.take(ztf_bands, filterid_minus_1)
        df["lc.band"] = pd.Series(band, dtype=pd.ArrowDtype(pa.string()), index=df["lc"].nest.flat_index)
    assert df["lc.band"].notna().all()
    df = df.drop(columns=["lc.filterid"])

    df = df[["koid", "matches", "lc"]]

    return df


def count(df):
    old_size = len(df)
    assert df["lc.band"].notna().all()
    
    count_columns = ["nobs_g", "nobs_r", "nobs_i", "nobs"]
    if len(df) == 0:
        counts = pd.DataFrame(
            dict.fromkeys(count_columns, np.array([], np.uint16)),
        )
    else:  
        counts = count_nested(df, "lc", by="band", join=False)
        counts = counts.rename(
            columns={"n_lc_g": "nobs_g", "n_lc_r": "nobs_r", "n_lc_i": "nobs_i"},
        )
        for col in count_columns:
            if col in counts.columns:
                continue
            counts[col] = np.array(0, dtype=np.uint16)
        counts["nobs"] = counts["nobs_g"] + counts["nobs_r"] + counts["nobs_i"]
        counts = counts[count_columns].astype(np.uint16)

    counts["matches"] = df["matches"]
    counts["koid"] = df["koid"]

    assert len(df) == old_size
    
    return counts


class DummyClient:
    dashboard_link = "<NO DASK CLUSTER>"


def main():
    args = parse_cli_args()

    search_filter = args.region if isinstance(args.region, lsdb.PixelSearch) else None

    columns = [
        "objectid",
        "filterid",
        "lightcurve.hmjd",
        "lightcurve.mag",
        "lightcurve.magerr",
        "lightcurve.catflags",
    ]

    catalog = lsdb.open_catalog(
        # "s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats",
        "/ocean/projects/phy210048p/shared/hats/catalogs/ztf_dr24/ztf_dr24_lc",
        columns=columns,
        search_filter=search_filter,
    )

    reference = catalog[["objectid", "objra", "objdec"]]
    
    lcs = catalog.map_partitions(prepare_lc_catalog)

    xmatch = reference.crossmatch_nested(
        lcs,
        nested_column_name="lc",
        algorithm=SelfmatchAlgo(
            n_neighbors=24,
            radius_arcsec=1.0,
            id_col="objectid",
        ),
    ).map_partitions(
        finalize_self_match,
    ).query(
        f"lc.list_lengths >= {MIN_NOBS}",
    )

    if isinstance(args.region, slice):
        xmatch = xmatch.partitions[args.region]

    if args.head:
        xmatch = xmatch.map_partitions(lambda df: df.head(5))
    
    if args.action == "embed":
        xmatch = xmatch.map_partitions(embed, gpu="gpu" in args.cluster, cores_per_gpu=args.gpu_node.cores_per_gpu)
        n_workers_local_cluster = 2
        hats_name = "ztfdr24_astra_embeddings"
    elif args.action == "count":
        xmatch = xmatch.map_partitions(count)
        n_workers_local_cluster = 16
        hats_name = "ztfdr24_nobs"
    elif args.action == "none":
        n_workers_local_cluster = 16
        hats_name = "ztfdr24_selfmatch"
    else:
        raise ValueError(f"Unknown action: {args.action}")

    n_workers_local_cluster = min(n_workers_local_cluster, xmatch.npartitions)

    if args.debug:
        xmatch = xmatch.partitions[0]
        cluster_cb = lambda: nullcontext(None)
        client_cb = lambda cluster: nullcontext(DummyClient())
    else:
        match args.cluster:
            case "slurm-gpu":
                def cluster_cb():
                    cluster = Cluster(
                        processes=1,
                        queue="GPU-shared",
                        cores=args.gpu_node.cores_per_gpu,
                        memory=f"{args.gpu_node.memory_per_gpu}GB",
                        walltime="24:00:00",
                        job_extra_directives=[
                            f"--gres=gpu:{args.gpu_node.name}:1",
                        ],
                        python="pixi run uv run --with=onnxruntime-gpu==1.26 python"
                    )
                    cluster.adapt(maximum_jobs=16)
                    return cluster
            case "local-cpu":
                cluster_cb = lambda: LocalCluster(
                    n_workers=n_workers_local_cluster,
                    threads_per_worker=1,
                    memory_limit="128GB",
                    local_directory=tmp_folder(),
                )
            case _:
                raise ValueError(f"--cluster={args.cluster} is unknown")
        client_cb = lambda cluster: Client(cluster)

    print(f"{xmatch.npartitions = }")
    t1 = perf_counter()
    with cluster_cb() as cluster, client_cb(cluster) as client:
        print(f"Dask dashboard: {client.dashboard_link}")
        if args.output is None:
            df = xmatch.compute()
        else:
            xmatch.write_catalog(args.output, catalog_name=hats_name, resume=True)
    print(f"Computed in {perf_counter() - t1: .1f}s")
    if args.output is None:
        print(df.head().to_string())
        print(dict(df.dtypes))
        print(f"{df.shape = }")


if __name__ == "__main__":
    main()
