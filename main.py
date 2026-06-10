from argparse import ArgumentParser
from contextlib import nullcontext
from time import perf_counter

import lsdb
import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
from astra_infer import Infer, preprocess_many
from dask.distributed import Client
from lsdb.core.crossmatch import crossmatch_args
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from nested_pandas.utils import count_nested
from upath import UPath


MIN_NOBS = 1000
NDIMS_EMBEDDING = 512
MODEL_PATH = UPath("./best_contrastive.onnx")
NUMPY_TYPE = np.float16
ARROW_TYPE = pa.list_(pa.float16(), list_size=NDIMS_EMBEDDING)
PANDAS_DTYPE = pd.ArrowDtype(ARROW_TYPE)


def parse_cli_args(argv=None):
    parser = ArgumentParser(description="Self match ZTF DR and do stuff")
    parser.add_argument("action", choices=["embed", "count", "none"])
    parser.add_argument("--region", default=None, type=parse_region,
                        help="'order,pixel' for healpix, 'start:end' for partition range, 'full' or omit for all")
    parser.add_argument("--head", action="store_true",
                        help="run with .head(5) on each partition")
    parser.add_argument("--output", default=None, type=UPath,
                        help="Output path, if not provided, compute into memory and print the head")
    parser.add_argument("--debug", action="store_true",
                        help="run in debug mode: no Dask Cluster, single partition only")
    return parser.parse_args(argv)


def prepare_lc_catalog(df):
    df = df.query("lightcurve.hmjd.notna() and lightcurve.catflags == 0 and lightcurve.magerr > 0")
    df = df.dropna(subset=["lightcurve"])
    df = df.drop(columns=["lightcurve.catflags"])
    df["lightcurve.filterid"] = df["filterid"]
    df = df.drop(columns=["filterid"])
    return df


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


def embed(df):
    so = ort.SessionOptions()
    # so.intra_op_num_threads = 4
    # so.inter_op_num_threads = 1
    model = Infer(
        MODEL_PATH,
        # providers=['CUDAExecutionProvider'],
        sess_options=so,
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
    df["lc.band"] = df["lc.filterid"].map({1: "g", 2: "r", 3: "i"})
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
        "s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats",
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
        xmatch = xmatch.map_partitions(embed)
        n_workers = 2
        hats_name = "ztfdr24_astra_embeddings"
    elif args.action == "count":
        xmatch = xmatch.map_partitions(count)
        n_workers = 8
        hats_name = "ztfdr24_nobs"
    elif args.action == "none":
        n_workers = 8
        hats_name = "ztfdr24_selfmatch"
    else:
        raise ValueError(f"Unknown action: {args.action}")

    n_workers = min(n_workers, xmatch.npartitions)

    if args.debug:
        xmatch = xmatch.partitions[0]
        client_cb = lambda: nullcontext(DummyClient())
    else:
        client_cb = lambda: Client(n_workers=n_workers, threads_per_worker=1, memory_limit="128GB")

    print(f"{xmatch.npartitions = }")
    t1 = perf_counter()
    with client_cb() as client:
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
