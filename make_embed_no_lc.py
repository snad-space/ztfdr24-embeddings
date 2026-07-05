#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pyarrow"]
# ///
"""Copy the hats/embed collection to hats/embed_no_lc, dropping the "lc" column.

Partition files are rewritten with polars (streaming), ancillary files are
copied as-is, and dataset/_metadata & _common_metadata are regenerated with
pyarrow from the new files. Run e.g. with `uv run make_embed_no_lc.py`.
"""

import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


ROOT = Path("/ocean/projects/phy210048p/malanche/ztfdr24-embeddings/hats")
SRC = ROOT / "embed"
DST = ROOT / "embed_no_lc"
CATALOG = "ztfdr24_astra_embeddings"
DROP_COLUMN = "lc"


def rewrite_partitions() -> list[Path]:
    src_dataset = SRC / CATALOG / "dataset"
    dst_dataset = DST / CATALOG / "dataset"
    files = sorted(src_dataset.glob("Norder=*/Dir=*/Npix=*.parquet"))
    if not files:
        raise RuntimeError(f"no partition files found under {src_dataset}")
    dst_files = []
    for i, src_file in enumerate(files, 1):
        dst_file = dst_dataset / src_file.relative_to(src_dataset)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pl.scan_parquet(src_file).select(pl.exclude(DROP_COLUMN)).sink_parquet(dst_file)
        dst_files.append(dst_file)
        if i % 500 == 0 or i == len(files):
            print(f"{i}/{len(files)} partitions rewritten", flush=True)
    return dst_files


def copy_ancillary_files() -> None:
    for name in ("collection.properties",):
        shutil.copy2(SRC / name, DST / name)
    for name in ("properties", "hats.properties", "partition_info.csv",
                 "skymap.fits", "point_map.fits"):
        shutil.copy2(SRC / CATALOG / name, DST / CATALOG / name)
    thumbnail = SRC / CATALOG / "data_thumbnail.parquet"
    pl.read_parquet(thumbnail).select(pl.exclude(DROP_COLUMN)).write_parquet(
        DST / CATALOG / "data_thumbnail.parquet"
    )


def write_parquet_metadata(dst_files: list[Path]) -> None:
    dst_dataset = DST / CATALOG / "dataset"
    schema = pq.read_schema(dst_files[0])
    metadata_collector = []
    for f in dst_files:
        md = pq.read_metadata(f)
        md.set_file_path(str(f.relative_to(dst_dataset)))
        metadata_collector.append(md)
    pq.write_metadata(schema, dst_dataset / "_metadata",
                      metadata_collector=metadata_collector)
    pq.write_metadata(schema, dst_dataset / "_common_metadata")


def update_estsize() -> None:
    size_kib = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) // 1024
    for props in (DST / "collection.properties",
                  DST / CATALOG / "properties",
                  DST / CATALOG / "hats.properties"):
        lines = props.read_text().splitlines(keepends=True)
        lines = [f"hats_estsize={size_kib}\n" if line.startswith("hats_estsize=") else line
                 for line in lines]
        props.write_text("".join(lines))


def main() -> None:
    if DST.exists():
        raise RuntimeError(f"{DST} already exists, remove it first")
    (DST / CATALOG / "dataset").mkdir(parents=True)
    dst_files = rewrite_partitions()
    copy_ancillary_files()
    write_parquet_metadata(dst_files)
    update_estsize()
    print(f"done: {DST}")


if __name__ == "__main__":
    main()
