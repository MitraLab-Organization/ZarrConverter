#!/usr/bin/env python3
"""
TIFF -> OME-Zarr (NGFF 0.4 / Zarr v2) converter for local Horta-compatible output.

- Input TIFF may be:
    * 3D: Z,Y,X
    * 4D: C,Z,Y,X
    * 5D: T,C,Z,Y,X
- Output is always written as T,C,Z,Y,X
- Uses local filesystem output only (no ome_zarr dependency)
"""

import os
import json
import argparse

import tifffile
import dask.array as da
import numpy as np
import zarr
import numcodecs


def get_xy_from_tiff_resolution(page):
    """
    Convert TIFF XResolution/YResolution to micrometers per pixel.

    TIFF resolution means pixels per unit.
    ResolutionUnit:
      2 = inch
      3 = centimeter
    """
    tags = page.tags

    x_res = tags.get("XResolution")
    y_res = tags.get("YResolution")
    unit_tag = tags.get("ResolutionUnit")

    if not x_res or not y_res or not unit_tag:
        return None, None

    try:
        x_num, x_den = x_res.value
        y_num, y_den = y_res.value
        x_ppu = x_num / x_den
        y_ppu = y_num / y_den
    except Exception:
        return None, None

    if x_ppu <= 0 or y_ppu <= 0:
        return None, None

    if unit_tag.value == 2:       # inch
        unit_um = 25400.0
    elif unit_tag.value == 3:     # cm
        unit_um = 10000.0
    else:
        return None, None

    x_um = unit_um / x_ppu
    y_um = unit_um / y_ppu
    return float(x_um), float(y_um)


def get_scales_from_tiff(tif):
    """
    Try to infer voxel sizes from TIFF metadata.
    Returns x_scale, y_scale, z_scale, unit.
    Defaults to 1.0 if unavailable.
    """
    x_scale = None
    y_scale = None
    z_scale = None
    unit = "micrometer"

    ij = tif.imagej_metadata or {}
    if isinstance(ij, dict):
        if "spacing" in ij:
            try:
                z_scale = float(ij["spacing"])
            except Exception:
                pass
        if "unit" in ij and ij["unit"]:
            u = str(ij["unit"]).strip().lower()
            if u in ("um", "micron", "microns", "micrometer", "micrometers"):
                unit = "micrometer"

    rx, ry = get_xy_from_tiff_resolution(tif.pages[0])
    if rx is not None:
        x_scale = rx
    if ry is not None:
        y_scale = ry

    if x_scale is None:
        x_scale = 1.0
    if y_scale is None:
        y_scale = 1.0
    if z_scale is None:
        z_scale = 1.0

    return float(x_scale), float(y_scale), float(z_scale), unit


def ensure_5d(img):
    """
    Convert input array to T,C,Z,Y,X
    """
    if img.ndim == 2:         # Y,X
        img = img[None, None, None, :, :]
    elif img.ndim == 3:       # Z,Y,X
        img = img[None, None, :, :, :]
    elif img.ndim == 4:       # C,Z,Y,X
        img = img[None, :, :, :, :]
    elif img.ndim == 5:       # T,C,Z,Y,X
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    return img


def build_pyramid(data, n_levels):
    """
    Build pyramid by downsampling spatial dims z,y,x by factors up to 2 each level.
    Uses mean downsampling.
    """
    pyramid = [data]
    current = data

    for _ in range(1, n_levels):
        fz = 2 if current.shape[2] >= 2 else 1
        fy = 2 if current.shape[3] >= 2 else 1
        fx = 2 if current.shape[4] >= 2 else 1

        if fz == fy == fx == 1:
            break

        current = da.coarsen(
            np.mean,
            current,
            axes={2: fz, 3: fy, 4: fx},
            trim_excess=True,
        )
        pyramid.append(current)

    return pyramid


def normalize_to_uint16(arr, do_normalize):
    """
    Convert one dask array to uint16.
    """
    if np.issubdtype(arr.dtype, np.floating) and do_normalize:
        vmin = float(arr.min().compute())
        vmax = float(arr.max().compute())
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
            arr = arr * 65535.0
        else:
            arr = da.zeros_like(arr, dtype=np.float32)
        arr = da.clip(arr, 0, 65535).astype(np.uint16)
        return arr

    if arr.dtype != np.uint16:
        return arr.astype(np.uint16)

    return arr


def compute_display_window(level0_tc_zyx):
    """
    Compute display window from first T/C plane.
    """
    plane = level0_tc_zyx[0, 0]
    vals = da.percentile(plane.ravel(), [0.5, 99.5]).compute()
    window_start, window_end = map(float, vals)
    min_val = float(plane.min().compute())
    max_val = float(plane.max().compute())
    return window_start, window_end, min_val, max_val


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TIFF to OME-Zarr v2 (NGFF 0.4).")
    parser.add_argument("input_path", help="Input TIFF")
    parser.add_argument("output_path", help="Output .zarr directory")
    parser.add_argument("--scale-x", type=float, default=None, help="Override X voxel size")
    parser.add_argument("--scale-y", type=float, default=None, help="Override Y voxel size")
    parser.add_argument("--scale-z", type=float, default=None, help="Override Z voxel size")
    parser.add_argument("--scale-unit", default="micrometer", help="Voxel unit")
    parser.add_argument("--levels", type=int, default=8, help="Number of pyramid levels")
    parser.add_argument("--chunk-z", type=int, default=1)
    parser.add_argument("--chunk-y", type=int, default=256)
    parser.add_argument("--chunk-x", type=int, default=256)
    parser.add_argument("--channel-color", default="FFFFFF")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with tifffile.TiffFile(args.input_path) as tif:
        img = tif.asarray()
        dtype = img.dtype
        auto_x, auto_y, auto_z, auto_unit = get_scales_from_tiff(tif)

    x_scale = args.scale_x if args.scale_x is not None else auto_x
    y_scale = args.scale_y if args.scale_y is not None else auto_y
    z_scale = args.scale_z if args.scale_z is not None else auto_z
    unit = args.scale_unit if args.scale_unit else auto_unit

    print(f"Input TIFF shape: {img.shape}, dtype: {dtype}")
    print(f"Using voxel scales (x, y, z) [{unit}]: {x_scale}, {y_scale}, {z_scale}")

    img = ensure_5d(img)
    print(f"Reordered/normalized shape (T,C,Z,Y,X): {img.shape}")

    chunks = (1, 1, args.chunk_z, args.chunk_y, args.chunk_x)
    data = da.from_array(img, chunks=chunks)

    pyramid = build_pyramid(data, args.levels)
    print(f"Writing {len(pyramid)} pyramid levels")

    os.makedirs(args.output_path, exist_ok=True)
    store = zarr.DirectoryStore(args.output_path)
    root = zarr.group(store=store, overwrite=True)

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=1)

    for i, arr in enumerate(pyramid):
        print(f"Preparing level {i}, shape={arr.shape}")
        arr = arr.rechunk(chunks)
        arr = normalize_to_uint16(arr, do_normalize=not args.no_normalize)

        zarr_arr = root.create_dataset(
            str(i),
            shape=arr.shape,
            chunks=chunks,
            dtype="uint16",
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )

        da.store(arr, zarr_arr)

    window_start, window_end, min_val, max_val = compute_display_window(data)

    scale_list = []
    for arr in pyramid:
        z_ratio = data.shape[2] / arr.shape[2]
        y_ratio = data.shape[3] / arr.shape[3]
        x_ratio = data.shape[4] / arr.shape[4]
        scale_list.append([
            1.0,
            1.0,
            z_scale * z_ratio,
            y_scale * y_ratio,
            x_scale * x_ratio,
        ])

    axes = [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]

    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": scale_list[i]}
            ],
        }
        for i in range(len(pyramid))
    ]

    multiscales = [
        {
            "name": "/",
            "version": "0.4",
            "axes": axes,
            "datasets": datasets,
        }
    ]

    default_z = int(img.shape[2] // 2) if img.shape[2] > 0 else 0

    omero = {
        "id": 1,
        "version": "0.4",
        "name": os.path.basename(args.output_path),
        "rdefs": {"defaultT": 0, "defaultZ": default_z, "model": "color"},
        "channels": [
            {
                "label": "Channel 0",
                "active": True,
                "coefficient": 1,
                "color": args.channel_color,
                "family": "linear",
                "inverted": False,
                "window": {
                    "start": window_start,
                    "end": window_end,
                    "min": min_val,
                    "max": max_val,
                },
            }
        ],
    }

    attrs = {
        "multiscales": multiscales,
        "omero": omero,
    }

    zattrs_path = os.path.join(args.output_path, ".zattrs")
    zgroup_path = os.path.join(args.output_path, ".zgroup")

    with open(zattrs_path, "w") as f:
        json.dump(attrs, f, indent=4)

    with open(zgroup_path, "w") as f:
        json.dump({"zarr_format": 2}, f, indent=4)

    print(f"Done: wrote OME-Zarr to {args.output_path}")
    print(f"Level 0 expected shape: {img.shape}")
    z = zarr.open_group(args.output_path, mode="r")
    print("Datasets:", list(z.array_keys()))
    print(f"Level 0 actual shape:   {z['0'].shape}")

if __name__ == "__main__":
    main()