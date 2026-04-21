# TIFF → OME-Zarr Converter (Horta-Compatible)

## Overview

This script converts TIFF image stacks into OME-Zarr (NGFF 0.4, Zarr v2) format for visualization (e.g., Horta).

---

## 🔧 Environment Setup (REQUIRED)

It is strongly recommended to use a clean Python environment to avoid dependency conflicts.

### Option 1: venv (recommended)

```bash
python3 -m venv zarr-env
source zarr-env/bin/activate
pip install --upgrade pip
pip install tifffile dask zarr==2.18.7 numcodecs==0.15.1 numpy
```

### Option 2: conda

```bash
conda create -n zarr-env python=3.10 -y
conda activate zarr-env
pip install tifffile dask zarr==2.18.7 numcodecs==0.15.1 numpy
```

### ⚠️ Important Version Notes

- Use **zarr < 3** (v2.x required)
- Use compatible numcodecs (e.g., 0.15.1)
- Newer versions may break NGFF compatibility

---

## Installation (inside environment)

```bash
pip install tifffile dask zarr numcodecs numpy
```

---

## Usage

```bash
python zarrconverter.py input.tif output.zarr
```

### Example

```bash
python zarrconverter.py \
    input.tif \
    output.zarr \
    --scale-x 0.1102 \
    --scale-y 0.1102 \
    --scale-z 0.5
```

---

## Arguments

- input_path: Input TIFF
- output_path: Output Zarr directory
- --scale-x/y/z: Override voxel size
- --levels: Pyramid levels (default 8)
- --chunk-z/y/x: Chunk sizes
- --channel-color: Display color
- --no-normalize: Disable normalization

---

## Output Structure

```
output.zarr/
├── 0/
├── 1/
├── ...
├── .zattrs
├── .zgroup
```

---

## ⚠️ Critical Compatibility Note (VERY IMPORTANT)

### Dimension Separator Issue

Horta and NGFF viewers require:

```python
dimension_separator="/"
```

If missing:
- Viewer fails to load
- Data appears empty
- Pyramid breaks

### Fix (already in script)

```python
dimension_separator="/"
```

### If dataset already broken

You must rewrite using:

```python
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
chunk_key_encoding=V2ChunkKeyEncoding("/")
```

---

## Common Issues

### 1. Missing dask
```
pip install dask
```

### 2. Wrong Z scaling
```
--scale-z <value>
```

### 3. Environment conflicts
Fix by recreating environment:

```bash
rm -rf zarr-env
python3 -m venv zarr-env
```

### 4. Zarr not loading
Most common cause:
- Missing dimension separator

---

## Validation

```python
import zarr
z = zarr.open("output.zarr")
print(z['0'].shape)
```

