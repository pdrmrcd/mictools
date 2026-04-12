import numpy as np
import pandas as pd
from h5py import File
from glob import glob
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import yaml

import matplotlib.pyplot as plt

from apstools.utils import getDatabase

from mictools.config import *

def load_cat(catalog_name='19id_isn'):
    cat = getDatabase(catalog_name=catalog_name)
    return cat

def load_cat_scan(scanno, cat=None, stream="primary"):
    if cat is None:
        cat = load_cat()
    if stream == "primary":
        scan = cat[scanno].primary.read().to_pandas()
        # scan = scan.iloc[8:]  # remove first 8 rows which are usually bad
    elif stream == "baseline":
        scan = cat[scanno].baseline.read().to_pandas()
    return scan

def load_file_scan(scanno, path=None, stream="primary"):
    path = get_path(path)
    file = path+f'/Scan_{scanno:04d}.h5'
    with File(file, "r") as f:
        # data = {}
        if stream == "primary":
            dset = f[f"entry/data/"]
            data = {key: dset[key][:] for key in dset.keys()}
        elif stream == "baseline":
            dset = f[f"entry/instrument/bluesky/streams/baseline/"]
            data = {key: dset[key]["value"][:] for key in dset.keys()}
        elif stream == "metadata":
            dset = yaml.safe_load(f[f"entry/instrument/bluesky/metadata/plan_args"][()])
            data = {key: dset[key] for key in dset.keys()}
    df = pd.DataFrame(data)
    return df

def load_scan(scanno, path=None, stream="primary", cat=None):
    if cat is not None:
        scan = load_cat_scan(scanno, cat, stream)
    else:
        scan = load_file_scan(scanno, path, stream)
    return scan


def file_names(scanno, detector, path=None):
    path = get_path(path)
    file_path = path+f'/Raw/Scan_{scanno:04d}/'+detector.upper()+f"/scan_{scanno:04d}_*.h5"
    files = glob(file_path)
    files.sort()
    return files

def get_scan_info(scanno, detector='socketserver', path=None):
    path = get_path(path)
    data_dic = {}
    files = file_names(scanno, detector, path)
    data_dic['num_files'] = len(files)
    with File(files[0], "r") as f:
        dset = f["entry/data/data"]
        data_dic['file_len'] = len(dset)
    baseline = load_file_scan(scanno, path, stream="baseline")
    data_dic['xi'] = baseline["sample_mic_x"][0]
    data_dic['yi'] = baseline["sample_y"][0]
    metadata = load_file_scan(scanno, path, stream="metadata")
    data_dic['x_min'] = metadata['x_min'][0]
    data_dic['x_max'] = metadata['x_max'][0]
    data_dic['shape'] = (metadata['x_npts'][0], metadata['y_npts'][0])
    return data_dic

def load_image_from_scan(imno, scanno, detector, path=None):
    path = get_path(path)
    files = file_names(scanno, detector, path)
    scan_info = get_scan_info(scanno, detector, path)
    file_index = imno // scan_info['file_len']
    imno_in_file = imno % scan_info['file_len']
    file = files[file_index]
    with File(file, "r") as f:
        dset = f["entry/data/data"]
        frame = dset[imno_in_file]
    return frame

def load_interferometry_data(scanno, path=None, reduction=1):
    path = get_path(path)
    _keys = ("Counter1", "Counter2", "Counter3", "I7 (Y ds)", "I8 (Y us-ob)",
              "I9 (Y us-ib)", "I12 (Z)", "I15 (X)", "C1 radial", "C2 radial", 
              "C3 radial", "C4 axial", "C5 axial", "C6 axial", "C7 radial", 
              "I3 (HKB-us)", "I4 (HKB-ds)", "I5 (VKB-us)", "I6 (VKB-ds)", 
              "I10 (X-us)", "I11 (X-ds)", "I13 (HKB-us)", "I14 (HKB-ds)", 
              "I1 (VKB-us)"
        )
    files = file_names(scanno, "socketserver", path)
    frames = []
    for file in files:
        with File(file, "r") as f:
            frame = pd.DataFrame(f["entry/data/data"][::reduction], 
                                 columns=_keys).set_index("Counter1")
            frames.append(frame)
    df = pd.concat(frames, axis=0)
    
    return df


