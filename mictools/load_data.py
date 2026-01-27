import numpy as np
import pandas as pd
from h5py import File
from glob import glob
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
from functools import partial

import matplotlib.pyplot as plt

from apstools.utils import getDatabase

from mictools.config import *

def load_cat(catalog_name='19id_isn'):
    cat = getDatabase(catalog_name=catalog_name)
    return cat

def load_scan(scanno, cat=None, stream_name="primary"):
    if cat is None:
        cat = load_cat()
    if stream_name == "primary":
        scan = cat[scanno].primary.read().to_pandas()
        # scan = scan.iloc[8:]  # remove first 8 rows which are usually bad
    elif stream_name == "baseline":
        scan = cat[scanno].baseline.read().to_pandas()
    return scan

def file_names(scanno, detector, path=None):
    path = get_path(path)
    file_path = path+f'/Raw/Scan_{scanno:04d}/'+detector.upper()+f"/scan_{scanno:04d}_*.h5"
    files = glob(file_path)
    return files

def get_scan_info(scanno, detector, path=None):
    path = get_path(path)
    data_dic = {}
    files = file_names(scanno, detector, path)
    data_dic['num_files'] = len(files)
    with File(files[0], "r") as f:
        dset = f["entry/data/data"]
        data_dic['file_len'] = len(dset)
        data_dic['shape'] = (len(dset), len(files))
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



# def process_data_from_roi(frame, roi, type="sum"):
#     roi_data = frame[roi[0]:roi[1], roi[2]:roi[3]]
#     total_intensity = np.sum(roi_data)
#     if type=="sum":
#         return total_intensity
#     elif type=="com":
#         y_indices, x_indices = np.indices(roi_data.shape)
#         com_x_roi = np.sum(x_indices * roi_data) / total_intensity
#         com_y_roi = np.sum(y_indices * roi_data) / total_intensity
#         com_x = com_x_roi + roi[2]
#         com_y = com_y_roi + roi[0]
#         return com_x, com_y

# def load_data_from_roi(scanno, detector, path, roi):
#     '''
#     Loads time series data from a region of interest (ROI) in flyscan 
#     HDF5 files. Returns an Nx2 array where the first column is timestamps 
#     and the second column is summed data over the ROI.
#     Parameters:
#     - scanno: Scan number (int)
#     - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
#     - roi: Region of interest defined as (y_start, y_end, x_start, x_end) 
#            (list of int)
#     '''
#     files = file_names(scanno, detector, path)
#     data_list = []
#     times_list = []
#     for file in files:
#         with File(file, "r") as f:
#             dset = f["entry/data/data"]
#             data_roi = np.sum(dset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(1,2))
#             data_list.append(data_roi)
#             tset = f["entry/instrument/NDAttributes/NDArrayTimeStamp"]
#             times_list.append(tset[:])
#     data = np.concatenate(data_list, axis=0)
#     times = np.concatenate(times_list, axis=0)
#     data_array = np.concatenate([times[:, np.newaxis], data[:, np.newaxis]], axis=1)
#     return data_array

# def load_data_from_roi(scanno, detector, roi, path=None, n_workers=None):
#     '''
#     Loads processed data from a region of interest (ROI) in 
#     flyscan HDF5 files using parallel processing. Returns an Nx4 array 
#     where the first column is timestamps, and the following columns are 
#     intensity in ROI, COM y-position, and COM x-position.
    
#     Parameters:
#     - scanno: Scan number (int)
#     - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
#     - path: Path to data files (str)
#     - roi: Region of interest defined as (y_start, y_end, x_start, x_end) 
#            (list of int)
#     - n_workers: Number of parallel workers (int, optional). 
#                  Defaults to cpu_count() - 1
#     '''
#     files = file_names(scanno, detector, path)
    
#     # Determine number of workers
#     if n_workers is None:
#         n_workers = max(1, cpu_count() - 1)
    
#     # Create partial function with fixed roi parameter
#     process_func = partial(process_single_file, roi=roi)
    
#     # Process files in parallel
#     with Pool(processes=n_workers) as pool:
#         results = pool.map(process_func, files)
    
#     # Concatenate results
#     intensity = np.concatenate([r['intensity'] for r in results], axis=0)
#     com_y = np.concatenate([r['com_y'] for r in results], axis=0)
#     com_x = np.concatenate([r['com_x'] for r in results], axis=0)
#     times = np.concatenate([r['times'] for r in results], axis=0)
    
#     data_array = np.concatenate([
#         times[:, np.newaxis], 
#         intensity[:, np.newaxis],
#         com_y[:, np.newaxis], 
#         com_x[:, np.newaxis]
#     ], axis=1)
    
#     return data_array



# def create_interpolated_data(scanno, detector, path, roi, roi_type="sum", th=0):
#     roi_dict = {"sum": 1, "com_y": 2, "com_x": 3}
#     detector_data = load_data_from_roi(scanno, detector, path, roi)
#     data_pts = detector_data[:, roi_dict[roi_type]]

#     interf_data = load_interferometry_data(scanno, path)
#     avg_interf = interf_data.groupby('Counter3').mean()[1:len(data_pts)+1]
#     if len(avg_interf) < len(data_pts):
#         raise ValueError("Not enough interferometry data points for the " \
#         "detector data. Interferometry data points: {}, Detector data points: " \
#         "{}".format(len(avg_interf), len(data_pts)))
#     x_pts = avg_interf['I15 (X)'].values/np.cos(-1*np.radians(th))
#     y_pts = avg_interf['I7 (Y ds)'].values
#     x_pts /= 1e4  # convert to microns
#     y_pts /= 1e4  # convert to microns
#     pts = np.stack((x_pts, y_pts), axis=1)

#     scan_info = get_scan_info(scanno, detector, path)
#     ny, nx = scan_info['shape']

#     x = np.linspace(pts[:, 0].min(), pts[:, 0].max(), nx)
#     y = np.linspace(pts[:, 1].min(), pts[:, 1].max(), ny)
#     X, Y = np.meshgrid(x, y)

#     # Interpolate onto grid
#     Z_linear = griddata(pts, data_pts, (X, Y), method='linear')    # smooth, NaN outside convex hull
#     Z_nearest = griddata(pts, data_pts, (X, Y), method='nearest')  # fills everywhere

#     # Fill gaps outside convex hull using nearest neighbor
#     Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)

#     return X, Y, Z

# def plot_flyscan_data(scanno, detector, path, roi, roi_type="sum", th=0, vmin=None, vmax=None):

#     X, Y, Z = create_interpolated_data(scanno, detector, path, roi, roi_type, th)

#     plt.figure(figsize=(6,5))
#     plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
#     plt.colorbar(label=f'{roi_type} intensity')
#     plt.xlabel('X (um)')
#     plt.ylabel('Y (um)')
#     plt.title(f'Scan {scanno}, Detector {detector}, ROI Type: {roi_type}')
#     plt.show()



# def interferometry_counts_to_nm(df):
#     for col in df:
#         if col.startswith("I"):
#             df[col] = df[col]/10
#     return df

# def calculate_focal_position(df):
#     pass




# def load_1d_data(scanno,
#                 detector,
#                 roi,
#                 path,
#                 column='I7 (Y ds)',
#                 ):
    
#     '''
#     Combines data from a 1D scan for a particular detector with a determined ROI.
#     - scanno: Scan number (int)
#     - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
#     - roi: Region of interest defined as (y_start, y_end, x_start, x_end) 
#            (list of int)
#     - column: Optional. Can be used to change the interferometry data picked for the scan.
#             defaults to "I7 (Y ds)".

#     returns position_data, detector_data
#     '''
    
#     data = load_data_from_roi(scanno, detector, path, roi)
#     interf_data = load_interferometry_data(scanno, path)
#     avg_interf = interf_data.groupby('Counter3').mean()[1:]

#     pos_data = avg_interf[column].values
#     det_data = data[:,1]

#     return pos_data, det_data

# def load_2d_data(scanno):
#     pass

