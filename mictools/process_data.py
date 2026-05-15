import numpy as np
import pandas as pd
from h5py import File, ExternalLink
from multiprocessing import Pool, cpu_count
from scipy.interpolate import griddata
from functools import partial
import os

from .load_data import file_names
from .load_data import load_interferometry_data
from .load_data import get_scan_info
from .load_data import load_scan
from .config import get_path
from .roi_utils import Roi as ROI



def process_roi_file(file, roi):
    '''
    Process a single HDF5 file and extract ROI data.
    
    Parameters:
    - file: Path to HDF5 file (str)
    - roi: Region of interest defined as (y_start, y_end, x_start, x_end)
    
    Returns:
    - Dictionary containing intensity, COM positions, and timestamps
    '''
    # Create coordinate grids for the ROI
    y_coords = np.arange(roi.y_start, roi.y_end)
    x_coords = np.arange(roi.x_start, roi.x_end)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    with File(file, "r") as f:
        dset = f["entry/data/data"]
        data_roi = dset[:, roi.y_start:roi.y_end, roi.x_start:roi.x_end]
        
        # Calculate total intensity for each frame
        total_intensity = np.sum(data_roi, axis=(1, 2))
        
        # Store original intensity
        intensity = total_intensity.copy()
        
        # Avoid division by zero
        total_intensity = np.where(total_intensity == 0, 1, total_intensity)
        
        # Calculate COM using vectorized operations
        com_y = np.sum(data_roi * yy[np.newaxis, :, :], axis=(1, 2)) / total_intensity
        com_x = np.sum(data_roi * xx[np.newaxis, :, :], axis=(1, 2)) / total_intensity
        
        tset = f["entry/instrument/NDAttributes/NDArrayTimeStamp"]
        times = tset[:]
    
    return {
        'intensity': intensity,
        'com_y': com_y,
        'com_x': com_x,
        'times': times
    }

def process_tetramm_file(file, ch: int):
    '''
    Process a single Tetramm HDF5 file and extract channel current.
    
    Parameters:
    - file: Path to HDF5 file (str)
    - ch: Channel number (int) 
    
    Returns:
    - Dictionary containing intensity, COM positions, and timestamps
    '''

    if not isinstance(ch, int) or ch < 0 or ch > 3:
        raise ValueError("Channel number must be an integer between 0 and 4.")
    
    with File(file, "r") as f:
        dset = f["entry/data/data"]
        data = dset[:, 0, ch-1]
    
    return {
        f'Current {ch}': data,
    }

def process_sum_detector_file(file):
    '''
    Process a single detector HDF5 file and sum all frames into one 2D image.
    '''
    with File(file, "r") as f:
        dset = f["entry/data/data"]
        return np.sum(dset, axis=0)

def process_stack_detector_file(file):
    '''
    Process a single detector HDF5 file and return all frames as a 3D array.
    '''
    with File(file, "r") as f:
        dset = f["entry/data/data"]
        return dset[:]


def sum_detector_image(scanno, detector, path=None, n_workers=None):
    '''
    Sum all detector images across all files for a given scan.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
    - path: Path to data files (str)
    - n_workers: Number of parallel workers (int, optional).
                 Defaults to cpu_count() - 1

    Returns:
    - 2D numpy array containing the summed detector image.
    '''
    path = get_path(path)
    files = file_names(scanno, detector, path)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found for scan {scanno} and detector '{detector}' in path '{path}'."
        )

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        summed_files = pool.map(process_sum_detector_file, files)

    summed_image = np.sum(summed_files, axis=0)
    return summed_image


def stack_detector_image(scanno, detector, path=None, n_workers=None):
    '''
    Stack all detector images across all files for a given scan.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
    - path: Path to data files (str)
    - n_workers: Number of parallel workers (int, optional).
                 Defaults to cpu_count() - 1

    Returns:
    - 3D numpy array with shape (n_frames_total, y, x).
    '''
    path = get_path(path)
    files = file_names(scanno, detector, path)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found for scan {scanno} and detector '{detector}' in path '{path}'."
        )

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        stacked_files = pool.map(process_stack_detector_file, files)

    stacked_image = np.concatenate(stacked_files, axis=0)
    return stacked_image




def process_detector_data(scanno, 
                     detector, 
                     roi=None, 
                     ch=None,
                     custom_proc=None, 
                     path=None, 
                     n_workers=None,
                     replace=False):
    '''
    Loads processed data from a region of interest (ROI) or Tetramm current
    data in flyscan HDF5 files using parallel processing. 
    For ROI mode, it returns an Nx4 array where the first column is 
    timestamps, and the following columns are intensity in ROI, 
    COM y-position, and COM x-position.
    For Tetramm mode, it returns an Nx1 array with the tetramm current data.
    
    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str). Can be 'me7', 'xrd', 'ptycho'.
    - path: Path to data files (str)
    - roi: Region of interest defined from roi_utils.py as 
        roiN = roi(y_start, y_end, x_start, x_end, name="roiN")
    - n_workers: Number of parallel workers (int, optional). 
                 Defaults to cpu_count() - 1
    '''

    path = get_path(path)
    
    # Check if processed data already exists
    if roi is not None:
        processed_path = path + f'/Processed/{detector.upper()}/Scan_{scanno:04d}_{roi.name}.h5'
        if os.path.exists(processed_path) and not replace:
            with File(processed_path, 'r') as f:
                data = {key: f['entry/data'][key][:] for key in f['entry/data'].keys()}
            return pd.DataFrame(data)
    elif ch is not None:
        processed_path = path + f'/Processed/{detector.upper()}/Scan_{scanno:04d}_channel_{ch}.h5'
        if os.path.exists(processed_path) and not replace:
            with File(processed_path, 'r') as f:
                current = f[f'entry/data/Current {ch}'][:]
            return pd.DataFrame({f'Current {ch}': current})
    elif custom_proc is not None:
        processed_path = path + f'/Processed/{detector.upper()}/Scan_{scanno:04d}_{custom_proc}.h5'
    
    # Data processing

    files = file_names(scanno, detector, path)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if roi is not None:

        # Check if roi is an instance of roi class
        if not isinstance(roi, ROI):
            raise ValueError("roi must be an instance of roi class from roi_utils.py" \
            "defined from roi_utils.py as roiN = roi(y_start, y_end, x_start, x_end, name=\"roiN\")")
    
        # Create partial function with fixed roi parameter
        process_func = partial(process_roi_file, roi=roi)
        
        # Process files in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_func, files)
        
        # Concatenate results
        intensity = np.concatenate([r['intensity'] for r in results], axis=0)
        com_y = np.concatenate([r['com_y'] for r in results], axis=0)
        com_x = np.concatenate([r['com_x'] for r in results], axis=0)
        times = np.concatenate([r['times'] for r in results], axis=0)
        
        data_array = np.concatenate([
            times[:, np.newaxis], 
            intensity[:, np.newaxis],
            com_y[:, np.newaxis], 
            com_x[:, np.newaxis]
        ], axis=1)

        df = pd.DataFrame(data_array, columns=['Timestamp', 
                                            'Intensity', 
                                            'COM_Y', 
                                            'COM_X'])
        
    elif ch is not None:

        # Create partial function with fixed channel parameter
        process_func = partial(process_tetramm_file, ch=ch)
        
        # Process files in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_func, files)
        
        # Concatenate results
        current_data = np.concatenate([r[f'Current {ch}'] for r in results], axis=0)
        
        df = pd.DataFrame(current_data, columns=[f'Current {ch}'])

    elif custom_proc is not None:

        raise 'Custom data processing needs to be ran previously.'

    # Ensure processed directory exists
    save_dir = os.path.dirname(processed_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save to HDF5
    if roi is not None:
        with File(processed_path, 'w') as f:
            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'

            nxdata = entry.create_group('data')
            nxdata.attrs['NX_class']   = 'NXdata'
            nxdata.attrs['signal']     = 'Intensity'
            nxdata.attrs['axes']       = 'Timestamp'
            nxdata.attrs['roi_name']   = roi.name
            nxdata.attrs['roi_y_start'] = roi.y_start
            nxdata.attrs['roi_y_end']   = roi.y_end
            nxdata.attrs['roi_x_start'] = roi.x_start
            nxdata.attrs['roi_x_end']   = roi.x_end

            nxdata.create_dataset('Timestamp', data=df['Timestamp'].values)
            ds = nxdata.create_dataset('Intensity', data=df['Intensity'].values)
            ds.attrs['long_name'] = 'ROI Intensity'
            nxdata.create_dataset('COM_Y', data=df['COM_Y'].values)
            nxdata.create_dataset('COM_X', data=df['COM_X'].values)

        link_path = f'entry/data/{detector.upper()}/{roi.name}'

    elif ch is not None:
        with File(processed_path, 'w') as f:
            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'

            nxdata = entry.create_group('data')
            nxdata.attrs['NX_class'] = 'NXdata'
            nxdata.attrs['signal']   = f'Current {ch}'

            ds = nxdata.create_dataset(f'Current {ch}', data=df[f'Current {ch}'].values)
            ds.attrs['units']     = 'A'
            ds.attrs['long_name'] = f'Tetramm channel {ch} current'

        link_path = f'entry/data/{detector.upper()}/channel_{ch}'

    master_path = path + f'/Scan_{scanno:04d}.h5'
    with File(master_path, 'a') as f:
        if link_path in f:
            del f[link_path]
        f[link_path] = ExternalLink(processed_path, 'entry/data')

    return df


def process_position_data(scanno, 
                          path=None, 
                          processing_method='averaging', 
                          th=None, 
                          replace=False):
    '''
    Loads and processes position data from flyscan HDF5 files.
    Returns a DataFrame with timestamps and positions.
    
    Parameters:
    - scanno: Scan number (int)
    '''

    path = get_path(path)

    # Check if processed data already exists
    processed_path = path + f'/Processed/SOCKETSERVER/Scan_{scanno:04d}_position.h5'
    if os.path.exists(processed_path) and not replace:
        with File(processed_path, 'r') as f:
            triggers   = f['entry/data/Trigger'][:]
            x_position = f['entry/data/X_Position'][:]
            y_position = f['entry/data/Y_Position'][:]
        df = pd.DataFrame({'Trigger': triggers, 'X_Position': x_position, 'Y_Position': y_position})
        return df
    
    interf_data = load_interferometry_data(scanno, path)

    if th is None:
        baseline_data = load_scan(scanno, stream='baseline', path=path)
        th = baseline_data['sample_theta'].mean()

    # We drop the first point as it has not trigger data
    # For now, we will just average the data for each trigger
    avg_interf = interf_data.groupby('Counter3').mean()[1:]
    triggers = avg_interf.index.values
    if processing_method == 'basic':
        x_pos = avg_interf['I15 (X)'].values/np.cos(-1*np.radians(th))
        y_pos = avg_interf['I7 (Y ds)'].values
    elif processing_method == 'averaging':
        avg_interf = avg_interf - avg_interf.iloc[0]  # subtract the first point to set it as origin
        x1 = avg_interf['I15 (X)'].values
        x2 = avg_interf['I10 (X-us)'].values
        x3 = avg_interf['I11 (X-ds)'].values
        y1 = avg_interf['I7 (Y ds)'].values
        y2 = avg_interf['I8 (Y us-ob)'].values
        y3 = avg_interf['I9 (Y us-ib)'].values
        z = avg_interf['I12 (Z)'].values
        x_avg = (x1 + x2 + x3) / 3 
        y_avg = (y1 + y2 + y3) / 3
        x_pos = -np.sqrt(x_avg**2 + z**2)
        y_pos = y_avg


    # x_pos /= 1e4  # convert to microns
    # y_pos /= 1e4  # convert to microns
    x_pos_um = [xi/1e4 for xi in x_pos]  # convert to microns
    y_pos_um = [yi/1e4 for yi in y_pos]  # convert to microns

    df = pd.DataFrame({'Trigger': triggers,
                       'X_Position': x_pos_um,
                       'Y_Position': y_pos_um})
    
    df['X_Position'] = -1 * (df['X_Position'] - df['X_Position'].iloc[0])
    df['Y_Position'] = df['Y_Position'] - df['Y_Position'].iloc[0]
    
    # Ensure processed directory exists
    processed_dir = path + f'/Processed/SOCKETSERVER'
    os.makedirs(processed_dir, exist_ok=True)

    h5_path = processed_dir + f'/Scan_{scanno:04d}_position.h5'
    with File(h5_path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'

        nxdata = entry.create_group('data')
        nxdata.attrs['NX_class']  = 'NXdata'
        nxdata.attrs['signal']    = 'Y_Position'
        nxdata.attrs['axes']      = 'X_Position'
        nxdata.attrs['x_indices'] = [0]

        nxdata.create_dataset('Trigger', data=df['Trigger'].values)

        ds = nxdata.create_dataset('X_Position', data=df['X_Position'].values)
        ds.attrs['units']     = 'um'
        ds.attrs['long_name'] = 'Sample X position'

        ds = nxdata.create_dataset('Y_Position', data=df['Y_Position'].values)
        ds.attrs['units']     = 'um'
        ds.attrs['long_name'] = 'Sample Y position'

    master_path = path + f'/Scan_{scanno:04d}.h5'
    with File(master_path, 'a') as f:
        if 'entry/data/Position' in f:
            del f['entry/data/Position']
        f['entry/data/Position'] = ExternalLink(h5_path, 'entry/data')

    
    return df

def mesh_detector_data(scanno, 
                       detector, 
                       roi=None, 
                       roi_type="Intensity", 
                       ch=None, 
                       th=None, 
                       path=None,
                       norm_detector=False,
                       norm_ch=None):
    # Load the data
    path = get_path(path)
    if th is None:
        baseline_data = load_scan(scanno, stream='baseline', path=path)
        th = baseline_data['sample_theta'].mean()
    position_data = process_position_data(scanno, th=th, path=path)
    detector_data = process_detector_data(scanno, detector, roi=roi, ch=ch, path=path)
    if norm_detector:
        if not norm_ch:
            norm_ch=1
        norm_data = process_detector_data(scanno, norm_detector, ch=norm_ch, path=path)
        detector_data = detector_data/norm_data

    # Align lengths
    min_len = min(len(detector_data), len(position_data))
    position_data = position_data[:min_len]
    if roi is not None:
        detector_data = detector_data[roi_type][:min_len]
    elif ch is not None:
        detector_data = detector_data[f'Current {ch}'][:min_len]

    scan_info = get_scan_info(scanno, detector, path)
    ny, nx = scan_info['shape']

    x = np.linspace(position_data['X_Position'].min(), position_data['X_Position'].max(), nx)
    y = np.linspace(position_data['Y_Position'].min(), position_data['Y_Position'].max(), ny)
    X, Y = np.meshgrid(x, y)

    # Interpolate onto grid
    pts = position_data[['X_Position', 'Y_Position']].values
    data_pts = detector_data.values
    Z_linear = griddata(pts, data_pts, (X, Y), method='linear')    # smooth, NaN outside convex hull
    Z_nearest = griddata(pts, data_pts, (X, Y), method='nearest')  # fills everywhere

    # Fill gaps outside convex hull using nearest neighbor
    Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)

    return X, Y, Z
