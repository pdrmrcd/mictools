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
from .roi_utils import RoiRegistry
from .roi_utils import _resolve_roi


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

        # Temporary correction for ghost frame implemented by the xpress3

        if f["entry/instrument/NDAttributes/NDArrayUniqueId"][0] == -1:
            intensity = intensity[1:]
            com_y = com_y[1:]
            com_x = com_x[1:]
            times = times[1:]

    return {
        'intensity': intensity,
        'com_y': com_y,
        'com_x': com_x,
        'times': times
    }

def process_tetramm_file(file, channels):
    '''
    Process a single Tetramm HDF5 file and extract current for one or more
    channels in a single read (the file is opened once regardless of how many
    channels are requested).

    Parameters:
    - file: Path to HDF5 file (str)
    - channels: Channel number (int) or iterable of channel numbers, each
        between 1 and 4

    Returns:
    - Dictionary mapping 'Current {ch}' -> array, for each requested channel
    '''

    channels = [channels] if isinstance(channels, int) else list(channels)
    for ch in channels:
        if not isinstance(ch, int) or ch < 1 or ch > 4:
            raise ValueError("Channel number must be an integer between 1 and 4.")

    with File(file, "r") as f:
        dset = f["entry/data/data"]
        data = dset[:, 0, :]

    return {f'Current {ch}': data[:, ch - 1] for ch in channels}



def process_roi_data(scanno,
                      detector,
                      roi,
                      path=None,
                      n_workers=None,
                      replace=False,
                      roi_override=False):
    '''
    Loads processed ROI data (intensity + center of mass) from flyscan HDF5
    files using parallel processing. Returns an Nx4 DataFrame where the first
    column is timestamps, and the following columns are intensity in ROI,
    COM y-position, and COM x-position.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str). Must be an area detector such as
        'me7', 'xrd', 'ptycho', 'rayspec'.
    - roi: Region of interest defined from roi_utils.py as
        roiN = roi(y_start, y_end, x_start, x_end, name="roiN")
    - path: Path to data files (str)
    - n_workers: Number of parallel workers (int, optional).
                 Defaults to cpu_count() - 1
    '''

    path = get_path(path)

    # Accept a Roi instance or a registered ROI name; enforce name/geometry uniqueness.
    roi = _resolve_roi(roi, path, register=False, override=roi_override)

    if roi is None:
        raise ValueError(
            "process_roi_data requires a non-None 'roi' argument "
            "(a Roi instance or a registered ROI name)."
        )

    # Check if processed data already exists
    processed_path = path + f'/Processed/Scan_{scanno:04d}/{detector.lower()}.h5'
    group_path = f'entry/data/{roi.name}'
    if os.path.exists(processed_path) and not replace:
        with File(processed_path, 'r') as f:
            if group_path in f:
                data = {key: f[f'{group_path}/{key}'][:] for key in f[group_path].keys()}
                return pd.DataFrame(data)

    # Data processing

    files = file_names(scanno, detector, path)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

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

    # # Temporary correction for ghost frame implemented by the xpress3 for ME7 and RAYSPEC. This should be removed once the issue is fixed at the source.
    # if detector.upper() == 'ME7' or detector.upper() == 'RAYSPEC':
    #     with File(files[0], "r") as f:
    #         if f['entry/instrument/NDAttributes/NDArrayUniqueId'][0] == -1:
    #             df = df.iloc[1:].reset_index(drop=True)

    # Ensure processed directory exists
    save_dir = os.path.dirname(processed_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save to HDF5 (multiple ROIs for this detector share one file, each
    # living in its own subgroup under entry/data)
    with File(processed_path, 'a') as f:
        entry = f.require_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        f.require_group('entry/data')

        if group_path in f:
            del f[group_path]
        nxdata = f.create_group(group_path)
        nxdata.attrs['NX_class']   = 'NXdata'
        nxdata.attrs['signal']     = 'Intensity'
        nxdata.attrs['axes']       = 'Timestamp'
        nxdata.attrs['auxiliary_signals'] = ['COM_Y', 'COM_X']
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

    # Generating external link in master file
    master_path = path + f'/Scan_{scanno:04d}.h5'
    with File(master_path, 'a') as f:
        if f'entry/data/{detector.upper()}/Processed Data/{roi.name}' in f:
            del f[f'entry/data/{detector.upper()}/Processed Data/{roi.name}']
        f[f'entry/data/{detector.upper()}/Processed Data/{roi.name}'] = ExternalLink(processed_path, group_path)

    return df


def process_tetramm_data(scanno,
                          detector,
                          ch=None,
                          path=None,
                          n_workers=None,
                          replace=False):
    '''
    Loads processed Tetramm current data from flyscan HDF5 files using
    parallel processing. Returns a DataFrame with one 'Current {ch}' column
    per requested channel. Each raw file is opened only once per worker, even
    when several channels are requested, and already-cached channels are read
    back from the processed file instead of being reprocessed.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str). Must be a member of the 'tetramm' family,
        e.g. 'tetramm', 'tetramm1', 'tetramm2'.
    - ch: Channel number (int), a list/tuple of channel numbers, or None.
        Channels must be integers between 1 and 4. If None (default), all
        four channels (1, 2, 3, 4) are processed.
    - path: Path to data files (str)
    - n_workers: Number of parallel workers (int, optional).
                 Defaults to cpu_count() - 1
    '''

    path = get_path(path)

    if ch is None:
        channels = [1, 2, 3, 4]
    elif isinstance(ch, (list, tuple)):
        channels = list(ch)
    else:
        channels = [ch]

    for c in channels:
        if not isinstance(c, int) or c < 1 or c > 4:
            raise ValueError(
                f"Tetramm channel numbers must be integers between 1 and 4, got {c!r}."
            )

    processed_path = path + f'/Processed/Scan_{scanno:04d}/{detector.lower()}.h5'

    # Split into channels already cached (loaded back from the processed
    # file) and channels that still need to be extracted from the raw files.
    cached = {}
    to_process = []
    if not replace and os.path.exists(processed_path):
        with File(processed_path, 'r') as f:
            for c in channels:
                group_path = f'entry/data/channel_{c}'
                if group_path in f:
                    cached[c] = pd.DataFrame({f'Current {c}': f[f'{group_path}/Current {c}'][:]})
                else:
                    to_process.append(c)
    else:
        to_process = list(channels)

    if to_process:
        files = file_names(scanno, detector, path)

        # Determine number of workers
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        # Each worker opens a file once and extracts every requested channel
        # from it in one read (see process_tetramm_file).
        process_func = partial(process_tetramm_file, channels=to_process)

        # Process files in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_func, files)

        processed = {}
        for c in to_process:
            current_data = np.concatenate([r[f'Current {c}'] for r in results], axis=0)
            processed[c] = pd.DataFrame(current_data, columns=[f'Current {c}'])

        # Ensure processed directory exists
        save_dir = os.path.dirname(processed_path)
        os.makedirs(save_dir, exist_ok=True)

        with File(processed_path, 'a') as f:
            entry = f.require_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            f.require_group('entry/data')

            for c in to_process:
                group_path = f'entry/data/channel_{c}'
                if group_path in f:
                    del f[group_path]
                nxdata = f.create_group(group_path)
                nxdata.attrs['NX_class'] = 'NXdata'
                nxdata.attrs['signal']   = f'Current {c}'

                ds = nxdata.create_dataset(f'Current {c}', data=processed[c][f'Current {c}'].values)
                ds.attrs['units']     = 'nA'
                ds.attrs['long_name'] = f'Tetramm channel {c} current'

        # Generating external links in master file
        master_path = path + f'/Scan_{scanno:04d}.h5'
        with File(master_path, 'a') as f:
            for c in to_process:
                group_path = f'entry/data/channel_{c}'
                link_path = f'entry/data/{detector.upper()}/Current {c}'
                if link_path in f:
                    del f[link_path]
                f[link_path] = ExternalLink(processed_path, group_path)

        cached.update(processed)

    return pd.concat([cached[c] for c in channels], axis=1)


def process_detector_data(scanno,
                     detector,
                     roi=None,
                     ch=None,
                     path=None,
                     n_workers=None,
                     replace=False,
                     roi_override=False):
    '''
    Dispatches to the appropriate per-detector processing function based on
    the detector name, and returns whatever that function returns.

    - ROI detectors ('me7', 'xrd', 'ptycho', 'rayspec') are routed to
      process_roi_data (Nx4 DataFrame: Timestamp, Intensity, COM_Y, COM_X).
    - Tetramm-family detectors ('tetramm', 'tetramm1', 'tetramm2', ...) are
      routed to process_tetramm_data (a DataFrame with one 'Current {ch}'
      column per requested channel).
    - Any other detector name returns a message stating it is not yet
      supported, instead of raising.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str).
    - roi: Region of interest defined from roi_utils.py as
        roiN = roi(y_start, y_end, x_start, x_end, name="roiN")
    - ch: Tetramm channel number (int), a list/tuple of channel numbers
        (each between 1 and 4), or None to process all four channels.
        Only meaningful for tetramm-family detectors.
    - path: Path to data files (str)
    - n_workers: Number of parallel workers (int, optional).
                 Defaults to cpu_count() - 1
    '''

    ROI_DETECTORS = {'xrd', 'ptycho', 'me7', 'rayspec'}

    detector_key = detector.lower()

    if detector_key in ROI_DETECTORS:
        return process_roi_data(scanno, detector, roi=roi, path=path,
                                 n_workers=n_workers, replace=replace,
                                 roi_override=roi_override)
    elif detector_key.startswith('tetramm'):
        return process_tetramm_data(scanno, detector, ch=ch, path=path,
                                     n_workers=n_workers, replace=replace)
    else:
        return f"Detector '{detector}' is not yet supported."


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
    processed_path = path + f'/Processed/Scan_{scanno:04d}/position.h5'
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
    y_pos_um = [yi/1e4 for yi in y_pos] # type: ignore # convert to microns

    df = pd.DataFrame({'Trigger': triggers,
                       'X_Position': x_pos_um,
                       'Y_Position': y_pos_um})
    
    df['X_Position'] = -1 * (df['X_Position'] - df['X_Position'].iloc[0])
    df['Y_Position'] = df['Y_Position'] - df['Y_Position'].iloc[0]
    
    # Ensure processed directory exists
    processed_dir = os.path.dirname(processed_path)
    os.makedirs(processed_dir, exist_ok=True)

    h5_path = processed_path
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
                       norm_ch=None,
                       abs_pos=True,
                       replace=False,
                       missed_frame_position='Beginning',
                       roi_override=False):

    # Load the data

    path = get_path(path)
    master_path = path + f'/Scan_{scanno:04d}.h5'

    # Accept a Roi instance or a registered ROI name; register-on-use so the
    # scan can be recorded against the ROI (push-tracked reproducibility).
    roi = _resolve_roi(roi, path, register=True, override=roi_override)

    # Define image group path and processed data path based on roi or channel
    if roi is not None:
        RoiRegistry.load(path).record_usage(roi.name, scanno)
        images_path = f'entry/data/{detector.upper()}/Images/{roi.name}_{roi_type}'
        processed_path = path + f'/Processed/Scan_{scanno:04d}/{detector.lower()}.h5'
        parent_dataset = f'{processed_path}::entry/data/{roi.name}/{roi_type}'
    elif ch is not None:
        images_path = f'entry/data/{detector.upper()}/Images/channel_{ch}'
        processed_path = path + f'/Processed/Scan_{scanno:04d}/{detector.lower()}.h5'
        parent_dataset = f'{processed_path}::entry/data/channel_{ch}'

    if th is None:
        baseline_data = load_scan(scanno, stream='baseline', path=path)
        th = baseline_data['sample_theta'].mean()
    position_data = process_position_data(scanno, th=th, path=path, replace=replace)
    detector_data = process_detector_data(scanno, detector, roi=roi, ch=ch, path=path, replace=replace)
    if norm_detector:
        if not norm_ch:
            norm_ch=1
        norm_data = process_detector_data(scanno, norm_detector, ch=norm_ch, path=path, replace=replace)
        for col in detector_data.columns:
            detector_data[col] = detector_data[col]/norm_data[f'Current {norm_ch}']

    # Align lengths
    frame_mismatch = len(detector_data) - len(position_data)
    if frame_mismatch > 0:
        print(f'{frame_mismatch} frames mismatch. Attempting correction')
        if roi is not None:
            detector_data = detector_data[roi_type][:(-1*frame_mismatch)] if missed_frame_position=='Beginning' else detector_data[roi_type][frame_mismatch:]
        elif ch is not None:
            detector_data = detector_data[f'Current {ch}'][:(-1*frame_mismatch)] if missed_frame_position=='Beginning' else detector_data[f'Current {ch}'][frame_mismatch:]
    elif frame_mismatch < 0:
        print(f'{frame_mismatch} frames mismatch. Attempting correction')
        position_data = position_data[:(-1*frame_mismatch)] if missed_frame_position=='Beginning' else position_data[frame_mismatch:]
    else:
        if roi is not None:
            detector_data = detector_data[roi_type]
        elif ch is not None:
            detector_data = detector_data[f'Current {ch}']

    # min_len = min(len(detector_data), len(position_data))
    # position_data = position_data[:min_len]
    # if roi is not None:
    #     detector_data = detector_data[roi_type][:min_len]
    # elif ch is not None:
    #     detector_data = detector_data[f'Current {ch}'][:min_len]

    scan_info = get_scan_info(scanno, detector, path)
    nx, ny = scan_info['shape']

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

    if abs_pos:
        scan_info = get_scan_info(scanno, detector, path)
        xi = scan_info['xi']
        yi = scan_info['yi']
        xmin = scan_info['x_min'] * 1e-3
        X = X * 1e-3 + xi + xmin
        Y = Y * -1e-3 + yi

    # For master file saving with neXus structure.
    x_axis = X[0, :]
    y_axis = Y[:, 0]

    with File(master_path, 'a') as f:
        if f.get(images_path) is not None:
            del f[images_path]
        nximages = f.require_group(images_path)
        nximages.attrs['NX_class']       = 'NXdata'
        nximages.attrs['signal']         = 'Z'
        nximages.attrs['axes']           = ['Y', 'X']
        nximages.attrs['parent_dataset'] = parent_dataset
        if roi is not None:
            nximages.attrs['roi_type'] = roi_type
        nximages.create_dataset('X', data=x_axis)
        nximages.create_dataset('Y', data=y_axis)
        nximages.create_dataset('Z', data=Z)

    return X, Y, Z
