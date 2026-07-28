import numpy as np
from h5py import File
from multiprocessing import Pool, cpu_count

from .load_data import file_names
from .config import get_path


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


def sum_detector_images(scanno, detector, path=None, n_workers=None):
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
