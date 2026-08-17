import os
from functools import partial
import numpy as np
from h5py import File, ExternalLink
from multiprocessing import Pool, cpu_count
import fabio
import pyFAI

from .load_data import file_names, raw_data_dir, data_reference
from .config import get_analysis_path, get_path
from .roi_utils import _validate_name


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


# Short unit label / human-readable name for the radial axis, keyed by the
# pyFAI unit string passed as `unit=`. Falls back to the raw unit string for
# anything not listed here.
_RADIAL_UNIT_LABELS = {
    "2th_deg":  ("deg",         "Scattering angle (2θ)"),
    "2th_rad":  ("rad",         "Scattering angle (2θ)"),
    "q_A^-1":   ("1/Angstrom",  "Scattering vector (q)"),
    "q_nm^-1":  ("1/nm",        "Scattering vector (q)"),
    "r_mm":     ("mm",          "Radial distance"),
}

# Populated once per worker process by _init_integration_worker (not once per
# file / frame - see its docstring).
_worker_state = {}


def _load_mask(mask_path):
    '''
    Load a pyFAI-compatible mask array from disk, or return None.

    '.npy' files are loaded with numpy.load; anything else is handed to
    fabio.open (the image-I/O library pyFAI itself depends on), which covers
    common detector mask formats ('.edf', '.tif', ...).
    '''
    if mask_path is None:
        return None
    if mask_path.endswith('.npy'):
        return np.load(mask_path)
    return fabio.open(mask_path).data


def _init_integration_worker(poni_path, mask):
    '''
    Pool initializer: build the AzimuthalIntegrator once per worker process,
    not once per file.

    pyFAI caches an internal sparse-matrix integration engine (CSR/LUT) on an
    AzimuthalIntegrator instance, keyed on (frame shape, npt, unit, mask),
    the first time integrate1d is called with a given combination, and
    reuses it on every later call with the same parameters. Loading the
    calibration fresh per file (a naive per-task pyFAI.load) would throw
    that cache away every time; building it once per worker and letting it
    persist across every file/frame that worker handles is what keeps
    per-frame integration cheap over a multi-thousand-frame flyscan.
    '''
    _worker_state['ai'] = pyFAI.load(poni_path)
    _worker_state['mask'] = mask


def _process_azimuthal_integration_file(file, npt, unit, radial_range,
                                         error_model, polarization_factor,
                                         method):
    '''
    Perform 1D azimuthal integration of every frame in a single detector
    HDF5 file, using the calibration set up by _init_integration_worker.

    Parameters:
    - file: Path to HDF5 file (str)
    - npt, unit, radial_range, error_model, polarization_factor, method:
        Forwarded to pyFAI's AzimuthalIntegrator.integrate1d for every frame.

    Returns:
    - Dictionary with 'radial' (1D, shared by every frame in this file),
      'I' (2D, shape (n_frames, npt)), 'I_errors' (2D, same shape, or None
      if error_model is falsy), and 'Timestamp' (1D).
    '''
    ai = _worker_state['ai']
    mask = _worker_state['mask']

    with File(file, "r") as f:
        dset = f["entry/data/data"]
        n_frames = dset.shape[0]
        radial = None
        intensity = np.empty((n_frames, npt))
        errors = np.empty((n_frames, npt)) if error_model else None

        for i, frame in enumerate(dset):   # iterate directly, one frame at a time
            result = ai.integrate1d(
                frame, npt, unit=unit, mask=mask, radial_range=radial_range,
                error_model=error_model, polarization_factor=polarization_factor,
                method=method,
            )
            if radial is None:
                radial = result.radial
            intensity[i] = result.intensity
            if errors is not None:
                errors[i] = result.sigma

        times = f["entry/instrument/NDAttributes/NDArrayTimeStamp"][:]

        # Same ghost-frame correction as process_data.process_roi_file.
        if f["entry/instrument/NDAttributes/NDArrayUniqueId"][0] == -1:
            intensity = intensity[1:]
            errors = errors[1:] if errors is not None else None
            times = times[1:]

    return {'radial': radial, 'I': intensity, 'I_errors': errors, 'Timestamp': times}


def process_azimuthal_integration(scanno,
                                   detector,
                                   poni_file,
                                   integration_name,
                                   mask_file=None,
                                   path=None,
                                   npt=2000,
                                   unit="2th_deg",
                                   radial_range=None,
                                   error_model="poisson",
                                   polarization_factor=0.99,
                                   method="csr",
                                   n_workers=None,
                                   replace=False):
    '''
    Azimuthally integrate every frame of a flyscan detector into a single
    per-scan 1D-spectrum dataset, using parallel per-file processing, HDF5
    caching, and NeXus-style attributes/master-file linking - the same
    pattern process_data.process_roi_data uses for ROI intensity/COM.

    Each frame's 2D image is reduced to a 1D I(radial) spectrum with pyFAI
    (radial being 2theta, q, etc. depending on `unit`); frames from every raw
    file in the scan are combined into one dataset sharing a single radial
    axis. Multiple named integrations (e.g. different calibrations or masks)
    can coexist for the same detector as sibling groups, keyed by
    `integration_name` - the same way multiple ROIs share one detector's
    processed file, keyed by roi.name.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str), e.g. 'xrd'.
    - poni_file: Name of the .poni calibration file, resolved relative to
        the experiment's analysis/ directory (config.get_analysis_path).
    - integration_name: Name for this integration's HDF5 group (str,
        path-safe - see roi_utils._validate_name). Distinguishes multiple
        calibrations/masks stored under the same detector's processed file.
    - mask_file: Name of a mask file (any fabio-readable format, or '.npy'),
        resolved relative to the analysis/ directory, or None for no mask.
    - path: Path to data files (str)
    - npt: Number of radial bins (int)
    - unit: pyFAI radial unit, e.g. '2th_deg' (default) or 'q_A^-1'.
    - radial_range: Optional (min, max) tuple to crop the radial axis.
    - error_model: Passed to pyFAI's integrate1d to compute per-bin
        uncertainties (default 'poisson'), or None/False to skip error
        propagation.
    - polarization_factor: Synchrotron polarization correction factor
        (default 0.99, typical for a horizontally-polarized undulator
        beam), or None to disable the correction.
    - method: pyFAI integration method (default 'csr' - a sparse-matrix
        method that amortizes well across many frames on CPU and needs no
        GPU). Pass e.g. ('full', 'csr', 'opencl') to use a GPU, and consider
        lowering n_workers to avoid oversubscribing it.
    - n_workers: Number of parallel worker processes (int, optional).
                 Defaults to cpu_count() - 1

    Returns:
    - Dictionary with 'radial' (1D array, shared by every frame),
      'I' (2D array, shape (n_frames_total, npt)), 'I_errors' (2D array,
      same shape, or None if error_model is falsy), and 'Timestamp'
      (1D array, shape (n_frames_total,)).
    '''

    path = get_path(path)
    integration_name = _validate_name(integration_name)

    # Check if processed data already exists
    processed_path = path + f'/Processed/Scan_{scanno:04d}/{detector.lower()}.h5'
    group_path = f'entry/data/{integration_name}'
    if os.path.exists(processed_path) and not replace:
        with File(processed_path, 'r') as f:
            if group_path in f:
                data = {
                    'radial': f[f'{group_path}/Radial'][:],
                    'I': f[f'{group_path}/Intensity'][:],
                    'Timestamp': f[f'{group_path}/Timestamp'][:],
                }
                errors_path = f'{group_path}/Intensity_errors'
                data['I_errors'] = f[errors_path][:] if errors_path in f else None
                return data

    # Data processing

    files = file_names(scanno, detector, path)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found for scan {scanno} and detector '{detector}' in path '{path}'."
        )

    analysis_path = get_analysis_path(path)
    poni_path = os.path.join(analysis_path, poni_file)
    mask_path = os.path.join(analysis_path, mask_file) if mask_file else None
    mask = _load_mask(mask_path)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    process_func = partial(
        _process_azimuthal_integration_file, npt=npt, unit=unit,
        radial_range=radial_range, error_model=error_model,
        polarization_factor=polarization_factor, method=method,
    )

    # Process files in parallel; each worker builds its own AzimuthalIntegrator
    # once (via _init_integration_worker) and reuses it for every frame.
    with Pool(processes=n_workers, initializer=_init_integration_worker,
              initargs=(poni_path, mask)) as pool:
        results = pool.map(process_func, files)

    # Concatenate results. `radial` is identical for every frame by
    # construction (same geometry/npt/unit/mask/radial_range throughout the
    # scan), so any single result's radial axis applies to the whole scan.
    radial = results[0]['radial']
    intensity = np.concatenate([r['I'] for r in results], axis=0)
    times = np.concatenate([r['Timestamp'] for r in results], axis=0)
    errors = np.concatenate([r['I_errors'] for r in results], axis=0) if error_model else None

    unit_label, long_name = _RADIAL_UNIT_LABELS.get(str(unit), (str(unit), str(unit)))

    # Ensure processed directory exists
    save_dir = os.path.dirname(processed_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save to HDF5 (multiple integrations for this detector share one file,
    # each living in its own subgroup under entry/data, like ROIs/channels)
    with File(processed_path, 'a') as f:
        entry = f.require_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        f.require_group('entry/data')

        if group_path in f:
            del f[group_path]
        nxdata = f.create_group(group_path)
        nxdata.attrs['NX_class']            = 'NXdata'
        nxdata.attrs['signal']              = 'Intensity'
        nxdata.attrs['axes']                = ['Timestamp', 'Radial']
        nxdata.attrs['integration_name']    = integration_name
        nxdata.attrs['poni_file']           = poni_file
        nxdata.attrs['mask_file']           = mask_file if mask_file else 'none'
        nxdata.attrs['unit']                = str(unit)
        nxdata.attrs['npt']                 = npt
        nxdata.attrs['error_model']         = error_model if error_model else 'none'
        if polarization_factor is not None:
            nxdata.attrs['polarization_factor'] = polarization_factor

        # Provenance: every raw frame of this detector was integrated to get
        # here. The attributes above record the parameters of the step; this
        # records its input.
        nxdata.attrs['parent_dataset'] = data_reference(raw_data_dir(scanno, detector, path))
        nxdata.attrs['operation']      = 'azimuthal_integration'

        # Geometry snapshot for provenance/reproducibility.
        calibration = pyFAI.load(poni_path)
        nxdata.attrs['wavelength']        = calibration.wavelength
        nxdata.attrs['detector_distance'] = calibration.dist

        rds = nxdata.create_dataset('Radial', data=radial)
        rds.attrs['units']     = unit_label
        rds.attrs['long_name'] = long_name

        nxdata.create_dataset('Timestamp', data=times)

        ds = nxdata.create_dataset('Intensity', data=intensity)
        ds.attrs['long_name'] = 'Azimuthally integrated intensity'

        if errors is not None:
            eds = nxdata.create_dataset('Intensity_errors', data=errors)
            eds.attrs['long_name'] = 'Poisson counting uncertainty (1-sigma)'

    # Generating external link in master file
    master_path = path + f'/Scan_{scanno:04d}.h5'
    with File(master_path, 'a') as f:
        link_path = f'entry/data/{detector.upper()}/Processed Data/{integration_name}'
        if link_path in f:
            del f[link_path]
        f[link_path] = ExternalLink(processed_path, group_path)

    return {'radial': radial, 'I': intensity, 'I_errors': errors, 'Timestamp': times}