import numpy as np
import pandas as pd
import h5py
from h5py import File
from glob import glob
import os
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
import yaml



def load_xspress3(scanno, path, detector="ME7"):
    files = file_names(scanno, detector, path)
    print(f"Loading XRF data from {files}")
    frames = None
    for file in files:
        with File(file, "r") as f:
            frame = f['entry/data/data'][:]
            if frames is None:
                frames = frame
            else:
                frames = np.concatenate((frames, frame), axis=0)
    return np.asarray(frames)

def load_processed_xspress3(file_path):
    data_dict = {}
    try:
        with h5py.File(file_path, "r") as f:
            data = f["data/data"][:]
            ch_names = f["data/ch_names"][:].astype(str).tolist()
            x_val = f["data/x_val"][:]
            y_val = f["data/y_val"][:]
            data_dict = {"data": data, "ch_names": ch_names, "x_val": x_val, "y_val": y_val}
        return data_dict
    except KeyError:
        raise KeyError(f"Data group not found in HDF5 file: {file_path}")

def interferometry_counts_to_nm(df):
    for col in df:
        if col.startswith("I"):
            df[col] = df[col]/10
    return df

def load_interferometry_data(scanno, path, reduction=1, aggregate_by=None):
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
                                 columns=_keys)
            frames.append(frame)
    df = pd.concat(frames, axis=0)
    df = interferometry_counts_to_nm(df)

    if aggregate_by is not None:
        math_dict = {k: "mean" for k in _keys if k not in ["Counter1", "Counter2", "Counter3"]}
        math_dict["Counter1"] = lambda x: x.max() - x.min()
        math_dict["Counter2"] = "first"
        math_dict["Counter3"] = "first"
        df = df.groupby(aggregate_by).agg(math_dict)
        return df   

    return df

def file_names(scanno, detector, path):
    path = path+detector.upper()+f"/scan_{scanno:04d}_*.h5"
    print(f"Path: {path}")
    files = glob(path)
    return files


def get_xy_positions_from_nexus(scanno, path):
    bs_path = os.path.join(path, "bluesky")
    scan_num_str = f"{scanno:04d}"

    x_center = None
    y_center = None

    for fn in os.listdir(bs_path):
        if scan_num_str in fn:
            try:
                with h5py.File(os.path.join(bs_path, fn), "r") as f:
                    metadata = f["entry/instrument/bluesky/metadata/initial_args"][()]
                    if isinstance(metadata, bytes):
                        metadata = metadata.decode("utf-8")
                    initial_args = yaml.safe_load(metadata)
                    x_center = initial_args["x_center"]
                    y_center = initial_args["y_center"]
            except KeyError:
                print(f"Data group not found in Nexus file: {fn}")
    return x_center, y_center


def process_interferometry_data(
    scanno, 
    path, 
    reduction=1, 
    aggregate_by="Counter3", 
    output_file=False, 
    apply_coarse_offset=True, 
    x_key="I15 (X)", 
    y_key="I7 (Y ds)"
):
    
    df = load_interferometry_data(
        scanno, 
        path, 
        reduction = reduction, 
        aggregate_by = aggregate_by
    )

    if apply_coarse_offset:
        x_center, y_center = get_xy_positions_from_nexus(scanno, path)
        print(f"X center: {x_center}, Y center: {y_center}")
        if all([x_center is not None, y_center is not None]):
            y_pos = df[y_key]/1e6
            x_pos = df[x_key]/1e6
            x_range_mm = x_pos.max() - x_pos.min()
            y_range_mm = y_pos.max() - y_pos.min()
            x_pos_offset = (x_pos - x_pos.min())-x_range_mm/2
            y_pos_offset = (y_pos - y_pos.min())-y_range_mm/2
            df[f"{x_key}_coarse"] = x_center - x_pos_offset
            df[f"{y_key}_coarse"] = y_center - y_pos_offset

    if output_file:
        output_path = os.path.join(path, "analysis/SOCKETSERVER")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"scan_{scanno:04d}.h5")
        
        with h5py.File(output_file, "w") as f:
            # Create a group named "data"
            data_group = f.create_group("data")
            
            # Store each column as a key in the data group
            for col in df.columns:
                data_group.create_dataset(col, data=df[col].values)
            
            # Store the index/times if needed
            f.create_dataset("times", data=df.index.values)
    else:
        return df


element_dict = {
    "Au_L": [900,1200],
}
def process_xps3_data(scanno, path, detector="ME7", output_file=False, 
                      element_dict=element_dict, x_key="I15 (X)", y_key="I7 (Y ds)",
                      skip_position_index=1, apply_coarse_offset=True, skip_end_index=-1):

    xrf_data_all = load_xspress3(scanno, path, detector=detector)
    # xrf_data_sum = np.sum(xrf_data_all,axis=1)
    print(f"XRF data all: {xrf_data_all.shape}")
    xrf_data_sum = xrf_data_all[:,1,:]
    print(f"XRF data sum: {xrf_data_sum.shape}")
    position_data = process_interferometry_data(scanno, path, apply_coarse_offset=apply_coarse_offset)
    print(f"socket server data: {position_data.columns}")

    x_interfo_positions = position_data[x_key.replace("_coarse", "")][skip_position_index:skip_end_index]
    y_interfo_positions = position_data[y_key.replace("_coarse", "")][skip_position_index:skip_end_index]
    x_positions = position_data[x_key][skip_position_index:skip_end_index]
    y_positions = position_data[y_key][skip_position_index:skip_end_index]
    step_size = int(np.median(np.abs(np.diff(y_interfo_positions))))
    print(f"Step size: {step_size}")

    x_interp_all = np.linspace(x_positions.min(), x_positions.max(), step_size)
    y_interp_all = np.linspace(y_positions.min(), y_positions.max(), step_size)

    print(f"X positions: {x_positions.shape}, xrf data shape: {xrf_data_all.shape[0]}")

    if xrf_data_all.shape[0] != x_positions.shape[0]:
        x_pos_data = x_positions[:xrf_data_all.shape[0]]
        y_pos_data = y_positions[:xrf_data_all.shape[0]]
        print(f"X positions: {x_pos_data.shape[0]}, Y positions: {y_pos_data.shape[0]}")
        print(f"XRF data: {xrf_data_all.shape}")
    else:
        x_pos_data = x_positions
        y_pos_data = y_positions

    x_interp_data = np.linspace(x_pos_data.min(), x_pos_data.max(), step_size)
    y_interp_data = np.linspace(y_pos_data.min(), y_pos_data.max(), step_size)

    xrf_interp_all = []
    for element, peak_range in element_dict.items():
        sel_c = np.sum(xrf_data_sum[:x_pos_data.shape[0],peak_range[0]:peak_range[1]],axis=1)
        c_interp_data = griddata((x_pos_data, y_pos_data), sel_c, (x_interp_data[None,:], y_interp_data[:,None]), method='linear')
        print(c_interp_data.shape)
        
        # # Use RectBivariateSpline for 2D-to-2D interpolation (avoids Qhull errors)
        # # Create interpolator from the first grid
        # interp_func = RectBivariateSpline(x_interp_data, y_interp_data, c_interp_data.T, kx=3, ky=3)
        # # Evaluate on the new grid
        # c_interp_all = interp_func(x_interp_all, y_interp_all)
        
        xrf_interp_all.append(c_interp_data)

    data_dict = {}
    data = np.array(xrf_interp_all)
    ch_names = list(element_dict.keys())
    x_val = x_interp_all
    y_val = y_interp_all
    dict_label = ['data', 'ch_names', 'x_val', 'y_val']
    for l in dict_label:
        if l not in locals():
            raise KeyError(f"Required dataset '{l}' not found in HDF5 file")
        else:
            data_dict[l] = locals()[l]

    if output_file:
        output_path = os.path.join(path, f"analysis/{detector}")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"scan_{scanno:04d}.h5")
        with h5py.File(output_file, "w") as f:
            data_group = f.create_group("data")
            for l in dict_label:
                data_group.create_dataset(l, data=data_dict[l])
        return data_dict
    else:
        return data_dict
