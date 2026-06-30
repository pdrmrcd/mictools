import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go

from .load_data import *
from .process_data import *
from .config import *

def find_closest_trigger(scanno, x, y, path=None):
    # TODO: Include abs_pos option
    path = get_path(path)
    position_data = process_position_data(scanno, path)
    position_data['distance'] = np.sqrt((position_data['X_Position'] - x)**2 + (position_data['Y_Position'] - y)**2)
    closest_trigger = position_data.loc[position_data['distance'].idxmin()]['Trigger']
    return int(closest_trigger)

def plot_closest_frame(scanno, detector, x, y, path=None, log_scale=False, **kwargs):
    imno = find_closest_trigger(scanno, x, y, path)
    frame = load_image_from_scan(scanno, detector, imno, path)
    plt.imshow(frame, norm=colors.LogNorm(vmin=1, vmax=frame.max()) if log_scale else None, **kwargs)
    plt.show()

def plot_flyscan(scanno, 
                 detector, 
                 roi=None, 
                 roi_type="Intensity", 
                 ch=None, 
                 th=None, 
                 path=None,
                 norm_detector=None,
                 norm_ch=None,
                 abs_pos=True,
                 **kwargs):
    # Load the data
    X, Y, Z = mesh_detector_data(scanno, 
                                 detector, 
                                 roi=roi, 
                                 roi_type=roi_type, 
                                 ch=ch, 
                                 th=th, 
                                 norm_detector=norm_detector,
                                 norm_ch=norm_ch,
                                 path=path,
                                 abs_pos=abs_pos,
                                 )

    # Plot the data
    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(X, Y, Z, shading='auto', **kwargs)
    fig.colorbar(pcm, ax=ax, label=f'{roi_type}')

    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')

    ax.set_title(f'Scan {scanno} - {detector} - {roi.name if roi else ch}')

    if abs_pos:
        ax.invert_yaxis() # Invert y-axis to match the physical layout of the scan

    plt.show()


def plot_meshed_data(X, Y, Z, ax = None, fig = None, **kwargs):

    if ax is None or fig is None:
        fig, ax = plt.subplots()

    pcm = ax.pcolormesh(X, Y, Z, shading='auto', **kwargs)
    fig.colorbar(pcm, ax=ax)

    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')

    plt.show()


def plot_sum_detector_image(scanno,
                            detector,
                            summed_image=None,
                            path=None,
                            n_workers=None,
                            log_scale=False,
                            colorscale="Inferno",
                            cmin=None,
                            cmax=None,
                            width=None,
                            height=None,
                            title=None,
                            **kwargs):
    '''
    Plot summed detector image interactively with Plotly.

    Parameters:
    - scanno: Scan number (int)
    - detector: Detector name (str)
    - path: Path to data files (str)
    - n_workers: Number of workers used by sum_detector_image (int, optional)
    - log_scale: Plot log10(image + 1) if True (bool)
    - colorscale: Plotly colorscale name (str)
    - cmin: Lower limit for color axis (float, optional)
    - cmax: Upper limit for color axis (float, optional)
    - width: Figure width in pixels (int, optional)
    - height: Figure height in pixels (int, optional)
    - title: Plot title (str, optional)
    - kwargs: Additional keyword args passed to go.Heatmap

    Returns:
    - Plotly Figure object
    '''
    if summed_image is None:
        summed_image = sum_detector_image(
            scanno=scanno,
            detector=detector,
            path=path,
            n_workers=n_workers
        )

    image_to_plot = np.log10(summed_image + 1) if log_scale else summed_image
    colorbar_title = "log10(Intensity + 1)" if log_scale else "Intensity"
    plot_title = title if title is not None else f"Summed Image - Scan {scanno} - {detector}"

    fig = go.Figure(
        data=go.Heatmap(
            z=image_to_plot,
            colorscale=colorscale,
            zmin=cmin,
            zmax=cmax,
            colorbar={"title": colorbar_title},
            **kwargs
        )
    )

    fig.update_layout(
        title=plot_title,
        width=width,
        height=height,
        xaxis_title="X pixel",
        yaxis_title="Y pixel"
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.show()
    return fig
