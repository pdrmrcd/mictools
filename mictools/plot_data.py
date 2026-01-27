import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .load_data import *
from .process_data import *
from .config import *

def plot_flyscan(scanno, detector, roi, roi_type="Intensity", th=0, path=None, **kwargs):
    # Load the data
    X, Y, Z = mesh_roi_data(scanno, detector, roi, roi_type, th, path)

    # Plot the data
    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(X, Y, Z, shading='auto', **kwargs)
    fig.colorbar(pcm, ax=ax, label=f'{roi_type}')

    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')

    plt.show()
