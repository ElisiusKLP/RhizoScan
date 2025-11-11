"""
Plot implementation of the PlotStep function.
"""
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import messagebox
import tempfile
import os
import cv2
from PIL import Image
from mne.time_frequency import psd_array_welch
from mne.preprocessing import read_ica
#typing
from mne.io import BaseRaw
#custom
from src.core.step_types import PlotStep, plot_step
from src.utils.plotting import create_psd_plot

class PlotPSD(PlotStep):
    def __init__(self, fmin=1.0, fmax=60.0, active=True, save=True, show=False):
        super().__init__("PlotPSD", active=active, save=save, show=show)
        self.fmin = fmin
        self.fmax = fmax

    def plot(self, data):
        """
        raw: mne.io.Raw object
        Returns a matplotlib figure.
        """
        print(f"Plotting PSD ({self.fmin}-{self.fmax} Hz)")
        fig = data.plot_psd(fmin=self.fmin, fmax=self.fmax, show=False)  # returns fig object
        title = f"Sub {self.context.sub_id} PSD ({self.fmin}-{self.fmax} Hz)"
        fig.suptitle(title)
        
        return fig
    
@plot_step(name="PlotPSD", save=False, show=False)
def plot_psd(
    data: BaseRaw,
    fmin: int = 0,
    fmax: int = 60,
    n_fft: int = 1024,
    average: str = "mean",
    *,
    context=None
):
    """
    
    """
    print(f"Plotting PSD ({fmin}-{fmax} Hz)")

    sfreq = data.info["sfreq"]

    raw_psds, freqs = psd_array_welch(
        data.copy().pick("mag").get_data(), n_fft=n_fft,
        sfreq=sfreq, fmin=fmin, fmax=fmax, average=average
    )
    # create psd plot
    sub_id = context.sub_id if context else "No_id"
    title = f"Sub {sub_id} PSD ({fmin}-{fmax} Hz)"
    fig = create_psd_plot(
        freqs=freqs,
        raw_psds=raw_psds,
        title=title,
        )
    
    return fig

