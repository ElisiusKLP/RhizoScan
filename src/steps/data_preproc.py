"""
Implementations of "data preprocessing steps".

Returns:
    data
"""
import numpy as np
import mne
from
#typing
from mne import io, Epochs, Evoked, find_events, set_log_level, pick_types
from mne.io import BaseRaw
#internal
from src.core.step_types import DataProcStep, data_proc_step


@data_proc_step(name="Crop", save=True)
def crop_data(
    data,
    stim_channel: str,
    min_buffer: float = - 2.0, 
    max_buffer: float = 2.0
):
    print(f"Cropping full raw file.")
    events = find_events(data, stim_channel=stim_channel, shortest_event=1)
    tmin = data.times[events[0][0]] + min_buffer
    tmin = max(0.0, tmin)  # Ensure tmin >= 0
    tmax = data.times[events[-1][0]] + max_buffer
    tmax = min(tmax, data.times[-1])  # Ensure tmax <= data length
    return data.copy().crop(tmin=tmin, tmax=tmax)
    
class Crop(DataProcStep):
    def __init__(
            self, 
            stim_channel: str,
            save=True, 
            min_buffer: float = - 2.0, 
            max_buffer: float = 2.0
        ):
        """
        
        Args:
            stim_channel (str): find_events() stim channel for min max cropping.
            channel_types (dict): 
        """
        super().__init__(name="Crop", save=save)
        self.stim_channel = stim_channel
        self.min_buffer = min_buffer
        self.max_buffer = max_buffer

    def proc(self, data: BaseRaw):
        print(f"Cropping full raw file.")
        
        # reference stim events for finding earliest and latest event time 
        events = find_events(data, stim_channel=self.stim_channel, shortest_event=1)
        tmin = data.times[events[0][0]] + self.min_buffer
        if tmin < 0.0:
            tmin = 0.0
        tmax = data.times[events[-1][0]] + self.max_buffer
        if tmax > data.times[-1]:
            tmax = data.times[-1] 

class Filter(DataProcStep):
    def __init__(self, l_freq=1.0, h_freq=40.0, save=True):
        super().__init__(name = "Filter", save=save)
        self.l_freq = l_freq
        self.h_freq = h_freq

    def proc(self, data):
        print(f"Filtering {self.l_freq}-{self.h_freq} Hz")
        data.filter(self.l_freq, self.h_freq)
        return data

@data_proc_step(name="set_channels", save=False)
def set_channels(data, ch_dict: dict):
    """
    Sets channels for a channel dict.
    Useful for setting simulus type channels.

    Args:
        ch_dict (dict): Channel type dict, e.g. 
            {
            "UDIO001":"stim"
            }
    """
    # Check if channels exist before attempting to set types
    invalid_channels = set(ch_dict.keys()) - set(data.ch_names)
    if invalid_channels:
        raise ValueError(f"Channels do not exist: {invalid_channels}")

    try:
        raw = data.set_channel_types(ch_dict)
    except (TypeError, ValueError) as e:
        raise ValueError("Trying to set inappropriate channel types, or channels do not exist") from e
    
@data_proc_step(name="zapline_denoise", save=False)
def apply_zapline_denoising(
    data: BaseRaw,
    fline: float = 50.0,
    n_chunks: int = 10,
    spot_sz: int = 7,
    win_sz: int = 12,
    nfft: int = 2048,
    n_iter_max: int = 30,
    mag_only: bool = True,
) -> BaseRaw:
    """
    Apply ZAPLINE iterative line noise removal to MEG data using meegkit.dss.dss_line_iter.

    Operates only on magnetometer channels by default for efficiency and safety.

    Args:
        data (Raw): MNE Raw object (must be preloaded).
        fline (float): Power line frequency (e.g., 50.0 or 60.0 Hz).
        n_chunks (int): Split data into this many chunks to reduce memory load.
        spot_sz (int): Frequency spot size (in FFT bins) around fline.
        win_sz (int): Window size (in samples) for local SNR estimation.
        nfft (int): FFT length for spectral estimation.
        n_iter_max (int): Maximum number of iterations per chunk.
        mag_only (bool): If True, only denoise magnetometer channels.

    Returns:
        Raw: Copy of input with ZAPLINE-applied data.
    """
    raw = data.copy().load_data()  # Ensure preloaded
    raw_data = raw.get_data()

    # Identify magnetometer channels
    if mag_only:
        mag_ix = np.array([i for i, ch_type in enumerate(raw.get_channel_types()) if ch_type == "mag"])
        if len(mag_ix) == 0:
            raise ValueError("No magnetometer channels found for ZAPLINE denoising.")
        data_to_denoise = raw_data[mag_ix]
        ch_names = [raw.ch_names[i] for i in mag_ix]
    else:
        data_to_denoise = raw_data
        mag_ix = np.arange(data_to_denoise.shape[0])
        ch_names = raw.ch_names

    sfreq = raw.info["sfreq"]
    n_channels, n_samples = data_to_denoise.shape

    # Split data into chunks along time axis
    chunks = np.array_split(data_to_denoise, n_chunks, axis=1)
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        # meegkit expects (n_samples, n_channels)
        chunk_T = np.moveaxis(chunk, 0, -1)  # (n_channels, n_t) → (n_t, n_channels)
        cleaned_T, _ = dss_line_iter(
            chunk_T,
            fline=fline,
            sfreq=sfreq,
            spot_sz=spot_sz,
            win_sz=win_sz,
            nfft=nfft,
            n_iter_max=n_iter_max
        )
        # Back to (n_channels, n_t)
        cleaned = np.moveaxis(cleaned_T, -1, 0)
        cleaned_chunks.append(cleaned)

    # Recombine
    cleaned_data_mag = np.hstack(cleaned_chunks)

    # Reconstruct full data
    new_data = raw.get_data()
    if mag_only:
        new_data[mag_ix, :] = cleaned_data_mag
    else:
        new_data = cleaned_data_mag

    # Create new Raw object with cleaned data
    new_raw = raw.copy()
    new_raw._data = new_data  # Safe because we copied

    return new_raw