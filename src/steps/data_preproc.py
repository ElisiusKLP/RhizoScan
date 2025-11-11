# src/steps/data_preproc.py
"""
Implementations of "data preprocessing steps".

Returns:
    data
"""
from pathlib import Path
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from meegkit.dss import dss_line_iter
#typing
from mne import io, Epochs, Evoked, find_events, set_log_level, pick_types
from mne.io import BaseRaw, RawArray
#internal
from src.core.step_types import DataProcStep, data_proc_step
#logging
import logging
logger = logging.getLogger(__name__)

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
        ValueError(f"Channels do not exist: {invalid_channels}")

        print("Here are the avaliable channel types:")
        print(print(set(data.get_channel_types())))
        print("Looking for existing stim channels...")
        stim_ch_names = [ch for ch, ch_type in zip(data.ch_names, data.get_channel_types()) if ch_type == 'stim']
        print(stim_ch_names)

        raise

    try:
        raw = data.set_channel_types(ch_dict)
        
    except (TypeError, ValueError) as e:
        raise ValueError("Trying to set inappropriate channel types, or channels do not exist") from e
    
    return raw

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
    Chunks data into n_chunks for computational efficiency.

    Operates only on magnetometer channels by default for efficiency and safety.
    Defaults to mag_only as gradiometer power noise act differently in frequencies.

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
    info = raw.info
    raw_data: np.ndarray = raw.get_data()

    # Identify magnetometer channels
    
    if mag_only:
        mag_ix = np.array([
            i for i, ch_type in enumerate(raw.get_channel_types()) 
            if ch_type == "mag"
        ])
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
    logger.debug(f"cleaned_data_mag {cleaned_data_mag[0:20]}")

    # Reconstruct full data
    new_data: np.ndarray = raw.get_data()
    if mag_only:
        new_data[mag_ix, :] = cleaned_data_mag
    else:
        new_data = cleaned_data_mag

    # Create new Raw object with cleaned data
    new_raw = RawArray(
        new_data,
        info,
        first_samp=raw.first_samp
    )

    return new_raw

@data_proc_step(name="run_ica_side_effect", save=False)
def run_ica_and_save(
    data: BaseRaw,
    ica_fpath: Path,
    n_components: int = 20,
    filt_low: float = 1.0,
    filt_high: float = 30.0,
    method: str = "fastica",
    random_state: int = 42,
    *,
    context=None
) -> BaseRaw:
    """
    Run ICA and save to disk, returning the filtered Raw object.
    Useful when pipeline must return Raw.

    Args:
        data (Raw): Input data.
        filt_low (float or None): Low cutoff for bandpass filter (Hz). If None, skip.
        filt_high (float or None): High cutoff for bandpass filter (Hz). If None, skip.
        method (str): ICA algorithm (e.g., "fastica", "picard").
        random_state (int): For reproducibility.

    Returns:
        Raw: Bandpass-filtered copy of input (used for ICA fitting).
    """
    raw_copy = data.copy().load_data()
    raw_copy.filter(filt_low, filt_high, verbose=False, n_jobs=1)

    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    ica.fit(raw_copy)
    if not isinstance(context.output_dir, Path):
        raise TypeError(f"context.out_dir is not Path type: {type(context.output_dir)}")
    ica_dir = context.output_dir / "ica"
    ica_dir.mkdir(parents=True, exist_ok=True)
    ica_fname = f"{context.sub_id}-ica.fif"
    ica_fpath = ica_dir / ica_fname
    ica.save(ica_fpath, overwrite=True)
    logger.info(f"saved ica to {ica_fpath}")
    
    # Return original raw
    return data

@data_proc_step(name="IcaCheck", save=False)
def ica_check(
    data: BaseRaw,
    crop_min: int = 100,
    crop_max: int = 300,
    filter_min: int = 1,
    filter_max: int = 20,
    *,
    context=None
):
    
    # crop to reduce memory overhead for visualisation - filter in a artifact range
    raw = data.crop(tmin=crop_min, tmax=crop_max).filter(filter_min, filter_max)
    # setup ica_path
    ica_path = context.output_dir / "ica" / f"{context.sub_id}-ica.fif"
    ica = read_ica(ica_path)

    temp_path = context.output_dir / "ica_check"
    temp_path.mkdir(parents=True, exist_ok=True)

    # Display ICA component topographies and force user interaction
    print("Displaying ICA Component Topographies...")
    ica_comps = ica.plot_components(show=True)  # Force display with blocking behavior
    
    # Display ICA source time series and force user interaction  
    print("Displaying ICA Source Time Series...")
    ica_sources = ica.plot_sources(raw, block=True, theme="dark", show=True)  # Force display with blocking

    comps_path = temp_path / "components.png"
    source_path = temp_path / "source.png"
    ica_comps.savefig(comps_path)
    ica_sources.savefig(source_path)

    ica.find_bads_ecg(raw)
    ica.find_bads_muscle(raw)

    print(f"Automatic exclusion excluded: {ica.exclude}")

    # Get total number of components
    n_components = ica.n_components_
    available_components = list(range(n_components))
    
    print(f"\nICA Components for {context.sub_id}")
    print(f"Inspect plots in directory: {temp_path}")
    print(f"Available components: {available_components}")
    print("Enter component numbers to exclude (comma-separated, e.g., '0,3,5'):")
    
    while True:
        try:
            user_input = input("Components to exclude (or 'done' to continue): ").strip()
            
            if user_input.lower() == 'done':
                break
            elif user_input == '':
                continue
            
            # Parse component numbers
            excluded_comps = [int(x.strip()) for x in user_input.split(',')]
            
            # Validate component numbers
            invalid_comps = [comp for comp in excluded_comps if comp not in available_components]
            if invalid_comps:
                print(f"Invalid component numbers: {invalid_comps}. Available: {available_components}")
                continue
            
            # Add to existing exclusions
            ica.exclude = sorted(set(ica.exclude + excluded_comps))
            print(f"Currently excluded components: {ica.exclude}")
            
        except ValueError:
            print("Please enter valid integers separated by commas (e.g., '0,3,5') or 'done'")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return data

    print(f"{context.sub_id}", "final excluded comps: ", ica.exclude)

    new_ica = ica.copy()
    new_ica_path = context.output_dir / "ica" / f"{context.sub_id}_clean-ica.fif"
    new_ica.save(new_ica_path, overwrite=True)

    return data