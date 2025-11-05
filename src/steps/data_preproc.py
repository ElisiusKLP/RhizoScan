"""
Implementations of "data preprocessing steps".

Returns:
    data
"""
import mne
from src.core.step_types import DataProcStep, data_proc_step

from mne import io, Epochs, Evoked, find_events, set_log_level, pick_types
from mne.io import BaseRaw

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
def set_ch(data, ch_dict: dict):
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