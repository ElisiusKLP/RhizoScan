# src/steps/data_preproc.pyi

from typing import Dict, Optional
from src.core.step_types import DataProcStep  # ← adjust if DataProcStep is elsewhere

# Signature: only the arguments the USER passes (not 'data'!)

def set_channels(ch_dict: Dict[str, str], *, save: bool = False) -> DataProcStep: ...

def crop_data(tmin: float = 0.0, tmax: Optional[float] = None, *, save: bool = False) -> DataProcStep: ...

def Filter(l_freq: Optional[float] = None, h_freq: Optional[float] = None, *, save: bool = False) -> DataProcStep: ...