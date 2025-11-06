# src/steps/data_preproc.pyi
from pathlib import Path
from typing import Dict, Optional
from src.core.step_types import DataProcStep  # ← adjust if DataProcStep is elsewhere


# Signature: only the arguments the USER passes (not 'data'!)

def set_channels(ch_dict: Dict[str, str], *, save: bool = False) -> DataProcStep: ...

def crop_data(tmin: float = 0.0, tmax: Optional[float] = None, *, save: bool = False) -> DataProcStep: ...

def Filter(
        l_freq: Optional[float] = None, 
        h_freq: Optional[float] = None, 
        *, 
        save: bool = False
    ) -> DataProcStep: ...

def apply_zapline_denoising(
    fline: float = 50.0,
    n_chunks: int = 10,
    spot_sz: int = 7,
    win_sz: int = 12,
    nfft: int = 2048,
    n_iter_max: int = 30,
    mag_only: bool = True,
) -> DataProcStep: ...
    
def run_ica_and_save(
    ica_fpath: Path,
    n_components: int = 20,
    filt_low: float = 1.0,
    filt_high: float = 30.0,
    method: str = "fastica",
    random_state: int = 42,
) -> DataProcStep: ...