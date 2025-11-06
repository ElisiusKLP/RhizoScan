# src/steps/plots.pyi

from typing import Dict, Optional
from src.core.step_types import PlotStep  # ← adjust if DataProcStep is elsewhere

# Signature: only the arguments the USER passes (not 'data'!)
def plot_psd(
        fmin: int = 0, 
        fmax: int = 60, 
        n_fft: int = 1024, 
        average: str = "mean", 
        *, 
        save: bool = True, 
        show: bool = False 
    ) -> PlotStep: ...