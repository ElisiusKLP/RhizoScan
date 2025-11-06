"""

"""
from pathlib import Path
import mne
from src.core.step_types import DataProcStep

class loadRaw(DataProcStep):
    def __init__(self, filepath: Path):
        """
        Loads Raw mne files in a catch-all manner.

        Args:
            filepath (Path): path to a mne.BaseRaw object.
        """
        super().__init__("loadRaw")
        self.filepath = filepath

    def run(self, data=None):
        print(f"Loading MEG data: {self.filepath}")
        raw = mne.io.read_raw(self.filepath, preload=True)
        return raw