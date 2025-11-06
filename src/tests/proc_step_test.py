# proc_step_test.py
import mne
from pathlib import Path
from src.steps.data_preproc import Filter
from src.steps.data_loader import loadRaw

test_file = "/home/elisius/projects/RhizoScan/src/tests/test_files/cwt.fif"

def test_load_raw(test_file: str):
    """Test that Load returns a Raw object."""
    test_file = test_file  # small test file
    assert Path(test_file).exists(), "Test data file does not exist"

    file_path = Path(test_file)
    step = loadRaw(filepath=file_path)
    raw = step.run(None)

    assert isinstance(raw, mne.io.BaseRaw), "LoadMEG did not return a Raw object"
    print("✅ LoadMEG test passed")

def test_filter_meg(test_file: str):
    """Test that FilterMEG correctly filters the data."""
    test_file = test_file
    raw = mne.io.read_raw_fif(test_file, preload=True)

    step = Filter(l_freq=20.0, h_freq=40.0)
    filtered = step.run(raw)

    assert isinstance(filtered, mne.io.BaseRaw), "Filter did not return a Raw object"
    # Optional: check that the data actually changed
    assert (filtered.get_data() != raw.get_data()).any(), "Filter did not modify the data"
    print("✅ Filter test passed")

if __name__ == "__main__":
    test_load_raw(test_file)
    test_filter_meg(test_file)