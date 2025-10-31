from pathlib import Path
import warnings
from src.core.step import Step
from mne.io import BaseRaw
from mne import Epochs, Evoked

class DataProcStep(Step):
    """Mandatory step that processes data and passes it forward."""
    def __init__(self, name, active=True, save=False):
        super().__init__(name, active=active, required=True)
        self.active = active
        self.save = save


    def run(self, data):
        if self.save and self.context is not None:
            # Create output dir
            out_dir = Path(self.context["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectory for files (optional)
            files_sub_dir = out_dir / "files"
            files_sub_dir.mkdir(parents=True, exist_ok=True)

            # Use counter for sequential mapping
            counter = self.context.get("file_counter", 0)
            filename = f"{self.context['sub_id']}_{counter:02d}_{self.name}.fif"
            filepath = files_sub_dir / filename

            # Check type and save appropriately
            if isinstance(data, (BaseRaw, Epochs, Evoked)):
                data.save(filepath, overwrite=True)
                print(f"Saved MNE object to {filepath}")
            else:
                warnings.warn(
                    f"[{self.name}] Could not save {filename}: "
                    f"invalid data type {type(data).__name__}.",
                    UserWarning
                )
                return data  # gracefully exit, no save

            # Increment counter in context
            self.context["file_counter"] = counter + 1

        return data


    def proc(self, data):
        raise NotImplementedError("DataProcStep must implement run()")


class PlotStep(Step):
    """Optional step that visualizes or logs data; does not modify it."""
    def __init__(self, name, active=True, required=True, save=False, show=False):
        super().__init__(name, active=active, required=required)
        self.save = save
        self.show = show

    def run(self, data):
        if not self.active:
            print(f"[{self.name}] skipped (inactive)")
            return data
        
        fig = self.plot(data)

        if self.save and self.context is not None:
            # create output dir
            out_dir = Path(self.context["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # create plot sub dir
            plot_sub_dir = out_dir / "plots"
            plot_sub_dir.mkdir(parents=True, exist_ok=True)
            # Use counter for sequential naming
            counter = self.context.get("plot_counter", 0)
            filename = f"{self.context["sub_id"]}_{counter:02d}_{self.name}.png"
            filepath = out_dir / filename
            fig.savefig(filename)
            print(f"Saved plot to {filename}")

            # Increment counter in context
            self.context["plot_counter"] = counter + 1

        if self.show:
            fig.show()

        return data

    def plot(self, data):
        raise NotImplementedError("PlotStep must implement plot()")
