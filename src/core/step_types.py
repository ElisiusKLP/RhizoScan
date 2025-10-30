from pathlib import Path
from src.core.step import Step

class DataProcStep(Step):
    """Mandatory step that processes data and passes it forward."""
    def __init__(self, name, active=True):
        super().__init__(name, active=active, required=True)

    def run(self, data):
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
            out_dir = Path(self.context["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Use counter for sequential naming
            counter = self.context.get("plot_counter", 0)
            filename = f"{self.context["sub_id"]}_{counter:02d}_{self.name}.png"
            filepath = out_dir / filename
            fig.savefig(filename)
            print(f"Saved plot to {filename}")

        if self.show:
            fig.show()

        return data

    def plot(self, data):
        raise NotImplementedError("PlotStep must implement plot()")
