"""
Plot implementation of the PlotStep function.
"""
import mne
import matplotlib.pyplot as plt

class PlotPSD(PlotStep):
    def __init__(self, fmin=1.0, fmax=60.0, active=True, save=False):
        super().__init__("PlotPSD", active=active, save=save)
        self.fmin = fmin
        self.fmax = fmax

    def plot(self, raw):
        """
        raw: mne.io.Raw object
        Returns a matplotlib figure.
        """
        print(f"Plotting PSD ({self.fmin}-{self.fmax} Hz)")
        fig = raw.plot_psd(fmin=self.fmin, fmax=self.fmax, show=False)  # returns fig object
        title = f"Sub {self.context["sub_id"]} PSD ({self.fmin}-{self.fmax} Hz)"
        fig.suptitle(title)
        return fig