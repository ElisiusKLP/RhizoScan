# src.utils.plotting
# Licensed code from Maciek Szul
import numpy as np
import matplotlib.pyplot as plt

def create_psd_plot(freqs, raw_psds, title):
    f, ax = plt.subplots(1, 1, figsize=(9,4))

    ax.plot(freqs, np.log10(raw_psds.T), lw=0.5);
    ax.set_xlim(0.0, 119.0)
    ax.set_ylabel("log10(PSD)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title(title)

    plt.tight_layout()
    return f