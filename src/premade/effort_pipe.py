# src/premade/effort_pipe.py
"""
Premade pipeline for effort learning.
"""
from pathlib import Path
from src.core.pipeline import Pipeline, PipelineContext
from src.core.step_types import data_proc_step
from src.steps.data_loader import loadRaw
from src.steps.data_preproc import (
    crop_data, 
    Filter, 
    set_channels, 
    apply_zapline_denoising,
    run_ica_and_save
)
from src.steps.plots import plot_psd
    

def create_effort_pipe():
        
    cwd = Path.cwd()
    print(f"Working dir: {cwd}")
    raw_path = cwd / "src" / "test_data" / "0007/20230523_000000/MEG/001.rest/files/cwt.fif"
    subject_id = "0007"
    output_dir = Path("output")
    context = PipelineContext(
        output_dir= cwd / "output",
        sub_id="0007"
    )
    pipe = Pipeline(
        name="effort",
        context=context
    )
    pipe.add_step(loadRaw(raw_path))
    pipe.add_step(set_channels(ch_dict={
            "UDIO001":"stim",
            "UADC009":"stim",
            "UADC010":"stim"
        }))
    pipe.add_step(crop_data())
    @data_proc_step(name="pick_meg_stim", save=False)
    def pick_channels(data):
        return data.pick(picks=["meg", "stim", "grad", "misc"], exclude=["eeg"])
    pipe.add_step(pick_channels(save=True))
    @data_proc_step(name="grad_comp", save=False)
    def grad_comp(data):
        return data.apply_gradient_compensation(3)
    pipe.add_step(grad_comp())
    pipe.add_step(plot_psd(fmin=0, fmax=120, n_fft=2048))
    pipe.add_step(apply_zapline_denoising(
        fline=50.0,
        n_chunks=10,
        spot_sz=7,
        win_sz=12,
        nfft=2048,
        n_iter_max=30
    ))
    pipe.add_step(plot_psd(fmin=0, fmax=120, n_fft=1024))
    pipe.add_step(run_ica_and_save(
        ica_fpath=output_dir/"ica"/f"{subject_id}-ica.fif",
        n_components=20,
        filt_low=1,
        filt_high=30,
        ))

    return pipe

if __name__ == '__main__':
    
    pipe = create_effort_pipe()
    pipe.show()
    pipe.run()