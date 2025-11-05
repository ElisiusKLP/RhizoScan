"""
Premade pipeline for effort learning.

"""
from pathlib import Path
from src.core.pipeline import Pipeline, PipelineContext
from src.steps.data_loader import loadRaw
from src.steps.data_preproc import crop_data, Filter, set_channels
from src.core.step_types import data_proc_step

def create_effort_pipe():
        
    cwd = Path.cwd()
    print(f"Working dir: {cwd}")
    raw_path = cwd / "src" / "test_data" / "0007/20230523_000000/MEG/001.rest/files/cwt.fif"
    
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
    

    return pipe

if __name__ == '__main__':
    
    pipe = create_effort_pipe()
    pipe.show()
    pipe.run()