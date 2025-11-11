# src/core/step_types.py
"""
These are the different types of steps we can create.
They each have slightly different attrbutes in their required arguments.
"""
from functools import wraps
from inspect import signature
from typing import Callable, Any
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
        # run a data processing step
        if not self.active:
            print(f"[{self.name}] skipped (inactive)")
            return data
        
        data = self.proc(data)

        if self.save and self.context is not None:
            # Create output dir
            out_dir = Path(self.context["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectory for files (optional)
            files_sub_dir = out_dir / "files"
            files_sub_dir.mkdir(parents=True, exist_ok=True)

            # Use counter for sequential mapping
            counter = self.context.file_counter
            filename = f"{self.context.sub_id}_f{counter:02d}_{self.name}.fif"
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
            self.context.file_counter = counter + 1

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
        assert fig is not None, "fig is not returned from PlotStep plot() implementation"

        if self.save and self.context is not None:
            # create output dir
            out_dir = self.context.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            # create plot sub dir
            plot_sub_dir = out_dir / "plots"
            plot_sub_dir.mkdir(parents=True, exist_ok=True)
            # Use counter for sequential naming
            counter = self.context.plot_counter
            filename = f"{self.context.sub_id}_p{counter:02d}_{self.name}.png"
            filepath = plot_sub_dir / filename
            fig.savefig(filepath)
            print(f"Saved plot to {filepath}")

            # Increment counter in context
            self.context.plot_counter = counter + 1

        if self.show:
            fig.show()

        return data

    def plot(self, data):
        raise NotImplementedError("PlotStep must implement plot()")


# -- DECORATOR WRAPPERS (for increased usability)
def data_proc_step(name: str, save: bool = False, active: bool = True):
    """Decorator to convert a function into a DataProcStep."""
    def decorator(func: Callable[..., Any]):  # Allow any number of parameters
        @wraps(func) # wraps the function
        def step_factory(*args, save=save, active=active, **kwargs):
            # Create the actual step class
            class FunctionStep(DataProcStep):
                def __init__(self, func_args=None, func_kwargs=None, save=save, active=active):
                    # initializes DataProcStep with the arguments such that proc() runs the wraped func
                    super().__init__(name=name, save=save, active=active)
                    self.func = func
                    self.func_args = func_args or ()
                    self.func_kwargs = func_kwargs or {}
                
                def proc(self, data):
                    print(f"Running {name}")

                    # Get the function signature
                    sig = signature(self.func)
                    
                    # Prepare kwargs to pass to the function
                    call_kwargs = dict(self.func_kwargs)
                    
                    # If the function accepts 'context', inject self.context
                    if 'context' in sig.parameters:
                        call_kwargs['context'] = self.context  # <-- this is the magic!
                    
                    return self.func(data, *self.func_args, **call_kwargs)
            
            # Return an instance of the step with the current arguments
            return FunctionStep(args, kwargs, save=save, active=active)
        
        return step_factory
    return decorator

def plot_step(name: str, save: bool = True, show: bool = False, active: bool = True):
    """Decorator to convert a function into a PlotStep."""
    def decorator(func: Callable[..., Any]):  # Allow any number of parameters
        @wraps(func)
        def step_factory(*args, save=save, show=show, active=active, **kwargs):
            # Create the actual step class
        # This creates a NEW class that INHERITS from PlotStep
            class FunctionPlotStep(PlotStep):
                def __init__(self, func_args=None, func_kwargs=None, save=save, show=show, active=active):
                    super().__init__(name=name, save=save, show=show, active=active)
                    self.func = func
                    self.func_args = func_args or ()
                    self.func_kwargs = func_kwargs or {}
            
                def plot(self, data):
                    print(f"Plotting {self.name}")
                    
                    # Get the function signature
                    sig = signature(self.func)
                    
                    # Prepare kwargs to pass to the function
                    call_kwargs = dict(self.func_kwargs)
                    
                    # If the function accepts 'context', inject self.context
                    if 'context' in sig.parameters:
                        call_kwargs['context'] = self.context  # <-- this is the magic!
                    
                    return self.func(data, *self.func_args, **call_kwargs)
                
            # Return an instance of the step with the current arguments
            return FunctionPlotStep(args, kwargs, save=save, show=show, active=active)
        
        return step_factory
    return decorator