# src/core/pipeline.py
"""
Main pipeline script that processes data from a single file to end.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import os
import re

@dataclass
class PipelineContext:
    """Used for passing context to child steps"""
    output_dir: Path | str | None  # Can be None initially
    sub_id: str | None = None
    plot_counter: int = 0
    file_counter: int = 0

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def update(self, mapping):
        for key, value in mapping.items():
            setattr(self, key, value)
    
    def copy(self) -> "PipelineContext":
        """Return a shallow copy of this context."""
        return replace(self)

class Pipeline:
    def __init__(self, 
                 name, 
                 context: PipelineContext,
                 auto_load: bool = False
                 ):
        self.name = name
        self.steps = []
        self.context = context
        self.auto_load = auto_load


    def add_step(self, step):
        self.steps.append(step)
        return step
    
    def set_context(self, **kwargs):
        """Set or update context values"""
        self.context.update(kwargs)
        return self

    def run(self, data=None):
        """The pipeline loops over each step."""
        self.context.plot_counter = 0# reset plot counter
        self.context.file_counter = 0 # reset plot counter
        if isinstance(self.context.output_dir, str):
            self.context.output_dir = Path(self.context.output_dir)

        # Find the most recent file that matches any step name
        most_recent_file = None
        step_at_which_to_resume = None
        
        if self.auto_load:
            files_dir = self.context.output_dir / "files"
            if files_dir.exists():
                files = list(files_dir.glob('*'))
                # match and extract the fXY pattern (f followed by 2 digits)
                f_pattern = re.compile(r'f(\d{2})')
                max_num = -1
                
                for file_path in files:
                    match = f_pattern.search(file_path.name)
                    if match:
                        num = int(match.group(1))
                        # Check if this file matches any of the step names
                        for j, step in enumerate(self.steps):
                            if step.name.lower() in file_path.name.lower():
                                if num > max_num:
                                    max_num = num
                                    most_recent_file = file_path
                                    step_at_which_to_resume = j
                                break  # Found a matching step, move to next file
                
                if max_num >= 0:
                    self.context.file_counter = max_num + 1
                    print(f"Found most recent file: {most_recent_file} for step at index {step_at_which_to_resume}")
                else:
                    self.context.file_counter = 0
            else:
                self.context.file_counter = 0

        current_data = data
        for i, step in enumerate(self.steps):
            if not step.active and step.required:
                raise RuntimeError(f"Required step '{step.name}' is inactive!")
            if not step.active:
                print(f"Skipping {step.name}")
                continue

            # Check if we should resume from a saved file
            if self.auto_load and most_recent_file and step_at_which_to_resume is not None:
                if i == step_at_which_to_resume:
                    # This is the step that matches the most recent file, load the data
                    print(f"Auto-Loading most recent data from {most_recent_file} for step {step.name}")
                    try:
                        from src.steps.data_loader import loadRaw  # Adjust import path as needed
                        raw_loader = loadRaw(most_recent_file)
                        # Inject context if the loader supports it
                        if hasattr(raw_loader, "set_context"):
                            raw_loader.set_context(self.context)
                        current_data = raw_loader.run()
                    except ImportError:
                        print(f"Could not import loadRaw step, running step normally")
                        if hasattr(step, "set_context"):
                            step.set_context(self.context)
                        current_data = step.run(current_data)
                    except Exception as e:
                        print(f"Error loading data from {most_recent_file}: {e}")
                        # Fall back to running the step normally
                        if hasattr(step, "set_context"):
                            step.set_context(self.context)
                        current_data = step.run(current_data)
                elif i < step_at_which_to_resume:
                    # Skip steps that were already completed
                    print(f"Skipping {step.name} (already completed based on saved file)")
                    continue
                else:
                    # Run steps that come after the loaded step
                    if hasattr(step, "set_context"):
                        step.set_context(self.context)
                    print(f"*Step {i}: {step.name} ...")
                    current_data = step.run(current_data)
            else:
                # Normal execution when auto_load is not enabled or no matching file found
                # Inject context dynamically if the step supports it
                if hasattr(step, "set_context"):
                    step.set_context(self.context)
                
                print(f"*Step {i}: {step.name} ...")
                current_data = step.run(current_data)
        return current_data
    
    def copy(self):
        """Create a copy of this pipeline"""
        new_pipeline = Pipeline(self.name, self.context.copy())
        new_pipeline.steps = [step for step in self.steps]  # Shallow copy of steps
        return new_pipeline
    
    def show(self):
        """Create a diagram of the step-flow."""
        print("-"*70)
        print(f"# Pipeline")
        print(f"Name: {self.name}")
        n_steps = len(self.steps)
        print(f"N steps: {n_steps}{os.linesep}")

        for i, step in enumerate(self.steps):
            print(f"{i}: {step.name}")
            if i < n_steps-1:
                arr = f"     |{os.linesep}     V"
                print(arr)

        print("-"*70)