"""
Main pipeline script that processes data from a single file to end.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import os

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
    def __init__(self, name, context: PipelineContext):
        self.name = name
        self.steps = []
        self.context = context


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

        current_data = data
        for step in self.steps:
            if not step.active and step.required:
                raise RuntimeError(f"Required step '{step.name}' is inactive!")
            if not step.active:
                print(f"Skipping {step.name}")
                continue
            
            # Inject context dynamically if the step supports it
            if hasattr(step, "set_context"):
                step.set_context(self.context)
            
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