"""
Main pipeline script that processes data from a single file to end.
"""
from dataclasses import dataclass
from pathlib import Path

class Pipeline:
    def __init__(self, name, context=None):
        self.name = name
        self.steps = []
        self.context = context or {}

    def add_step(self, step):
        self.steps.append(step)
        return step

    def run(self, data=None):
        """The pipeline loops over each step."""
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

@dataclass
class PipelineContext:
    output_dir: Path
    sub_id: str
    plot_counter: int = 0