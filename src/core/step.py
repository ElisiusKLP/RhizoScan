"""
Base step class that act as a node which can be activate or not and connect to other steps.
"""

class Step:
    def __init__(
            self, 
            name: str, 
            active: bool = True, 
            required: bool = True):
        self.name = name
        self.active = active
        self.required = required
        self.context = None
        self.next_steps: list[Step] = []

    def set_context(self, context):
        self.context = context

    def connect(self, next_step: "Step") -> "Step":
        """Connect current step's output to another step."""
        self.next_steps.append(next_step)
        return next_step

    def run(self, data):
        raise NotImplementedError("Subclasses must implement run().")
