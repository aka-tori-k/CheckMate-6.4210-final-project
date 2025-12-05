# src/simulation_setup/initialize_simulation.py
from pydrake.all import (
    LeafSystem,
    BasicVector
)

import numpy as np

class ModifiableTrajectorySource(LeafSystem):
    """
    A simple LeafSystem that outputs the value of a PiecewisePolynomial (full-state)
    at the current simulation time. The stored trajectory can be swapped at runtime
    by calling set_trajectory(traj).
    """

    def __init__(self, vector_size):
        super().__init__()
        self._vector_size = int(vector_size)
        self._traj = None  # will hold a PiecewisePolynomial whose output length == vector_size
        # no inputs, one vector output
        self.DeclareVectorOutputPort("value", BasicVector(self._vector_size), self.CalcOutput)

    def CalcOutput(self, context, output):
        t = context.get_time()
        if (self._traj is None) or (t < self._traj.start_time()) or (t > self._traj.end_time()):
            # Output zeros if no trajectory or out of traj time range
            output.SetFromVector(np.zeros(self._vector_size))
            return
        val = self._traj.value(t).flatten()
        if val.size != self._vector_size:
            raise RuntimeError(f"Trajectory produces length {val.size} but expected {self._vector_size}")
        output.SetFromVector(val)

    def set_trajectory(self, traj):
        """Set a new PiecewisePolynomial trajectory (value(t) length must match vector_size)."""
        if traj is None:
            self._traj = None
            return
        # Quick length check on first value
        if traj.value(traj.start_time()).size != self._vector_size:
            raise RuntimeError(f"Provided trajectory value length {traj.value(traj.start_time()).size} != expected {self._vector_size}")
        self._traj = traj
