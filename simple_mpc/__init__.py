"""Model-agnostic gradient-descent MPC.

Supports EulerianModelWrapper (occupancy-grid) and PropNetDiffDenModel
(particle-cloud GNN).  Model-specific logic is encapsulated in ModelAdapters
(see ``simple_mpc.adapters``).

Public API
----------
run_simple_mpc            – Run MPC and return a result dict compatible with
                            env.step_subgoal_ptcl().
load_simple_config        – Load the simple-MPC YAML config file.
make_adapter              – Factory: return the right ModelAdapter for a model.
benchmark_push_throughput – Measure push-model GPU throughput.
"""

from simple_mpc.mpc import run_simple_mpc, load_simple_config
from simple_mpc.adapters import make_adapter
from simple_mpc.benchmark import benchmark_push_throughput

__all__ = [
    'run_simple_mpc',
    'load_simple_config',
    'make_adapter',
    'benchmark_push_throughput',
]
