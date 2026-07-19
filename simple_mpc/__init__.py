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

Oracle (Genesis-as-model) MPC — see docs/oracle_mpc_plan.md. Not re-exported
here (unlike the above) so that ``import simple_mpc`` doesn't pull in Genesis
for callers who only need the learned-model path; import directly:
    from simple_mpc.oracle_mpc import run_oracle_mpc, load_oracle_config
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.sampling_optimizers import make_sampling_optimizer
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
