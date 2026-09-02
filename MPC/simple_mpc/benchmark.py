"""GPU throughput benchmark for models via their adapters."""

import time
import numpy as np
import torch


def benchmark_adapter_step(
    adapter,
    wkspc_w: float = 5.0,
    batch_sizes: list = None,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = 'cuda',
) -> None:
    """
    Benchmark the adapter's predict_step throughput for a range of batch sizes.
    Model-agnostic: works with any adapter (GNN, Eulerian, etc.).

    This function creates dummy states and actions, then times the adapter's
    predict_step() method. It tests the exact code path used in MPC, making it
    the most meaningful benchmark for the optimizer.

    Output columns
    --------------
    batch  – number of parallel candidates (≈ n_sample in the MPC config)
    ms/step – wall-clock milliseconds for one full forward pass
    cand/s  – throughput in candidates per second
    ms/iter – expected time per Adam iteration at this batch size (forward only)

    Parameters
    ----------
    adapter : ModelAdapter
        GNNAdapter, EulerianAdapter, or any adapter with predict_step()
    wkspc_w : float
        Workspace half-width (used for action range)
    batch_sizes : list, optional
        Batch sizes to test; defaults to [8, 16, 32, ..., 1024]
    n_warmup : int
        Warmup iterations (for JIT/cache warmup)
    n_runs : int
        Timed runs per batch size
    device : str
        PyTorch device ('cuda' or 'cpu')

    Usage
    -----
    >>> from simple_mpc.adapters import make_adapter
    >>> from simple_mpc.benchmark import benchmark_adapter_step
    >>> adapter = make_adapter(model, env, subgoal, cfg, cam_params, device='cuda')
    >>> benchmark_adapter_step(adapter, wkspc_w=5.0)
    """
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 50, 64, 100, 128, 200, 256, 512, 1024]

    # Create a dummy reference state to understand adapter state shape
    # (Use expand_state to match MPC's state creation)
    dummy_state = adapter.expand_state(
        torch.randn(1, *adapter._get_example_state_shape(), device=device),
        n_sample=1
    ).detach()

    act_lo = np.array([-wkspc_w, -wkspc_w, -wkspc_w * 0.85, -wkspc_w * 0.85],
                      dtype=np.float32)
    act_hi = -act_lo

    model_name = type(adapter).__name__
    state_repr = f"shape {tuple(dummy_state.shape[1:])}"

    print(f"\n── adapter step throughput benchmark ──────────────────────────────")
    print(f"   adapter: {model_name}")
    print(f"   state  : {state_repr}   device: {device}")
    print(f"   warmup : {n_warmup}   timed runs: {n_runs}")
    print(f"{'batch':>7}  {'ms/step':>9}  {'cand/s':>10}  {'ms/iter':>9}  {'status':>6}")
    print(f"{'─'*7}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*6}")

    for B in batch_sizes:
        try:
            # Create batch state using the adapter's expand_state
            state_batch = adapter.expand_state(dummy_state, n_sample=B).detach()

            # Create random actions
            act_np = np.random.uniform(act_lo, act_hi, (B, 4)).astype(np.float32)
            act_batch = torch.tensor(act_np, device=device)

            # Warmup (including JIT/cuBLAS cache warmup)
            for _ in range(n_warmup):
                with torch.no_grad():
                    adapter.predict_step(state_batch, act_batch)
            if device == 'cuda':
                torch.cuda.synchronize()

            # Timed runs
            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    adapter.predict_step(state_batch, act_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1e3 / n_runs

            cand_per_s = B / (elapsed_ms * 1e-3)
            print(f"{B:>7}  {elapsed_ms:>9.2f}  {cand_per_s:>10.0f}  "
                  f"{elapsed_ms:>9.2f}     ok")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = 'OOM' if 'out of memory' in str(exc).lower() else 'ERR'
            print(f"{B:>7}  {'':>9}  {'':>10}  {'':>9}  {msg:>6}")
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f"──────────────────────────────────────────────────────────────────\n")


def benchmark_push_throughput(
    model_dy,
    wkspc_w: float = 5.0,
    batch_sizes: list = None,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = 'cuda',
) -> None:
    """
    Benchmark the push model (predict_one_step_occ) throughput for a range of
    batch sizes.  Run this after constructing the EulerianModelWrapper to check
    how many candidates/second the GPU can process.

    Only works for Eulerian models (occupancy-based). GNN models use a different
    state representation (particles) and cannot be benchmarked with this function.

    Output columns
    --------------
    batch  – number of parallel candidates (≈ n_sample in the MPC config)
    ms/step – wall-clock milliseconds for one full forward pass of the push model
    cand/s  – throughput in candidates per second
    ms/iter – expected time per Adam iteration at this batch size (forward only)

    Usage
    -----
    >>> from simple_mpc import benchmark_push_throughput
    >>> benchmark_push_throughput(model_dy, wkspc_w=5.0)
    """
    # Only benchmark Eulerian models (occupancy-based)
    if not hasattr(model_dy, 'predict_one_step_occ'):
        print(f"\n── push throughput benchmark ────────────────────────────────────")
        print(f"   Skipping benchmark: model type '{type(model_dy).__name__}' does not support")
        print(f"   occupancy-based benchmarking (only EulerianModelWrapper is supported).")
        print(f"────────────────────────────────────────────────────────────────────\n")
        return
    
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 50, 64, 100, 128, 200, 256, 512, 1024]

    Nx, Ny = model_dy.grid_res
    act_lo = np.array([-wkspc_w, -wkspc_w, -wkspc_w * 0.85, -wkspc_w * 0.85],
                      dtype=np.float32)
    act_hi = -act_lo

    print(f"\n── push throughput benchmark ────────────────────────────────────")
    print(f"   model : {type(model_dy.user_model).__name__}")
    print(f"   grid  : {Nx} × {Ny}   device: {device}")
    print(f"   warmup: {n_warmup}   timed runs: {n_runs}")
    print(f"{'batch':>7}  {'ms/step':>9}  {'cand/s':>10}  {'ms/iter':>9}  {'status':>6}")
    print(f"{'─'*7}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*6}")

    for B in batch_sizes:
        try:
            occ = torch.rand(B, Nx, Ny, device=device)
            act_np = np.random.uniform(act_lo, act_hi, (B, 4)).astype(np.float32)
            act = torch.tensor(act_np, device=device)

            # warmup (including JIT / cuBLAS plan caches)
            for _ in range(n_warmup):
                with torch.no_grad():
                    model_dy.predict_one_step_occ(occ, act)
            if device == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    model_dy.predict_one_step_occ(occ, act)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1e3 / n_runs

            cand_per_s = B / (elapsed_ms * 1e-3)
            print(f"{B:>7}  {elapsed_ms:>9.2f}  {cand_per_s:>10.0f}  "
                  f"{elapsed_ms:>9.2f}     ok")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = 'OOM' if 'out of memory' in str(exc).lower() else 'ERR'
            print(f"{B:>7}  {'':>9}  {'':>10}  {'':>9}  {msg:>6}")
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f"─────────────────────────────────────────────────────────────────\n")
