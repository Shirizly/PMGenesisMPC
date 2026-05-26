"""
Optimal-transport-based action initialiser for MPC.

At each MPC step, call OTPlannerSparse.solve(source, goal) to compute a
sparse-support Sinkhorn OT plan between the current granule distribution and
the goal.  The resulting displacement field and divergence map expose where a
push action would achieve the most coherent mass transport.

Grid coordinate convention: coords are (col, row) = (x, y), origin at
lower-left, matching matplotlib's origin='lower' and the row/col layout of
the 2-D density arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ot


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OTResult:
    """All outputs produced by a single OTPlannerSparse.solve() call.

    Attributes
    ----------
    src_coords : (n_src, 2)  float  – (col, row) of each occupied source cell
    tgt_coords : (n_tgt, 2)  float  – (col, row) of each occupied goal cell
    v_sparse   : (n_src, 2)  float  – displacement vector per source cell
    vectors_2d : (n, n, 2)   float  – dense displacement field; zero outside source support
    div        : (n, n)      float  – signed divergence (zero outside support)
    div_mag    : (n, n)      float  – |div|; small values = coherent/laminar flow
    source_mask: (n, n)      bool   – True where source distribution is non-zero
    timings    : dict[str, float]   – wall-clock ms for each sub-step and total
    """
    src_coords  : np.ndarray
    tgt_coords  : np.ndarray
    v_sparse    : np.ndarray
    vectors_2d  : np.ndarray
    div         : np.ndarray
    div_mag     : np.ndarray
    source_mask : np.ndarray
    timings     : Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------

class OTPlannerSparse:
    """Sparse-support Sinkhorn OT planner for MPC action initialisation.

    Only cells that actually contain mass are included in the OT problem.
    This avoids the fabricated displacement vectors that arise from adding a
    uniform floor (eps_floor) to make the dense problem non-degenerate.

    Parameters
    ----------
    grid_size : int   – side length n of the square occupancy grid
    reg       : float – Sinkhorn entropy regularisation (higher = more diffuse plan)
    verbose   : bool  – print per-step timing when True
    """

    def __init__(self, grid_size: int, reg: float = 0.02, verbose: bool = False):
        self.n       = grid_size
        self.reg     = reg
        self.verbose = verbose
        self._coords = self._build_coords(grid_size)   # (n*n, 2) col-first, precomputed once

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, source: np.ndarray, goal: np.ndarray) -> OTResult:
        """Run the full OT pipeline for one MPC step.

        Parameters
        ----------
        source : (n, n) float/int – current granule density on the grid
        goal   : (n, n) float/int – target density on the grid

        Returns
        -------
        OTResult – all intermediate and final products (see OTResult docstring)
        """
        timings: Dict[str, float] = {}
        t_total = _tic()

        src_coords, tgt_coords, a, b = self._extract_support(source, goal, timings)
        C                            = self._build_cost_matrix(src_coords, tgt_coords, timings)
        P, log                       = self._run_sinkhorn(a, b, C, timings)
        v_sparse, vectors_2d         = self._barycentric_projection(P, a, src_coords, tgt_coords, timings)
        source_mask                  = source > 0
        div, div_mag                 = self._compute_divergence(vectors_2d, source_mask, timings)

        timings['TOTAL'] = _toc_val(t_total)
        if self.verbose:
            self._print_timings(timings)

        return OTResult(
            src_coords  = src_coords,
            tgt_coords  = tgt_coords,
            v_sparse    = v_sparse,
            vectors_2d  = vectors_2d,
            div         = div,
            div_mag     = div_mag,
            source_mask = source_mask,
            timings     = timings,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_distributions(self, source: np.ndarray, goal: np.ndarray) -> plt.Figure:
        """Side-by-side heatmaps of source and goal distributions.

        Parameters
        ----------
        source : (n, n) – source density
        goal   : (n, n) – goal density

        Returns
        -------
        matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(source, origin='lower', cmap='Blues')
        axes[0].set_title('Source distribution')
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(goal, origin='lower', cmap='Oranges')
        axes[1].set_title('Goal distribution')
        plt.colorbar(im1, ax=axes[1])
        fig.tight_layout()
        return fig

    def plot_vector_field(
        self,
        source    : np.ndarray,
        goal      : np.ndarray,
        result    : OTResult,
    ) -> plt.Figure:
        """Two-panel figure: (left) OT arrows over source; (right) projected positions over goal.

        Arrow tips land exactly at the barycentric-projection target position.
        Projected positions are shown as semi-transparent blue squares on the goal grid.

        Parameters
        ----------
        source : (n, n) – source density
        goal   : (n, n) – goal density
        result : OTResult – output of solve()

        Returns
        -------
        matplotlib Figure
        """
        n = self.n
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Left: one arrow per occupied source cell
        axes[0].imshow(source, origin='lower', cmap='Blues', alpha=0.6)
        axes[0].quiver(
            result.src_coords[:, 0], result.src_coords[:, 1],
            result.v_sparse  [:, 0], result.v_sparse  [:, 1],
            color='crimson', scale=1.0, scale_units='xy', angles='xy', width=0.003,
        )
        axes[0].set_title('OT transport plan  (one arrow per occupied cell)')

        # Right: projected positions as blue squares overlaid on goal
        projected  = result.src_coords + result.v_sparse   # (n_src, 2) col, row
        proj_grid  = np.zeros((n, n))
        for col, row in np.round(projected).astype(int):
            r = np.clip(row, 0, n - 1)
            c = np.clip(col, 0, n - 1)
            proj_grid[r, c] += 1

        axes[1].imshow(goal, origin='lower', cmap='Oranges', alpha=0.6)
        proj_rgba         = np.zeros((n, n, 4))
        proj_rgba[..., 0] = 0.13   # R  (steelblue)
        proj_rgba[..., 1] = 0.47   # G
        proj_rgba[..., 2] = 0.71   # B
        proj_rgba[..., 3] = np.where(proj_grid > 0, 0.5, 0.0)
        axes[1].imshow(proj_rgba, origin='lower', interpolation='nearest')
        axes[1].set_title('Projected positions over goal distribution')

        fig.tight_layout()
        return fig

    def plot_divergence(self, result: OTResult) -> plt.Figure:
        """Side-by-side: signed divergence and its magnitude (source support only).

        Low magnitude = spatially coherent flow = good push candidate region.
        Red = mass radiating outward (source); blue = mass converging (sink).

        Parameters
        ----------
        result : OTResult – output of solve()

        Returns
        -------
        matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        div     = result.div
        div_mag = result.div_mag

        vmax = np.percentile(np.abs(div[div != 0]), 99) if (div != 0).any() else 1.0
        im0 = axes[0].imshow(div, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0].set_title('Signed divergence  (red=source, blue=sink)')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(div_mag, origin='lower', cmap='hot_r')
        axes[1].set_title('Divergence magnitude  (dark = coherent / laminar)')
        plt.colorbar(im1, ax=axes[1])

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _build_coords(n: int) -> np.ndarray:
        """Build (n*n, 2) array of (col, row) coordinates for an n×n grid."""
        x, y = np.arange(n), np.arange(n)
        X, Y = np.meshgrid(x, y)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def _extract_support(
        self,
        source : np.ndarray,
        goal   : np.ndarray,
        timings: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract occupied cells and normalised marginals from source and goal.

        Parameters
        ----------
        source, goal : (n, n) density arrays
        timings      : dict to record elapsed ms in-place

        Returns
        -------
        src_coords : (n_src, 2)  col/row of occupied source cells
        tgt_coords : (n_tgt, 2)  col/row of occupied goal cells
        a          : (n_src,)    normalised source marginal
        b          : (n_tgt,)    normalised goal marginal
        """
        t0 = _tic()
        src_idx = np.where(source.flatten() > 0)[0]
        tgt_idx = np.where(goal  .flatten() > 0)[0]

        src_coords = self._coords[src_idx]
        tgt_coords = self._coords[tgt_idx]

        a = source.flatten()[src_idx].astype(float); a /= a.sum()
        b = goal  .flatten()[tgt_idx].astype(float); b /= b.sum()

        timings['support_extraction'] = _toc_val(t0)
        if self.verbose:
            print(f"  {'support extraction':<40s} {timings['support_extraction']:8.2f} ms"
                  f"  ({len(src_idx)} src, {len(tgt_idx)} tgt cells)")
        return src_coords, tgt_coords, a, b

    def _build_cost_matrix(
        self,
        src_coords : np.ndarray,
        tgt_coords : np.ndarray,
        timings    : dict,
    ) -> np.ndarray:
        """Compute normalised squared-Euclidean cost matrix of shape (n_src, n_tgt).

        Parameters
        ----------
        src_coords : (n_src, 2)
        tgt_coords : (n_tgt, 2)
        timings    : dict to record elapsed ms in-place

        Returns
        -------
        C : (n_src, n_tgt) float – cost matrix, values in [0, 1]
        """
        t0 = _tic()
        C  = ot.dist(src_coords, tgt_coords, metric='sqeuclidean')
        C  = C / C.max()
        timings['cost_matrix'] = _toc_val(t0)
        if self.verbose:
            print(f"  {'cost matrix':<40s} {timings['cost_matrix']:8.2f} ms"
                  f"  ({src_coords.shape[0]}×{tgt_coords.shape[0]})")
        return C

    def _run_sinkhorn(
        self,
        a      : np.ndarray,
        b      : np.ndarray,
        C      : np.ndarray,
        timings: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Run Sinkhorn algorithm to produce regularised OT plan P.

        Parameters
        ----------
        a : (n_src,) – source marginal
        b : (n_tgt,) – target marginal
        C : (n_src, n_tgt) – cost matrix
        timings : dict to record elapsed ms in-place

        Returns
        -------
        P   : (n_src, n_tgt) – OT transport plan
        log : dict – Sinkhorn convergence info (niter, etc.)
        """
        t0 = _tic()
        P, log = ot.sinkhorn(a, b, C, reg=self.reg, log=True)
        timings['sinkhorn'] = _toc_val(t0)
        if self.verbose:
            print(f"  {'Sinkhorn':<40s} {timings['sinkhorn']:8.2f} ms"
                  f"  (reg={self.reg}, {log.get('niter', '?')} iters)")
        return P, log

    def _barycentric_projection(
        self,
        P          : np.ndarray,
        a          : np.ndarray,
        src_coords : np.ndarray,
        tgt_coords : np.ndarray,
        timings    : dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-cell displacement vectors via barycentric projection.

        For each occupied source cell i:
            projected_i = (sum_j P_ij * tgt_coords_j) / a_i
            v_i         = projected_i - src_coords_i

        Parameters
        ----------
        P          : (n_src, n_tgt) – OT plan
        a          : (n_src,)       – source marginal
        src_coords : (n_src, 2)
        tgt_coords : (n_tgt, 2)
        timings    : dict to record elapsed ms in-place

        Returns
        -------
        v_sparse   : (n_src, 2) – displacement per occupied source cell
        vectors_2d : (n, n, 2)  – dense field; zero outside source support
        """
        t0          = _tic()
        T_x         = (P @ tgt_coords) / a[:, None]
        v_sparse    = T_x - src_coords

        n           = self.n
        src_idx     = np.ravel_multi_index(
            (src_coords[:, 1].astype(int), src_coords[:, 0].astype(int)), (n, n)
        )
        vectors_2d  = np.zeros((n, n, 2))
        vectors_2d.reshape(-1, 2)[src_idx] = v_sparse

        timings['barycentric_projection'] = _toc_val(t0)
        if self.verbose:
            print(f"  {'barycentric projection + scatter':<40s} "
                  f"{timings['barycentric_projection']:8.2f} ms")
        return v_sparse, vectors_2d

    def _compute_divergence(
        self,
        vectors_2d  : np.ndarray,
        source_mask : np.ndarray,
        timings     : dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute signed divergence and its magnitude via central finite differences.

        Results are zeroed outside the source support to suppress boundary artefacts
        introduced by np.gradient at the transitions from non-zero to zero.

        Parameters
        ----------
        vectors_2d  : (n, n, 2) – dense displacement field
        source_mask : (n, n)    – True where source has mass
        timings     : dict to record elapsed ms in-place

        Returns
        -------
        div     : (n, n) – signed divergence (positive = outward / source,
                           negative = inward / sink)
        div_mag : (n, n) – |div|; small values indicate coherent laminar flow
        """
        t0      = _tic()
        vx      = vectors_2d[..., 0]
        vy      = vectors_2d[..., 1]
        div     = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
        div_mag = np.abs(div)
        div     = div     * source_mask
        div_mag = div_mag * source_mask

        timings['divergence'] = _toc_val(t0)
        if self.verbose:
            print(f"  {'divergence':<40s} {timings['divergence']:8.2f} ms")
        return div, div_mag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_timings(timings: dict) -> None:
        """Print a formatted timing summary with sub-steps and total."""
        print("Computation times:")
        for key, ms in timings.items():
            if key != 'TOTAL':
                print(f"  {key:<40s} {ms:8.2f} ms")
        print(f"  {'TOTAL':<40s} {timings.get('TOTAL', 0.0):8.2f} ms")
        print()


# ---------------------------------------------------------------------------
# Module-level timing helpers
# ---------------------------------------------------------------------------

def _tic() -> float:
    """Return current time in seconds."""
    return time.perf_counter()

def _toc_val(t0: float) -> float:
    """Return milliseconds elapsed since t0."""
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _make_demo_distributions(n: int, rng: np.random.Generator):
    """Create a random square pile and a T-shaped goal on an n×n grid.

    Returns
    -------
    pile_2d : (n, n) – source distribution
    goal_2d : (n, n) – goal distribution
    """
    # Source: uniform random pile inside a square patch
    N      = 300
    sq_lo  = int(0.2 * n)
    sq_hi  = int(0.6 * n)
    pile_2d = np.zeros((n, n))
    pts     = rng.integers(sq_lo, sq_hi, size=(N, 2))
    for py, px in pts:
        pile_2d[py, px] += 1

    # Goal: T-shape on a regular grid
    goal_2d   = np.zeros((n, n))
    bar_row   = int(0.25 * n)
    bar_col_l = int(0.25 * n)
    bar_col_r = int(0.75 * n)
    stem_col  = int(0.50 * n)
    stem_bot  = int(0.75 * n)
    step      = 2
    for col in range(bar_col_l, bar_col_r + 1, step):
        goal_2d[bar_row, col] += 1
    for row in range(bar_row + step, stem_bot + 1, step):
        goal_2d[row, stem_col] += 1

    return pile_2d, goal_2d


if __name__ == '__main__':
    n   = 150
    rng = np.random.default_rng(42)

    pile_2d, goal_2d = _make_demo_distributions(n, rng)

    planner = OTPlannerSparse(grid_size=n, reg=0.001, verbose=True)
    result  = planner.solve(pile_2d, goal_2d)

    planner.plot_distributions(pile_2d, goal_2d)
    planner.plot_vector_field(pile_2d, goal_2d, result)
    planner.plot_divergence(result)
    plt.show()
