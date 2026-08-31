################################
# A SCRIPT FOR DATA COLLECTION #
################################

import argparse
import itertools
import yaml
import numpy as np
import torch
from pathlib import Path

# Relative imports: sandbox_manipulation_clean does `from .utilities.materials
# import *`, which only resolves inside the package. Run from the REPO ROOT as
#     python -m Genesis.data_collection_clean
# exactly as tests/scaling_investigation/benchmark_scaling.py and the probes are run. (Importing
# these as top-level modules fails, and so does running this file as a script
# from inside Genesis/ — the two failure modes are symmetric and neither works.)
from .sandbox_manipulation_clean import SandboxManipulation
from .state_library import (build_state_library, load_or_build_state_library,
                            StateLibrary, STATE_LIBRARY_FILENAME)

##################################
# PARAMS THAT REQUIRE RESTARTING #
##################################
BASIC_SETTING = "basic"
DEFAULT_SHAPES = ["cube"]
DEFAULT_NUM_PARTICLES = [40, 50]
PARTICLE_SIZES = np.linspace(0.005, 0.012, 5).tolist()

PARTICLE_FRICTIONS = np.linspace(0.05, 0.5, 5).tolist()
PARTICLE_DENSITIES = np.linspace(750, 5000, 5).tolist()
BOX_FRICTION = np.linspace(0.05, 0.5, 4).tolist()
PER_PARTICLE_VALUE_PROBABILITY = 0.5


def scalar_or_particle_values(value: float, n_particles: int, rng: np.random.Generator):
    if rng.random() >= PER_PARTICLE_VALUE_PROBABILITY:
        return float(value), None
    return float(value), rng.uniform(0.8 * value, 1.2 * value, n_particles).tolist()


def build_particle_size_settings(n_particles: int, rng: np.random.Generator):
    settings = []
    for size in PARTICLE_SIZES:
        base, sampled = scalar_or_particle_values(size, n_particles, rng)
        settings.append({"base": base, "sampled": sampled})
    return settings


def build_property_env_settings(n_particles: int, rng: np.random.Generator):
    env_settings = []
    for particle_friction, particle_density, box_friction in itertools.product(
        PARTICLE_FRICTIONS,
        PARTICLE_DENSITIES,
        BOX_FRICTION,
    ):
        friction_base, friction_sampled = scalar_or_particle_values(particle_friction, n_particles, rng)
        density_base, density_sampled = scalar_or_particle_values(particle_density, n_particles, rng)
        env_settings.append(
            {
                "particle_friction": friction_base,
                "sampled_particle_friction": friction_sampled,
                "particle_density": density_base,
                "sampled_particle_density": density_sampled,
                "box_friction": float(box_friction),
            }
        )
    return env_settings


def read_yaml(path: str):
    base_dir = Path(__file__).parent
    full_path = base_dir / path
    with open(full_path) as stream:
        return yaml.safe_load(stream)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect sandbox manipulation data.")
    parser.add_argument("--settings", nargs="+", default=BASIC_SETTING)
    parser.add_argument("--particle-shape", choices=["config", "cube", "sphere", "cylinder", "rectangle"], default=DEFAULT_SHAPES)
    parser.add_argument("--num-particles", nargs="+", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument("--particle-sizes", nargs="+", type=float, default=PARTICLE_SIZES)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--samples-per-env", type=int, default=5)
    parser.add_argument("--output-root", default="data/corl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viewer-type", choices=["observer", "bird", "leveled"], default=None)
    parser.add_argument(
        "--state-library", type=int, default=0, metavar="N_SETTLES",
        help="settle N fresh random piles once per build, expand them by the "
             "container's symmetries (x8 for a square tray), save them beside "
             "the data as '" + STATE_LIBRARY_FILENAME + "', and reset from that "
             "bank instead of re-settling for every material batch. 10-20 is a "
             "good range. 0 (default) keeps the current behaviour: a fresh "
             "shuffle_particles() + settle per batch.")
    parser.add_argument(
        "--state-library-damping", type=float, default=0.0,
        help="temporary viscous damping on the particles WHILE the state "
             "library settles, removed afterwards. A numerical convergence "
             "aid, not a physical model (real air drag is ~3e-5 of a 5 mm "
             "cube's weight). Safe here because this settle only needs to "
             "reach some valid resting configuration; it is deliberately not "
             "applied to the post-push settle, where it would bias the "
             "recorded s' toward smaller displacements.")
    parser.add_argument(
        "--rebuild-state-library", action="store_true",
        help="settle a fresh library even if a compatible one already exists "
             "beside the output. Off by default: settling is the dominant "
             "startup cost, and a library is reusable across runs.")
    parser.add_argument(
        "--no-state-library-augment", action="store_true",
        help="with --state-library, store only the settled states themselves, "
             "without the symmetry-expanded variants")
    parser.add_argument(
        "--placement-aware", action="store_true",
        help="draw each tool touchdown pose from its free configuration space "
             "so the plate does not descend into a particle; falls back to the "
             "blind draw per sample when no free placement exists")
    parser.add_argument(
        "--independent-travel-distance", action="store_true",
        help="let every env sample its own push length. Off by default: envs "
             "step in lockstep and the sweep is sized from the longest travel "
             "in the batch, so independent lengths make every env run for the "
             "longest one's duration (measured 1.54x of a 2.64x batching "
             "penalty at 8 envs). Sharing it costs only within-batch variation "
             "in one of five action dimensions.")
    parser.add_argument(
        "--perpendicular-pushes", action="store_true",
        help="push along the blade's face normal instead of in an independently "
             "drawn direction. This is the planar-pushing convention and it "
             "collapses the 5-DOF action to 4-DOF, which is what the "
             "switched-linear visual-foresight baseline assumes (see "
             "docs/linear_visual_foresight_baseline.md). The push-length "
             "distribution is unchanged.")
    parser.add_argument(
        "--push-length", type=float, default=None, metavar="METRES",
        help="fix every push's travel distance. With --perpendicular-pushes "
             "this collects a dataset supporting a SINGLE transition operator "
             "(one length bin, one canonical frame), which is the targeted "
             "collection the visual-foresight baseline needs first. Pushes that "
             "cannot reach the length inside the tray are truncated and "
             "reported — check that count is 0 before fitting.")
    parser.add_argument(
        "--pile-extent", type=float, default=None, metavar="METRES",
        help="spawn the particles inside a square of this HALF-WIDTH at the "
             "tray centre instead of spread over the whole tray, so they land "
             "as one compact multi-layer heap. Layers are derived from the area "
             "unless --pile-layers overrides. See docs/piled_collection.md.")
    parser.add_argument(
        "--pile-layers", type=int, default=None,
        help="force this many stacked spawn layers (default: derived from "
             "--pile-extent and the particle count). Layers are dropped, not "
             "interpenetrating; the settle collapses them into a natural pile.")
    parser.add_argument(
        "--pile-aware-actions", action="store_true",
        help="start every push in contact with the pile and sweep through it: "
             "the blade is placed one particle-width from the pile's near face "
             "and laterally aligned so its swath contains material. Supersedes "
             "--placement-aware and applies the perpendicular convention "
             "itself. Blind sampling put only ~14%% of the pile in a typical "
             "push's path and spent half the simulation budget on pushes too "
             "weak for any model to beat persistence on.")
    parser.add_argument(
        "--pile-clearance", type=float, default=None, metavar="METRES",
        help="blade-to-pile gap at the start of a pile-aware push "
             "(default: one particle size)")
    parser.add_argument(
        "--min-swath-particles", type=int, default=3,
        help="reject a pile-aware lateral alignment whose blade swath holds "
             "fewer than this many particles, and re-draw (default 3)")
    parser.add_argument(
        "--constant-params", action="store_true",
        help="use one fixed material setting instead of sweeping the "
             "friction x density x box-friction grid (100 batches). Values "
             "come from --particle-friction / --particle-density / "
             "--box-friction, and per-particle jitter is disabled.")
    parser.add_argument("--particle-friction", type=float, default=0.3)
    parser.add_argument("--particle-density", type=float, default=1000.0)
    parser.add_argument("--box-friction", type=float, default=0.3)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed every random draw in a run: material-property jitter and "
             "state-library selection (numpy) AND particle spawn poses, "
             "orientations and action sampling (torch). Without it a run is "
             "not reproducible at all -- which matters because a dataset is "
             "only as re-derivable as the actions that produced it, and "
             "because a run that produces something odd cannot be replayed to "
             "look at it. Recorded in each batch's saved config.")
    parser.add_argument(
        "--n-batches", type=int, default=None,
        help="limit how many material batches are collected (default: all). "
             "With --constant-params this is simply how many times the pile is "
             "reset and swept, i.e. the number of transition sequences.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Both generators, not just numpy: the particle spawn poses, the random
    # orientations and the whole action draw are torch calls on the GPU, so
    # seeding numpy alone would leave everything that actually shapes a
    # transition unseeded.
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    
    if args.samples_per_env <= 0:
        raise ValueError("--samples-per-env must be positive")

    config = read_yaml(f"configs/{BASIC_SETTING}.yaml")
    config.setdefault("data_collection", {})["seed"] = args.seed

    # Iterate shapes
    shapes = [args.particle_shape] if args.particle_shape != DEFAULT_SHAPES else DEFAULT_SHAPES
    for shape in shapes:
        config["material"]["shape"] = shape

        # Iterate number of particles
        for n_p in args.num_particles:
            config["material"]["n_particles"] = n_p
            if args.constant_params:
                # One fixed setting, no per-particle jitter: the physical
                # parameters are held constant so that n_objects is the only
                # thing varying across the run.
                env_settings = [{
                    "particle_friction": float(args.particle_friction),
                    "sampled_particle_friction": None,
                    "particle_density": float(args.particle_density),
                    "sampled_particle_density": None,
                    "box_friction": float(args.box_friction),
                }]
            else:
                env_settings = build_property_env_settings(n_p, rng)
            if args.n_batches is not None:
                env_settings = (env_settings * args.n_batches)[:args.n_batches]
            
            # Iterate particle sizes
            if args.particle_sizes == PARTICLE_SIZES:
                sizes = build_particle_size_settings(n_p, rng)
            elif len(args.particle_sizes) == 1:
                # A single size means "every particle is this size" — pass it
                # as a scalar. Passing it as a 1-element per-particle list
                # instead makes random_sequential_addition reject it, since
                # that list must be either length 2 (a range) or length
                # n_particles.
                sizes = [{"base": args.particle_sizes[0], "sampled": None}]
            else:
                sizes = [{"base": args.particle_sizes[0],
                          "sampled": args.particle_sizes}]
            # NOTE: the old `if n_p == 50: sizes = sizes[:-1]` hack lived here.
            # It was patching one cell of a general problem — most (n, size)
            # combinations simply cannot be placed in a 128 mm tray (see
            # docs/scaling_to_200_objects.md §4). Placement now falls back to
            # stacked layers, and genuinely infeasible combinations raise and
            # are skipped with a clear message rather than being special-cased.
            for size_setting in sizes:
                config["material"]["particle_size"] = size_setting["base"]
                config.setdefault("data_collection", {})["sampled"] = {}
                if size_setting["sampled"] is not None:
                    config["data_collection"]["sampled"]["particle_size"] = size_setting["sampled"]


                # iterate through material settings
                print(f"\n+++ shape={shape}, size={size_setting['base']}, n_particles={n_p} +++")

                if args.pile_extent is not None or args.pile_layers is not None:
                    config.setdefault("spawn", {})
                    if args.pile_extent is not None:
                        config["spawn"]["pile_extent"] = args.pile_extent
                    if args.pile_layers is not None:
                        config["spawn"]["pile_layers"] = args.pile_layers

                sm = SandboxManipulation(
                    config=config,
                    n_envs=args.n_envs,
                    debug=args.debug,
                    viewer_type=args.viewer_type,
                )

                sm.build()

                out_path = (
                    f"{args.output_root}/{shape}/n{n_p}/size{size_setting['base']}"
                )

                # Optional: pay for a handful of settles once, then reset from
                # the resulting bank for every material batch. A reset via
                # set_particle_state needs no settle at all (the stored state
                # is already at rest), which is 50-80x cheaper than
                # shuffle_particles + settle -- and the symmetry expansion means
                # far more distinct starts than settles paid for.
                state_library = None
                if args.state_library > 0:
                    print(f"\n=== building state library "
                          f"({args.state_library} settles) ===", flush=True)
                    try:
                        state_library = load_or_build_state_library(
                            sm,
                            Path(__file__).parent / out_path / STATE_LIBRARY_FILENAME,
                            n_settles=args.state_library,
                            augment=not args.no_state_library_augment,
                            damping=args.state_library_damping,
                            reuse=not args.rebuild_state_library)
                    except RuntimeError as e:
                        print(f"  could not build state library ({e}); "
                              f"falling back to per-batch shuffling")
                        state_library = None

                for property_idx, property_setting in enumerate(env_settings):
                    print(f"\n--- material batch {property_idx + 1}/{len(env_settings)}", flush=True)

                    sm.set_material_properties(property_setting)
                    try:
                        if state_library is not None:
                            state_library.apply(sm, rng)
                        else:
                            sm.shuffle_particles()
                        sm.collect_data_samples(
                            n_samples=args.samples_per_env,
                            path=out_path,
                            placement_aware=args.placement_aware,
                            shared_travel_distance=not args.independent_travel_distance,
                            perpendicular_pushes=args.perpendicular_pushes,
                            push_length=args.push_length,
                            pile_aware=args.pile_aware_actions,
                            pile_clearance=args.pile_clearance,
                            min_swath_particles=args.min_swath_particles,
                        )
                    except RuntimeError as e:
                        print(f"Maximum attempts reached, stopped retrying to shuffle, skipping: {e}")


                sm.destroy()


if __name__ == "__main__":
    main()
