# Future-integration models

Model architectures salvaged from `GranularDynamics2/myClasses/` during the
2026-07 cleanup.

## Integrated (registered in `registry/model_registry.py`)

- `NCAModels.py` — `NCAWithPhysics` (Neural Cellular Automata + residual U-Net
  correction). Registered as model type `nca`; see
  `configs/model/nca.yaml`.
- `SpatTransNet.py` — `EulerianSTN` (spatial-transformer / grid-warp push
  model). Registered as model type `spatial-transformer`; see
  `configs/model/spatial_transformer.yaml`.
- `UNetModels_modular.py` — `UNet` (config-driven depth/width/activation/
  bottleneck). Registered as model type `unet-modular`; see
  `configs/model/unet_modular.yaml`. The building-block classes in this file
  (`DoubleConv`, `DownBlock`, `UpBlock`, bottleneck variants, ...) are
  internal to `UNet` and not separately registered.

Each has a matching 1-epoch smoke-test training config under
`configs/training/*_corl_limited_1e_test.yaml`.

## Awaiting consolidation (not yet registered)

These files each define several distinct top-level model classes that
overlap in purpose with each other and with already-integrated models
(`model/NFDUNetFilm.py`, and the three models above). Rather than registering
each class as a separate one-off type, they should be reviewed together and
consolidated into a smaller number of configurable models (similar to how
`NFDUNetFiLM.channels`/`depth` replaced several fixed-depth variants) before
integration.

- `UNetModels.py` — eight plain-UNet variants (`UNetSmall`, `UNetMedium`,
  `UNetLarge`, `UNetDeepSmall`, `UNetOriginal`, `UNetDeep`, `UNetMixed`,
  `UNetStrided`), differing mainly in depth/width/activation/residual
  choices that `unet-modular` (`UNetModels_modular.UNet`) already covers in a
  single configurable class. Also provides `ConvBlock`, used by
  `UNetModels_conditioned.py`.
- `UNetModels_conditioned.py` — three physics-conditioned variants
  (`UNetConditioned`: concat-conditioning, `UNetFiLM`: FiLM-conditioning,
  `UNetFiLMNFD`: FiLM-conditioning with NFD-style skip connections). These
  predate `model/NFDUNetFilm.py`, which already generalizes the FiLM
  approach via `depth`/`channels`. Note: `UNetFiLMNFD.forward` currently has
  a bug — it calls `self.dec3(c1, dim=1)` / `self.dec2(c1, dim=1)`, passing a
  `dim=1` keyword that `ConvBlock.forward(self, x)` does not accept; fix
  before considering integration.

## Skipped for now

- `MultiExitUnet.py` — `UNetMultiExit` returns a dict of three exits
  (`low`/`mid`/`high`) rather than a single tensor, and the current
  `ModelTrainingWrapper`/loss contract expects one tensor per forward call.
  Integrating it either means discarding the auxiliary exits at training
  time (defeating the point of the architecture) or extending the loss
  framework with multi-exit supervision. Left out of this pass per explicit
  instruction.

## Not integrable as-is

- `Diff_Renderer.py` — `DiffRenderer` has a syntax error (`tool_mask = `
  with no right-hand side, line 33) and does not parse. Independent of that,
  its `forward(particle_positions, tool_mask, tool_params)` signature is a
  Lagrangian particle renderer, not an Eulerian `forward(x, props)` dynamics
  model — it doesn't fit the current `ModelTrainingWrapper` contract at all.
  Needs a decision on whether to fix and repurpose it, or drop it.
