# Future-integration models

Model architectures salvaged from `GranularDynamics2/myClasses/` during the
2026-07 cleanup. They are not wired into the registry yet but are intended to
be integrated into the current pipeline (register a factory in
`registry/model_registry.py` and add a config under `configs/model/`).

- `NCAModels.py` — Neural Cellular Automata dynamics models.
- `SpatTransNet.py` — Spatial-transformer push model.
- `MultiExitUnet.py` — U-Net with intermediate prediction exits.
- `UNetModels.py` — plain U-Net variants (also provides `ConvBlock` used by
  `UNetModels_conditioned.py`).
- `UNetModels_conditioned.py` — property-conditioned U-Net variants (UNetFiLM
  predecessor of `model/NFDUNetFilm.py`).
- `UNetModels_modular.py` — modular U-Net building blocks.
- `Diff_Renderer.py` — differentiable renderer.

All are self-contained (torch-only imports); `UNetModels_conditioned.py`
relative-imports `UNetModels.py` from this directory.
