# Data Repository for "Distillation of atomistic foundation models across architectures and chemical domains"

Included in this repository are the fine-tuned and distilled models used to perform simulations in the paper "Distillation of atomistic foundation models across architectures and chemical domains". Additionally, the synthetic datasets used to train the distilled models are included for each application.

### Repository Structure

Folders are organised by target system, excluding the `figure-generation` folder which contains scripts for generating the figures in the paper. Each folder contains the following:

- `synthetic-data.xyz`: The synthetic dataset used to train the distilled models.
- `distilled-ARCH`: Distilled modles for the target system.

### Loading Models

For all fine-tuned foundation models, the model can be loaded using the `graph_pes.models.load_model` function, the same applies for all non-ACE distilled models.
```python
from graph_pes.models import load_model

model = load_model("path/to/model.pt")
calculator = model.ase_calculator()
```
ACE models can be loaded as follows:
```python
from pyace import PyACECalculator

calculator = PyACECalculator("path/to/model.yace")
```

### Foundation Models

Foundation models used for fine-tuning in this work are available at the following links:

- [MACE-OFF24](https://github.com/ACEsuit/mace-off)
- [MACE-MP-0b3](https://github.com/ACEsuit/mace-foundations)
- [MatterSim-v1.0.0-1M](https://github.com/microsoft/mattersim/tree/main/pretrained_models)
- [Orb-v3](https://github.com/orbital-materials/orb-models?tab=readme-ov-file)

<!-- ### Licensing

All distilled models and datasets are licensed under the [MIT license](LICENSE.md). -->
