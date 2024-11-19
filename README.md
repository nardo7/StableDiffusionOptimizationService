# Readme
## Setup

1. clone the repository `git clone ...`
2. Install the environment with conda `conda env create -f environment.yml`
3. Activate environment `conda activate projectMSGAI`
4. You should be set up.


## Experiments

**Factors**: "batch_size", "num_inf_steps", "image_size"
**Levels**: `levels = [[8, 2], [25, 15], [(512, 512), (320, 320)]]`

The experiments are stored in the following path with the following format:
`experiments/inference_data/repetition_{repetition}__batch_size_{batch_size}__num_inf_steps_{num_inf_steps}__image_size_{image_size}/`

In the experiments folder you find an image grid of samples generated and a result.json with some relevant data from the experiment like clip score and runtime.
