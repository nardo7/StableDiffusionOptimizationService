from dataclasses import dataclass
import json
import logging
from typing import Any
import os
import PIL.Image
import pyDOE3
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from diffusion.services import SDService, SDServiceConfiguration, services
import torch
from tqdm import tqdm
from diffusers.utils import make_image_grid
from IPython.display import display
import datasets
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import time
from diffusion.data import create_sub_dataset, PartiPromptDataset
import PIL
import numpy as np
import math
import gc


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

@dataclass
class ExperimentConfiguration:
    """
    The configuration class for the experiment
    """
    path:str = ""
    name:str = ""
    overwrite_results = False
    dataset_path:str = ""

    def check(self):
        """
        Check the configuration for the experiment

        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        assert self.path != "" and self.path is not None, "Path must be provided"
        assert self.name != "" and self.name is not None, "Name must be provided"
        assert self.dataset_path != "" and self.dataset_path is not None, "Dataset path must be provided"

        return True

class Experiment:

    def __init__(self, logger, configuration: ExperimentConfiguration):
        configuration.check()
        self.configuration = configuration
        self.logger = logger

    def run(self, **kwargs):
        pass

    def _generate_experiment_name(self, experiment: dict, **kwargs):
        pass

    def _save_results(self, repetition, experiment, results: dict, image_grid: PIL.Image.Image):
        self.logger.info("Saving results")
        save_path = os.path.join(self.configuration.path, self._generate_experiment_name(experiment, repetition=repetition))
        save_file_path = os.path.join(save_path, "results.json")
        
        results_exists = os.path.exists(save_path)

        if results_exists and not self.configuration.overwrite_results:
            self.logger.error(f"Results file {save_file_path} already exists. Set overwrite_results to True to overwrite the file")
            raise ValueError(f"Results file {save_file_path} already exists. Set overwrite_results to True to overwrite the file")

        if not results_exists:
            os.makedirs(save_path)
        elif not self.configuration.overwrite_results:
            raise ValueError(f"Results file {save_file_path} already exists. Set overwrite_results to True to overwrite the file")

        self.logger.info(f"Saving results to {save_file_path}")
        with open(save_file_path, 'w' if results_exists else 'x') as f:
            json.dump(results, f)

        image_file_path = os.path.join(save_path, "image_grid.png")
        image_grid.save(image_file_path)

    def _load_dataset(self):
        # get current path
        
        # load dataset
        dataset = PartiPromptDataset(self.configuration.dataset_path)

        # dataset = create_sub_dataset(dataset, 4)

        return dataset

    def get_available_device(self) -> tuple[torch.device, str]:
        """Helper method to find best possible hardware to run
        Returns:
            torch.device used to run experiments.
            str representation of backend.
        """
        # Check if CUDA is available
        if torch.cuda.is_available():
            return torch.device("cuda:0"), "cuda"

        # Check if ROCm is available
        if torch.version.hip is not None and torch.backends.mps.is_available():
            return torch.device("rocm"), "rocm"

        # Check if MPS (Apple Silicon) is available
        if torch.backends.mps.is_available():
            return torch.device('mps'), "mps"

        # Fall back to CPU
        return torch.device("cpu"), "cpu"

@dataclass
class InferenceExperimentConfiguration(ExperimentConfiguration):

    factors = []
    levels = []
    repetitions = 2

    def check(self):
        super().check()
        assert len(self.factors) > 0, "Factors must be provided"
        assert len(self.levels) > 0, "Levels must be provided"
        assert self.repetitions > 0, "Repetitions must be greater than 0"


class InferenceExperiment(Experiment):

    def __init__(self, logger, configuration: InferenceExperimentConfiguration):
        super().__init__(logger, configuration)
        self.parameters = {
            factor: levels for factor, levels in zip(self.configuration.factors, self.configuration.levels)
        }
        self.levels_counts = [len(levels) for levels in self.configuration.levels]
        self.logger.info(f"Generating experiments with factors: {self.configuration.factors} and levels: {self.configuration.levels}")
        self.experiments = self._generate_experiments()
        device, backend = self.get_available_device()
        logger.info(f"Using device: {device} with backend: {backend}")
        generator = torch.Generator(device=device).manual_seed(20)
        self.service = services.SDService(device, torch.bfloat16, generator)
        self.service.warmup_models()

    def _generate_experiments(self):
       experiments = pyDOE3.fullfact(self.levels_counts)
       self.logger.info(f"DoE: {experiments}")
       self.experiment_configs = [  {self.configuration.factors[factor]: self.configuration.levels[factor][int(level)] for factor, level in enumerate(exp)} for exp in experiments]

    def _generate_experiment_name(self, experiment: dict, **kwargs):
        name = ""
        for k, v in kwargs.items():
            name += f"{k}_{v}__"
        for k, v in experiment.items():
            name += f"{k}_{v}__"
        return name[:-2]

    def run(self, **kwargs):
        dataset = self._load_dataset()
        skip = 0
        for repetition in range(self.configuration.repetitions):
            for experiment in tqdm(self.experiment_configs[skip:]):
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=4 if experiment["batch_size"] is None else experiment["batch_size"], shuffle=False, num_workers=2, prefetch_factor=10)
                images, clip_scores, runtime = self._run_experiment(experiment, dataloader)
                print("Images shape", images.shape)
                len_to_print = 8 if len(images) >= 8 else 4 if len(images) >= 4 else len(images)
                images_to_print = images[:len_to_print]
                
                image_grid = make_image_grid([PIL.Image.fromarray((img * 255).astype(np.uint8)) for img in images_to_print], rows= math.ceil(len_to_print / 4), cols=4 if len_to_print >= 4 else len_to_print)
                results = {
                    "experiment": experiment,
                    "clip_score": np.mean(clip_scores),
                    "clip_score variance": np.var(clip_scores),
                    "runtime (s/b)": np.mean(runtime),
                    "runtime variance (s/b)": np.var(runtime),
                    "runtimes":runtime.tolist(),
                    "clip_scores": clip_scores.tolist()
                }
                self._save_results(repetition, experiment, results, image_grid)
                del dataloader
                gc.collect()
                torch.cuda.empty_cache()

    def _calculate_clip_score(self, images, prompts):
        images_np = np.array(images * 255).astype(np.uint8)
        clip_score = clip_score_fn(torch.from_numpy(images_np).permute(0, 3, 1, 2), prompts).detach()
        del images_np
        return round(float(clip_score), 4)

    def _run_experiment(self, experiment: dict, dataloader: torch.utils.data.dataloader.DataLoader):
        self.logger.info(f"Running experiment: {experiment}")
        service_config = services.SDServiceConfiguration(device=self.service.device, 
                                                         image_size=experiment["image_size"], 
                                                         num_inf_steps=experiment["num_inf_steps"],
                                                         enable_cache=experiment["cache"])
        service_config.check()
        images = np.array([])
        clip_scores = np.zeros([len(dataloader)])
        runtime = np.zeros([len(dataloader)])
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = batch['Prompt']
            start = time.time()
            results = self.service.run(inputs, service_config)
            end = time.time()
            runtime[i] = end - start
            
            if images.shape[0] < 9:
                images = np.concatenate([images, np.array(results.images)]) if images.size > 0 else np.array(results.images)
            clip_score = self._calculate_clip_score(np.array(results.images), inputs)
            del inputs
            del results.images
            del results
            clip_scores[i] = clip_score
            
        return images, clip_scores, runtime


if __name__ == "__main__":
    # set an environment variable for this run
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Running inference experiment")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    logger.info("Running inference experiment")

    config = InferenceExperimentConfiguration()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    
    config.path = os.path.join(curr_path, "inference_data")
    config.name = "inference_experiment"
    config.overwrite_results = False
    config.factors = ["cache", "batch_size", "num_inf_steps", "image_size"]
    config.levels = [[False], [8, 2], [25, 15], [(512, 512), (256, 256)]]
    config.dataset_path = os.path.join(curr_path, "../data/PartiPrompts_120.tsv")
    experiment = InferenceExperiment(logger, config)
    experiment.run()



            