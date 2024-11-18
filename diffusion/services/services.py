from dataclasses import dataclass
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch._inductor.config
from diffusion.optimazition import compile_pipeline

@dataclass
class SDServiceConfiguration:
    """
    The configuration class for the stable diffusion service model
    """
    model_path:str
    device:torch.device
    image_size:tuple[int, int]
    num_inf_steps:int = 50

    def check(self):
        """
        Check the configuration for the service model

        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        assert self.image_size[0] % 8 == 0, "Image size must be divisible by 8"
        assert self.image_size[1] % 8 == 0, "Image size must be divisible by 8"
        assert self.num_inf_steps > 0, "Number of inference steps must be greater than 0"
        assert self.model_path != "" or self.model_path is None, "Model path must be provided"

        return True

    

class Service:
    """
    The service class that defines the interface for the service model
    """

    def run(self, inputs, **kwargs):
        """
        Run the service with the given inputs

        Args:
            inputs (torch.Tensor): The input tensor to the service model [batch_size, H, W, C]

        Returns:
            torch.Tensor: The output tensor of the service model [batch_size, H, W, C]
        """
        pass

class SDService(Service):
    """
    The service class for the stable diffusion service model
    """
    def __init__(self, model_path: str, device: torch.device, numerical_precision: torch.dtype, generator: torch.Generator):
        super().__init__()
        self.generator = generator
        self.pipeline = self._load_pipeline(model_path, device, numerical_precision)
        self.model_path = model_path
        self.precision = numerical_precision
        self.device = device

        # activate when on gpu
        # compile_pipeline(self.pipeline, with_dynamic_quant=True)

    def _load_pipeline(self, model_path:str, device:torch.device, numerical_precision: torch.dtype) -> DiffusionPipeline:
        """
        Load the service model

        Args:
            model_path (str): The path to the service model
            device (torch.device): The device to run the service model
            numerical_precision (torch.dtype): The numerical precision to run the service model
        """
        pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=numerical_precision, use_safetensors=True).to(device)
        # load faster scheduler https://huggingface.co/docs/diffusers/en/stable_diffusion#speed
        # works with 20 - 25 steps
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        # enable slicing attention if poor memory
        # pipeline.enable_attention_slicing()
        
        return pipeline

    def run(self, prompts, configuration:SDServiceConfiguration):
        """
        Run the service with the given inputs

        Args:
            inputs (torch.Tensor): The input tensor to the service model [batch_size, H, W, C]

        Returns:
            torch.Tensor: The output tensor of the service model [batch_size, H, W, C]
        """
        assert configuration.check(), "Invalid configuration for the service model"
        return self.pipeline(prompts, num_inference_steps=configuration.num_inf_steps, height=configuration.image_size[0], width=configuration.image_size[1], generator=self.generator)
