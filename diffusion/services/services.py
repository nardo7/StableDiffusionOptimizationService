from dataclasses import dataclass
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch._inductor.config
from diffusion.optimazition import compile_pipeline, add_cache

@dataclass
class SDServiceConfiguration:
    """
    The configuration class for the stable diffusion service model
    """
    device:torch.device
    image_size:tuple[int, int]
    num_inf_steps:int = 50
    enable_cache:bool = False

    def check(self):
        """
        Check the configuration for the service model

        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        assert self.image_size[0] % 8 == 0, "Image size must be divisible by 8"
        assert self.image_size[1] % 8 == 0, "Image size must be divisible by 8"
        assert self.num_inf_steps > 0, "Number of inference steps must be greater than 0"

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

    MODELS_BY_SIZE = {
        "512x512": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "256x256": "lambdalabs/miniSD-diffusers",
    }

    """
    The service class for the stable diffusion service model
    """
    def __init__(self, device: torch.device, numerical_precision: torch.dtype, generator: torch.Generator):
        super().__init__()
        self.generator = generator
        self.default_model_path = self.MODELS_BY_SIZE["512x512"]
        self.last_size = (512, 512)
        self.pipeline = self._load_pipeline(self.default_model_path, device, numerical_precision)
        self.precision = numerical_precision
        self.device = device

        self._optimize_pipeline(with_dynamic_quant=False)

    def _optimize_pipeline(self, with_dynamic_quant:bool) -> DiffusionPipeline:
        self.cache = add_cache(self.pipeline)
        self.is_cache_enabled = False
        # the following is not due to the lack of integration of DeepCache in torch.compile.
        # there are some opt libraries that provide such things (onediff, TensorRT)
        # if torch.cuda.is_available():
        #     compile_pipeline(self.pipeline, with_dynamic_quant)

    def warmup_models(self):
        """
        Warmup consists of compiling the models for later faster inference and not having to compile them again
        """
        if torch.cuda.is_available():
            for size in self.MODELS_BY_SIZE:
                self._load_pipeline(self.MODELS_BY_SIZE[size], self.device, self.precision)
                self._optimize_pipeline(True)

    def __size_to_string(self, image_size:tuple[int, int]) -> str:
        return f"{image_size[0]}x{image_size[1]}"

    def _load_pipeline(self, model_path:str, device:torch.device, numerical_precision: torch.dtype) -> DiffusionPipeline:
        """
        Load the service model

        Args:
            model_path (str): The path to the service model
            device (torch.device): The device to run the service model
            numerical_precision (torch.dtype): The numerical precision to run the service model
        """
        pipeline = DiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=numerical_precision, 
            use_safetensors=True if self.last_size != (256, 256) else False).to(device)
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
        configuration.check()

        # checking image size
        if self.last_size != configuration.image_size:
            self.last_size = configuration.image_size
            size = self.__size_to_string(configuration.image_size)
            self.pipeline = self._load_pipeline(self.MODELS_BY_SIZE[size], self.device, self.precision)
            self._optimize_pipeline(True)
            self.last_size = configuration.image_size
        
        # checking cache settings
        if configuration.enable_cache and not self.is_cache_enabled:
            self.is_cache_enabled = True
            self.cache.enable()
        elif not configuration.enable_cache and self.is_cache_enabled: 
            self.cache.disable()
            self.is_cache_enabled = False

        # run the inference
        return self.pipeline(prompts, num_inference_steps=configuration.num_inf_steps, height=configuration.image_size[0], width=configuration.image_size[1], generator=self.generator)
