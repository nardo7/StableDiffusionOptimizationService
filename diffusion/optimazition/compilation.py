from functools import partial
import torch
from diffusers import DiffusionPipeline
from torchao.quantization import  swap_conv2d_1x1_to_linear, quantize_
from torchao.quantization import int8_dynamic_activation_int8_weight
from DeepCache import DeepCacheSDHelper
# most of the code taken from https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion

def add_cache(pipeline: DiffusionPipeline) -> DeepCacheSDHelper:
    helper = DeepCacheSDHelper(pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    return helper

def compile_pipeline(pipeline: DiffusionPipeline, with_dynamic_quant=False):
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        if with_dynamic_quant:
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True
            pipeline.fuse_qkv_projections()

        pipeline.unet.to(memory_format= torch.channels_last)
        pipeline.vae.to(memory_format= torch.channels_last)

        if with_dynamic_quant:
            swap_conv2d_1x1_to_linear(pipeline.unet, conv_filter_fn)
            swap_conv2d_1x1_to_linear(pipeline.vae, conv_filter_fn)
            quantize_(pipeline.unet, int8_dynamic_activation_int8_weight(), dynamic_quant_filter_fn)
            quantize_(pipeline.vae, int8_dynamic_activation_int8_weight(), dynamic_quant_filter_fn)

        pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
        pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

def dynamic_quant_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (1280, 640),
            (1920, 1280),
            (1920, 640),
            (2048, 1280),
            (2048, 2560),
            (2560, 1280),
            (256, 128),
            (2816, 1280),
            (320, 640),
            (512, 1536),
            (512, 256),
            (512, 512),
            (640, 1280),
            (640, 1920),
            (640, 320),
            (640, 5120),
            (640, 640),
            (960, 320),
            (960, 640),
        ]
    )


def conv_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )