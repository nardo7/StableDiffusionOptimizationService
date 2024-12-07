# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

from calib.plugin_calib import PercentileCalibrator
import torch
from config import DYNAMIC_SHAPES, get_io_shapes, remove_nesting, update_dynamic_axes
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from onnx_utils.export import AXES_NAME, generate_dummy_inputs
from quantize import MODEL_ID

import modelopt.torch.opt as mto
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy._runtime.tensorrt.constants import SHA_256_HASH_LENGTH
from modelopt.torch._deploy._runtime.tensorrt.tensorrt_utils import prepend_hash_to_bytes
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import get_onnx_bytes_and_metadata


def generate_image(pipe, prompt, image_name):
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save(f"{image_name}.png")
    print(f"Image generated using {image_name} model saved as {image_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=[
            "sdxl-1.0",
            "sdxl-turbo",
            "sd1.5",
            "sd2.1",
            "sd2.1-base",
            "sd3-medium",
            "flux-dev",
            "flux-schnell",
        ],
    )
    parser.add_argument(
        "--restore-from", type=str, default=None, help="Path to the modelopt quantized checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="Input text prompt for the model",
    )
    parser.add_argument(
        "--onnx-load-path", type=str, default="", help="Path to load the ONNX model"
    )
    parser.add_argument(
        "--trt-engine-load-path", type=str, default=None, help="Path to load the TRT engine"
    )
    parser.add_argument(
        "--dq-only", action="store_true", help="Converts the ONNX model to a dq_only model"
    )
    parser.add_argument(
        "--torch", action="store_true", help="Generate an image using the torch pipeline"
    )
    args = parser.parse_args()

    if args.model in ["sd2.1", "sd2.1-base", "sd1.5"]:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16, safety_checker=None
        )
    elif args.model == "sd3-medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16
        )
    elif args.model in ["flux-dev", "flux-schnell"]:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

    if args.torch:
        pipe.to("cuda")
        generate_image(pipe, args.prompt, args.model)
        return

    # Save the backbone of the pipeline and move it to the GPU
    add_embedding = None
    backbone = None
    if hasattr(pipe, "transformer"):
        backbone = pipe.transformer
    elif hasattr(pipe, "unet"):
        backbone = pipe.unet
        add_embedding = (
            backbone.add_embedding if args.model not in ["sd2.1", "sd2.1-base", "sd1.5"] else None
        )
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")

    if args.restore_from:
        mto.restore(backbone, args.restore_from)

    backbone.to("cuda")

    # Generate dummy inputs for the backbone
    dummy_inputs = generate_dummy_inputs(args.model, "cuda", True)

    # Define dynamic axes for ONNX export
    dynamic_axes = AXES_NAME[args.model]

    # Postprocess the dynamic axes to match the input and output names with DeviceModel
    if args.onnx_load_path == "":
        update_dynamic_axes(args.model, dynamic_axes)

    compilation_args = DYNAMIC_SHAPES[args.model]

    # We only need to remove the nesting for SDXL models as they contain the nested input added_cond_kwargs
    # which are renamed by the DeviceModel
    if args.onnx_load_path != "" and args.model in ["sdxl-1.0", "sdxl-turbo"]:
        remove_nesting(compilation_args)

    # Define deployment configuration
    deployment = {
        "runtime": "TRT",
        "version": "10.3",
        "precision": "stronglyTyped",
        "onnx_opset": "17",
    }

    client = RuntimeRegistry.get(deployment)

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(
        model=backbone,
        dummy_input=dummy_inputs,
        onnx_load_path=args.onnx_load_path,
        dynamic_axes=dynamic_axes,
        onnx_opset=int(deployment["onnx_opset"]),
        remove_exported_model=False,
        dq_only=args.dq_only,
    )

    if not args.trt_engine_load_path:
        # Compile the TRT engine from the exported ONNX model
        compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
        # Save TRT engine for future use
        with open(f"{args.model}.plan", "wb") as f:
            # Remove the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            f.write(compiled_model[SHA_256_HASH_LENGTH:])
    else:
        with open(args.trt_engine_load_path, "rb") as f:
            compiled_model = f.read()
            # Prepend the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            compiled_model = prepend_hash_to_bytes(compiled_model)

    # The output shapes will need to be specified for models with dynamic output dimensions
    device_model = DeviceModel(
        client,
        compiled_model,
        metadata,
        compilation_args,
        get_io_shapes(args.model, args.onnx_load_path),
    )

    if hasattr(pipe, "unet") and add_embedding:
        setattr(device_model, "add_embedding", add_embedding)

    # Move the backbone back to the CPU and set the backbone to the compiled device model
    backbone.to("cpu")
    if hasattr(pipe, "unet"):
        pipe.unet = device_model
    elif hasattr(pipe, "transformer"):
        pipe.transformer = device_model
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")
    pipe.to("cuda")

    generate_image(pipe, args.prompt, args.model)
    print(f"Image generated using {args.model} model saved as {args.model}.png")

    print(f"Inference latency of the backbone of the pipeline is {device_model.get_latency()} ms")


if __name__ == "__main__":
    torch.serialization.safe_globals([PercentileCalibrator])
    main()
