model=$1
steps=$2
python diffusion/optimisation/quantization/quantize.py \
  --model $model \
  --format int8 --batch-size 2 \
  --calib-size 32 --collect-method global_min \
  --percentile 1.0 --alpha 0.8 \
  --quant-level 3.0 --n-steps $steps \
  --model-dtype Half \
  --quantized-torch-ckpt-save-path ./"$model"_int8.pt --onnx-dir onnx_"$model"
