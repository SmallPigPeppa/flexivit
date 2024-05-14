
python L2P_train_add_random_resize_4conv_fix14token_2range_deit3b.py \
  --max_epochs 5 \
  --precision 16 \
  --accelerator gpu \
  --devices 8 \
  --works 8 \
  --batch_size 32 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.resize_type pi \
  --model.weights deit3_base_patch16_224.fb_in22k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.results_path ./resize_weight.csv

python /ppio_net0/code/openapi.py stop 4fca613d27e9ae5f






