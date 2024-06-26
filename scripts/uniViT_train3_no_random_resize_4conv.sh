python uniViT_train3_no_random_resize_4conv.py \
  --max_epochs 5 \
  --precision 16 \
  --accelerator gpu \
  --devices 8 \
  --works 4 \
  --batch_size 64 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.results_path ./resize_weight.csv



