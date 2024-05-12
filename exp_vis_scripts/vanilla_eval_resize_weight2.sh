python vv_exp_vis_vanilla_eval_resize_weight2.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 8 \
  --batch_size 64 \
  --root /ppio_net0/torch_ds/imagenet \
  --ckpt_path ckpt/MSP/add_random_resize_4conv_fix14token_2range/last.ckpt \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \




