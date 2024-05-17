python L2P_4conv_eval_fix_14token_mvitV2_piresize_448.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 4 \
  --batch_size 64 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights mvitv2_small.fb_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type pi \
  --model.results_path ./L2P_exp/vanilla_resize_weight_pi_deit3b.csv


python L2P_4conv_eval_mvitV2_piresize_448.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 4 \
  --batch_size 64 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights mvitv2_small.fb_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type interpolate \
  --model.results_path ./L2P_exp/vanilla_resize_weight_bilinear_deit3b.csv





