python vanilla_eval_resize_weight_vitb21k_rebuttal_mspe_jt.py \
  --max_epochs 5 \
  --accelerator gpu \
  --works 8 \
  --precision 16 \
  --batch_size 64 \
  --log_every_n_steps 1 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type pi \
  --model.results_path ./L2P_exp/vit_base_patch16_224.augreg2_in21k_ft_in1k_pi_pos.csv









