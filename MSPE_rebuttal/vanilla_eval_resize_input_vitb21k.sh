#python vanilla_eval_resize_weight_deit3b.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 1 \
#  --works 4 \
#  --batch_size 64 \
#  --root /ppio_net0/torch_ds/imagenet \
#  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224 \
#  --model.resize_type pi \
#  --model.results_path ./L2P_exp/vanilla_resize_weight_pi_deit3b.csv


python vanilla_eval_resize_weight_vitb21k_rebuttal.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 4 \
  --batch_size 64 \
  --root /ppio_net0/torch_ds/imagenet \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type pi \
  --model.results_path ./L2P_exp/vit_base_patch16_224.augreg2_in21k_ft_in1k_pi.csv






