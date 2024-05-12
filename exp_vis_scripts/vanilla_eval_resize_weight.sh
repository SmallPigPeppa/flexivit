python exp_vis/vanilla_eval_resize_weight.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 1 \
  --works 8 \
  --batch_size 64 \
  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.resize_type pi \
  --model.results_path exp_vis/debug.csv



