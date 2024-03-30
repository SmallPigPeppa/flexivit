python uniViT_eval_fix_weight.py \
  --max_epochs 15 \
  --precision 16 \
  --accelerator gpu \
  --devices 8 \
  --works 4 \
  --batch_size 8 \
  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.results_path ./uniViT_fix_weight.csv



