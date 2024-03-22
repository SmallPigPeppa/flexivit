python vanilla_eval_resize_input.py.py \
  --accelerator gpu \
  --devices 8 \
  --precision 16 \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224 \
  --data.root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --data.num_classes 1000 \
  --model.patch_size 16 \
  --data.size 224 \
  --data.crop_pct 0.9 \
  --data.batch_size 256 \
  --model.results_path ./result.csv





