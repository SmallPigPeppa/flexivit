#python vanilla_eval_resize_input.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 8 \
#  --precision 16 \
#  --model.resize_type pi \
#  --model.weights vit_base_patch16_224 \
#  --data.root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --data.num_classes 1000 \
#  --model.patch_size 16 \
#  --data.size 224 \
#  --data.crop_pct 0.9 \
#  --data.batch_size 256 \
#  --model.results_path ./result.csv

#  --precision 16 \
#  --data.crop_pct 0.9 \
#  --data.batch_size 256 \
#vit_base_patch16_224.augreg_in21k_ft_in1k
#  --data.root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --data.num_classes 1000 \
python vanilla_eval_resize_input2.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 8 \
  --works 4 \
  --batch_size 256 \
  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.results_path ./vit_base_patch16_224.augreg2_in21k_ft_in1k.csv


