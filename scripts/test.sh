python eval_new_new.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 8 \
  --precision 16 \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg_in21k_ft_in1k \
  --data.root /ppio_net0/torch_ds/imagenet \
  --data.num_classes 1000 \
  --model.patch_size 16 \
  --model.ckpt_path ./vit_base_patch16_224.augreg_in21k_ft_in1k.pth \
  --data.size 224 \
  --data.crop_pct 0.9 \
  --data.batch_size 256 \
  --model.results_path ./result.csv




#python eval_new_new.py \
#  --accelerator gpu \
#  --devices 8 \
#  --precision 16 \
#  --model.resize_type pi \
#  --model.weights vit_base_patch16_224 \
#  --data.root /ppio_net0/torch_ds/imagenet/val \
#  --data.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.ckpt_path ./vit_base_patch16_224.pth \
#  --data.size 224 \
#  --data.crop_pct 0.9 \
#  --data.batch_size 256 \
#  --model.results_path ./result.csv

