#python vanilla_eval_resize_input_new.py \
#  --max_epochs 1 \
#  --accelerator gpu \
#  --devices 8 \
#  --works 4 \
#  --batch_size 32 \
#  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --model.resize_type pi \
#  --model.weights pvt_v2_b3.in1k \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224 \
#  --model.results_path ./L2P_exp/vanilla_resize_input_pvt.csv



python vanilla_eval_resize_input_new.py \
  --max_epochs 1 \
  --accelerator gpu \
  --devices 8 \
  --works 4 \
  --batch_size 32 \
  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --model.resize_type pi \
  --model.weights mvitv2_small.fb_in1k  \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224 \
  --model.results_path ./L2P_exp/vanilla_resize_input_mvit.csv


