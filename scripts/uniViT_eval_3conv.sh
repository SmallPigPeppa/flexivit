#ckpt_path=ckpt/uniViT/no-random-resize-4conv/last.ckpt
#ckpt_path=ckpt/uniViT/add_random_resize_4conv_fix14token/last.ckpt
ckpt_path=ckpt/uniViT/add_random_resize_3conv_fix14token_learnembed/last.ckpt


python uniViT_3conv_eval_fix_14token.py \
  --max_epochs 15 \
  --precision 16 \
  --accelerator gpu \
  --devices 8 \
  --works 4 \
  --batch_size 64 \
  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
  --ckpt_path ${ckpt_path} \
  --model.resize_type pi \
  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --model.num_classes 1000 \
  --model.patch_size 16 \
  --model.image_size 224


#python uniViT_3conv_eval_fix_anchor.py \
#  --max_epochs 15 \
#  --precision 16 \
#  --accelerator gpu \
#  --devices 8 \
#  --works 4 \
#  --batch_size 64 \
#  --root /mnt/mmtech01/dataset/lzy/ILSVRC2012 \
#  --ckpt_path ${ckpt_path} \
#  --model.resize_type pi \
#  --model.weights vit_base_patch16_224.augreg2_in21k_ft_in1k \
#  --model.num_classes 1000 \
#  --model.patch_size 16 \
#  --model.image_size 224


