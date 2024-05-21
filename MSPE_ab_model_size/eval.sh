ckpt_paths=(
    "ckpt/MSPE-ab/deit3s/last.ckpt"
)
scripts=(
     "L2P_4conv_eval_fix_14token.py"
)


for ckpt_path in "${ckpt_paths[@]}"; do
    for script in "${scripts[@]}"; do
        python ${script} \
            --max_epochs 15 \
            --precision 16 \
            --accelerator gpu \
            --devices 1 \
            --works 8 \
            --batch_size 64 \
            --root /ppio_net0/torch_ds/imagenet \
            --ckpt_path ${ckpt_path} \
            --model.resize_type pi \
            --model.weights deit3_small_patch16_224.fb_in22k_ft_in1k \
            --model.num_classes 1000 \
            --model.patch_size 16 \
            --model.image_size 224
    done
done
