ckpt_paths=(
#    "ckpt/L2P/add_random_resize_4conv_fix14token_24816_ratio/last.ckpt"
    "ckpt/L2P/add_random_resize_4conv_fix14token_2range_ratio/last.ckpt"
#    "ckpt/L2P/add_random_resize_4conv_fix14token_2range/last.ckpt"
#    "ckpt/L2P/add_random_resize_4conv_fix14token_24816/last.ckpt"
)

scripts=(
#    "L2P_4conv_eval_fix_14token.py"
    "L2P_4conv_eval_fix_14token_ratio.py"
#    "L2P_4conv_eval_fix_anchor.py"
#    "L2P_4conv_eval_fix_14token_24816.py"
#    "L2P_4conv_eval_fix_anchor_24816.py"
)


for ckpt_path in "${ckpt_paths[@]}"; do
    for script in "${scripts[@]}"; do
        python ${script} \
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
    done
done
