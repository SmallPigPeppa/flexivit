from timm import create_model

# model_name = 'vit_base_patch16_224'
# model_name = 'deit_base_distilled_patch16_224.fb_in1k'
# model_name = 'vit_base_patch16_clip_224.openai_ft_in1k'
# model_name = 'pvt_v2_b3.in1k'
# model_name = 'mobilenetv3_small_050.lamb_in1k'
# model_name = 'resnet18.a1_in1k'
model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
import timm
if __name__ == '__main__':
    net = create_model(model_name, pretrained=True)
    print(net.default_cfg["architecture"])
    model_fn = getattr(timm.models, net.default_cfg["architecture"])
    net = model_fn(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        dynamic_img_size=True
    )