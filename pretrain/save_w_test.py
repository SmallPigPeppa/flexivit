import torch
from timm import create_model

model_name = 'vit_base_patch16_224'
model_name = 'deit_base_distilled_patch16_224.fb_in1k'
# model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
if __name__ == '__main__':
    net = create_model(model_name, pretrained=True)
    print(net)
    # state_dict = create_model(model_name, pretrained=True).state_dict()
    # new_patch_size = (16, 16)
    # image_size = 224
    # net = create_model(
    #     model_name, img_size=image_size, patch_size=new_patch_size
    # )
    # net.load_state_dict(state_dict, strict=True)
    # model_path = f'{model_name}.pth'
    # torch.save(net.state_dict(), model_path)
    #
    # print(f'Model parameters saved to {model_path}')
