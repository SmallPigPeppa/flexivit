import timm
from timm import create_model

weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
model = create_model(weights, pretrained=True)
data_config = timm.data.resolve_model_data_config(model)
print(data_config)
