import timm
from timm import create_model
import inspect
import timm.models.pvt_v2
import timm.models.swin_transformer
import timm.models.mvitv2

model_name = 'pvt_v2_b3.in1k'
model_name = 'mvitv2_small.fb_in1k'
# 创建模型实例
net = create_model(model_name, pretrained=True)
print(net.default_cfg["architecture"])
# 获取模型构建函数
model_fn = getattr(timm.models, net.default_cfg["architecture"])
# 使用inspect获取函数的签名
sig = inspect.signature(model_fn)
# 打印函数支持的参数
print(sig)
