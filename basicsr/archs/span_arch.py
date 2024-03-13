from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F


# from basicsr.utils.registry import ARCH_REGISTRY


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.f_mul_x = nn.quantized.FloatFunctional()
        self.inplace = inplace
    
    """export-friendly version of SiLU"""
    
    # @staticmethod
    def forward(self, x):
        if self.inplace:
            result = x.clone()
            x = torch.sigmoid(x)
            return self.f_mul_x.mul(result, x)
        return self.f_mul_x.mul(x, torch.sigmoid(x))


class ACT2(nn.Module):
    def __init__(self):
        super(ACT2, self).__init__()
        self.f_add_half = nn.quantized.FloatFunctional()
        self.f_add_x = nn.quantized.FloatFunctional()
        self.f_mul_sim = nn.quantized.FloatFunctional()
        self.quant_mean = torch.quantization.QuantStub()
        self.mean = torch.Tensor([-0.5])
    
    def forward(self, x, input):
        mean_q = self.quant_mean(self.mean.to(x.device))
        out = self.f_mul_sim.mul(self.f_add_x.add(input, x),
                                 self.f_add_half.add(torch.sigmoid(input), mean_q))
        return out


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC_QAT(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, bias=True):
        super(Conv3XC_QAT, self).__init__()
        self.inp_planes = c_in
        self.out_planes = c_out
        self.block = [nn.Conv2d(self.inp_planes, self.out_planes, k, s, p, bias=bias)]
        self.block = nn.Sequential(*self.block)
    
    def forward(self, x):
        x = self.block(x)
        return x


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1
        
        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0,
                      bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )
        
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()
    
    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()
        
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        
        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1,
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        
        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        sk_w = F.pad(sk_w, [1, 1, 1, 1])
        
        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b
    
    def forward(self, x):
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0.0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)
        return out
    
    def rep_param(self):
        w1 = self.conv[0].weight.data.detach()
        b1 = self.conv[0].bias.data.detach()
        w2 = self.conv[1].weight.data.detach()
        b2 = self.conv[1].bias.data.detach()
        w3 = self.conv[2].weight.data.detach()
        b3 = self.conv[2].bias.data.detach()
        
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        
        weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2,
                                                                                                               3)
        bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        
        sk_w = self.sk.weight.data.detach()
        sk_b = self.sk.bias.data.detach()
        sk_w = F.pad(sk_w, [1, 1, 1, 1])
        
        RK = weight_concat + sk_w
        RB = bias_concat + sk_b
        
        return RK, RB


class SPAB_QAT(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB_QAT, self).__init__()
        if mid_channels is None:
            self.mid_channels = in_channels
        else:
            self.mid_channels = mid_channels
        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=3, padding=1,
                              stride=1, bias=True)
        backbone = []
        backbone += [SiLU(inplace=inplace)]
        backbone += [
            nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1,
                      bias=True)]
        backbone += [SiLU(inplace=inplace)]
        backbone += [
            nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1,
                      bias=True)]
        self.act2 = ACT2()
        
        self.backbone = nn.Sequential(*backbone)
    
    def forward(self, x):
        out1 = self.conv(x)
        out3 = self.backbone(out1)
        
        out = self.act2(x, out3)
        
        return out, out1,


class SPAB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = SiLU(inplace=True)
        self.act2 = ACT2()
    
    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        
        out3 = self.c3_r(out2_act)
        out = self.act2(x, out3)
        
        return out, out1


# @ARCH_REGISTRY.register()
class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """
    
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 upscale=4,
                 bias=True
                 ):
        super(SPAN, self).__init__()
        
        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        
        self.conv_1 = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)
        
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)
        self.upsampler = pixelshuffle_block(feature_channels, self.out_channels, upscale_factor=upscale)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.f_cat = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        x = self.quant(x)
        
        out_feature = self.conv_1(x)
        
        out_b1, out_feature_2 = self.block_1(out_feature)
        out_b2, out_b1_2 = self.block_2(out_b1)
        out_b3, out_b2_2 = self.block_3(out_b2)
        
        out_b4, out_b3_2 = self.block_4(out_b3)
        out_b5, out_b4_2 = self.block_5(out_b4)
        out_b6, out_b5_2 = self.block_6(out_b5)
        
        out_b6_2 = self.conv_2(out_b6)
        out = self.conv_cat(self.f_cat.cat([out_feature, out_b6_2, out_b1, out_b5_2], 1))
        output = self.upsampler(out)
        output = torch.clamp(output, min=0.0, max=255.0)
        output = self.dequant(output)
        
        return output
    
    def fuse_model(self):
        for name, module in self.named_children():
            if isinstance(module, SPAB):
                for n, m in module.named_children():
                    if isinstance(m, Conv3XC):
                        RK, RB = m.rep_param()
                        conv = Conv3XC_QAT(m.eval_conv.in_channels, m.eval_conv.out_channels, m.eval_conv.kernel_size,
                                           m.eval_conv.stride, m.eval_conv.padding).to(RK.device)
                        conv.block[0].weight.data = RK
                        conv.block[0].bias.data = RB
                        setattr(module, n, conv)
            elif isinstance(module, Conv3XC):
                RK, RB = module.rep_param()
                conv = Conv3XC_QAT(module.eval_conv.in_channels, module.eval_conv.out_channels,
                                   module.eval_conv.kernel_size,
                                   module.eval_conv.stride, module.eval_conv.padding).to(RK.device)
                conv.block[0].weight.data = RK
                conv.block[0].bias.data = RB
                setattr(self, name, conv)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import time
    
    model = SPAN(1, 1, upscale=2, feature_channels=48)
    # model.eval()
    inputs = torch.rand(1, 1, 32, 32)
    
    model.fuse_model()
    backend = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(model, inplace=True)
    
    # sr_inputs = (torch.rand(1, 1, 256, 256).cuda(),)
    sr = model(inputs)
    model.eval()
    saved_model = torch.quantization.convert(model, inplace=False)
    # int_sr = saved_model(inputs)
    # Convert to TorchScript
    # scripted_model = torch.jit.script(saved_model)
    # 导出模型为ONNX格式，将dynamic_axes参数设置为适应动态输入
    onnx_filename = "imageEn-SPAN-test.onnx"
    dynamic_axes = {'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    torch.onnx.export(saved_model, inputs, onnx_filename, verbose=False,  # 输出详细信息
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=dynamic_axes,
                      do_constant_folding=True,
                      export_params=True,
                      opset_version=13)  # producer_name="TICODE", producer_version=1.0 )
    # print(flop_count_table(FlopCountAnalysis(model, inputs)))
