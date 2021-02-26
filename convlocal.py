import math

from typing import Union, Tuple

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.functional import unfold

# Python类型提示（Type Hints）   基本类型有str list dict等等
# 参考 https://blog.csdn.net/weixin_41931548/article/details/89640223?spm=1001.2101.3001.4242   
#      https://blog.csdn.net/ypgsh/article/details/84992461
Pairable = Union[int, Tuple[int, int]]


def conv2d_local(input: torch.Tensor, 
                 weight: torch.Tensor,
                 bias=None,
                 padding: Pairable=0,
                 stride: Pairable=1,
                 dilation: Pairable=1,
                 data_format: str="channels_first"):
    """Calculate the local convolution.
    Args:
        input:
        weight:
        bias:
        padding:
        stride:
        dilation:
        data_format: For Keras compatibility
    Returns:
    """
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    out_height, out_width, out_channels, in_channels, kernel_height, kernel_width = weight.size()
    kernel_size = (kernel_height, kernel_width)

    # N x [in_channels * kernel_height * kernel_width] x [out_height * out_width]
    if data_format == "channels_first":
        cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
        reshaped_input = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
    else:
        # Taken from `keras.backend.tensorflow_backend.local_conv2d`
        stride_y, stride_x = _pair(stride)
        feature_dim = in_channels * kernel_height * kernel_width
        xs = []
        for i in range(out_height):
            for j in range(out_width):
                y = i * stride_y
                slice_row = slice(y, y + kernel_size[0])
                x = j * stride_x
                slice_col = slice(x, x + kernel_size[1])
                val = input[:, slice_row, slice_col, :].contiguous()
                xs.append(val.view(input.shape[0], 1, -1, feature_dim))
        concated = torch.cat(xs, dim=1)
        reshaped_input = concated

    output_size = out_height * out_width
    input_size = in_channels * kernel_height * kernel_width
    weights_view = weight.view(output_size, out_channels, input_size)
    permuted_weights = weights_view.permute(0, 2, 1)

    out = torch.matmul(reshaped_input, permuted_weights)
    out = out.view(reshaped_input.shape[0], out_height, out_width, out_channels).permute(0, 3, 1, 2)
    if data_format == "channels_last":
        out = out.permute(0, 2, 3, 1)

    if bias is not None:
        # 这里可以用广播机制实现
        final_bias = bias.expand_as(out)
        out = out + final_bias

    return out


class Conv2dLocal(Module):
    """A 2D locally connected layer.
    Attributes:
        weight (torch.Tensor): The weights. out_height x out_width x out_channels x in_channels x kernel_height x kernel_width
        kernel_size (Tuple[int, int]): The height and width of the convolutional kernels.
        stride (Tuple[int, int]): The stride height and width.
    """

    def __init__(self, in_height: int, in_width: int, 
                 in_channels: int, out_channels: int,
                 kernel_size: Pairable,
                 stride: Pairable = 1,
                 padding: Pairable = 0,
                 bias: bool = True,
                 dilation: Pairable = 1,
                 data_format="channels_first"):
        super(Conv2dLocal, self).__init__()

        self.data_format = data_format
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            if self.data_format == "channels_first":
                self.bias = Parameter(torch.Tensor(out_channels, self.out_height, self.out_width))
            else:
                self.bias = Parameter(torch.Tensor(self.out_height, self.out_width, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    # 使用装饰器将 将方法转换成属性， 使得属性不用直接暴漏在外面 
    # 参考 https://blog.csdn.net/dxk_093812/article/details/83212231
    @property 
    def input_shape(self):
        """The expected input shape for this module."""
        if self.data_format == "channels_first":
            shape = (self.in_channels, self.in_height, self.in_width)
        else:
            shape = (self.in_height, self.in_width, self.in_channels)
        return torch.Tensor(shape)

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input: torch.Tensor):
        return conv2d_local(input=input, 
                            weight=self.weight, 
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            data_format=self.data_format
                            )


class Conv1dLocal(Conv2dLocal):
    """A 1D locally connected layer.
    input.shape must be [batch_size, time_steps, 1 ,in_channels],
    if input.shape is [batch_size, in_channels, time_steps, 1], the parameter that data_format need to be modefied to channels_first
    """

    def __init__(self, in_height, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        two_dimensional_kernel = (kernel_size, 1)
        two_dimensional_stride = (stride, 1)
        two_dimensional_padding = (padding, 0)
        two_dimensional_dilation = (dilation, 1)
        super().__init__(in_height, 1, in_channels, out_channels, two_dimensional_kernel,
                         stride=two_dimensional_stride,
                         padding=two_dimensional_padding,
                         dilation=two_dimensional_dilation,
                         bias=bias,
                         data_format="channels_last")


class Flatten(Module):

    def forward(self, input: torch.Tensor):
        return input.view(-1)

if __name__ == '__main__':
    # 例 in_height 是序列长度 
    model = Conv1dLocal(in_height=12, in_channels=2, out_channels=5,
                    kernel_size=2)
    # 输入[batch_size, num_steps, 1, features] 1是固定的，因为序列数据的维度本来就是[num_steps, 1]
    input = torch.randn(256,12,1,2)
    output = model(input)
    print(f'model:{model}')
    print(f'input.shape:{input.shape}')
    print(f'output.shape:{output.shape}')

    for par in model.named_parameters():
        print(par[0],par[1].shape)

    print(f'model.input_shape:{model.input_shape}')