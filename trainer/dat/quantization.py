
import torch
import os

class QuantizerBase:
    pass

class RandomQuantizer(QuantizerBase):
    def __init__(self):
        self.b = 7
        self.s = torch.pow(torch.tensor(2), self.b) - 1
    def quantize(self, g):
        ### g is input to be quantized
        ### b is # of bits
        s = self.s  ## number of quantization levels
        norm = torch.norm(g)
        g_normalized = torch.abs(g) / norm
        l = torch.floor(g_normalized * s)
        p = (s * g_normalized - l)
        xi = (l + torch.distributions.binomial.Binomial(1, p).sample())*2 + (torch.sign(g) + 1) / 2
        xi = xi.byte()
        # p = (s * g_normalized - l)
        # xi = l + torch.distributions.binomial.Binomial(1, p).sample()
        # q = torch.norm(g) * torch.sign(g) * xi
        return xi, norm
    def dequantize(self, xi , norm):
        sign = torch.fmod(xi, 2).float()

        sign = sign * 2 - 1
        xi = (xi / 2).float()
        return norm * sign * xi / self.s

class Quantizer(QuantizerBase):
    def __init__(self):
        self.num_bits = 8
        pass
    def quantize(self, x):
        qmin = torch.tensor(0.).cuda()
        qmax = torch.tensor(2. ** self.num_bits - 1.).cuda()
        min_val, max_val = x.min(), x.max()

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale


        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = torch.floor(zero_point)
        q_x = zero_point + x / scale
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        return q_x, scale, zero_point

    def dequantize(self, q_x, scale, zero_point):
        return scale * (q_x.tensor.float() - zero_point)

