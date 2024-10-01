from typing import Optional
from enum import Enum

import torch

class RoundingMode(Enum):
    UP = 'up'
    DOWN = 'down'
    STOCHASTIC = 'stochastic'

def custom_round(
        sign: torch.Tensor,
        out: torch.Tensor,
        mode: RoundingMode.UP
) -> None:
    """
    Rounds a given tensor that represents the bytes
    of a bfloat16 number as int32 to another bfloat16
    number out
    """
    assert t.dtype == torch.int32
    assert out.dtype == torch.int32
    assert n_mantissa in [2, 3]
    
    pass

# Construct the full range of bfloat16
def bits_to_bfloat16(x):
    assert x.dtype == torch.int32
    x = x.to(dtype=torch.uint16)
    return x.view(dtype=torch.bfloat16)

def bfloat16_to_bits(x):
    x = x.view(dtype=torch.uint16)
    return x.to(dtype=torch.int32)

# Compositions functions for bfloat16, e4m3, e5m2
def compose_bfloat16(sign, exponent, mantissa):
    return (sign << 15) + (exponent << 7) + mantissa

def compose_e4m3(sign, exponent, mantissa):
    return (sign << 7) + (exponent << 3) + mantissa

def compose_e5m2(sign, exponent, mantissa):
    return (sign << 7) + (exponent << 2) + mantissa

# Decomposition functions for bfloat16, e4m3, e5m2
def decompose_16bit(x):
    assert x.dtype == torch.int32
    sign = (x & 0b1000_0000_0000_0000) >> 15
    exponent = (x & 0b0111_1111_1000_0000) >> 7
    mantissa = (x & 0b0000_0000_0111_1111)
    return sign, exponent, mantissa

def decompose_8bit_e4m3(x):
    assert x.dtype == torch.int32
    sign =     (x & 0b1000_0000) >> 7
    exponent = (x & 0b0111_1000) >> 3
    mantissa = (x & 0b0000_0111)
    return sign, exponent, mantissa

def decompose_8bit_e5m2(x):
    assert x.dtype == torch.int32
    sign =     (x & 0b1000_0000) >> 7
    exponent = (x & 0b0111_1100) >> 2
    mantissa = (x & 0b0000_0011)
    return sign, exponent, mantissa

@torch.jit.script
def round_to_fp8_represented_as_int8(
        t: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert t.dtype == torch.bfloat16
    assert out == None or \
        (out.dtype == torch.uint8 and t.shape == out.shape)

    if out is None:
        out = torch.zeros_like(t)
    
    return out

@torch.jit.script
def undo_int8_fp8(
        fp8_tensor: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert fp8_tensor.dtype == torch.uint8
    assert out == None or \
        (out.dtype == torch.uint8 and fp8_tensor.shape == out.shape)

    if out is None:
        out = torch.zeros_like(fp8_tensor)
    
