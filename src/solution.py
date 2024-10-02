from typing import Optional
from enum import Enum

import torch
import torch._prims as prims

# Construct the full range of bfloat16
@torch.jit.script
def bfloat16_to_bits(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    x = prims.view_element_type(x, torch.uint16)
    return x.to(dtype=torch.int32)

@torch.jit.script
def bits_to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int32
    x = x.to(dtype=torch.uint16)
    return prims.view_element_type(x, torch.bfloat16)

# Compositions functions for bfloat16, e4m3, e5m2
def compose_16bit(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 15) + (exponent << 7) + mantissa

def compose_e4m3(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 7) + (exponent << 3) + mantissa

def compose_e5m2(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 7) + (exponent << 2) + mantissa


# Decomposition functions for bfloat16, e4m3, e5m2
def decompose_16bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.int32
    sign = (x & 0b1000_0000_0000_0000) >> 15
    exponent = (x & 0b0111_1111_1000_0000) >> 7
    mantissa = (x & 0b0000_0000_0111_1111)
    return sign, exponent, mantissa

def decompose_8bit_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.int32
    sign =     (x & 0b1000_0000) >> 7
    exponent = (x & 0b0111_1000) >> 3
    mantissa = (x & 0b0000_0111)
    return sign, exponent, mantissa

def decompose_8bit_e5m2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.int32
    sign =     (x & 0b1000_0000) >> 7
    exponent = (x & 0b0111_1100) >> 2
    mantissa = (x & 0b0000_0011)
    return sign, exponent, mantissa

# Encode and decode functions for e5m2
def encode_as_e5m2(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    # Quantize to e5m2
    # Subtract bfloat16 bias
    e = (e - 127) % 0b1_0000_0000
    # Add e5m2 bias
    e = (e + 15) % 0b1_00_000
    # chop mantissa
    m = m >> 5
    return (s << 7) + (e << 2) + m

def decode_from_e5m2(encoded: torch.Tensor) -> torch.Tensor:
    assert encoded.dtype == torch.int32
    s, e, m = decompose_8bit_e5m2(encoded)
    # Update mantissa
    e = e + (127 - 15) % 0b1_0000_0000
    # Expand mantissa
    m = m << 5
    return (s << 15) + (e << 7) + m

def encode_as_e5m2_round_up(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    # Quantize to e5m2
    # Subtract bfloat16 bias
    e = (e - 127) % 0b1_0000_0000
    # Add e5m2 bias
    e = (e + 15) % 0b1_00_000
    # Round up mantissa
    m = (m >> 5) + 1
    # Handle overflow
    overflow = m == 0b100
    e = torch.where(overflow, e + 1, e)
    m = torch.where(overflow, 0, m)
    # Compose the result
    return compose_e5m2(s, e, m)

def encode_as_e5m2_stochastic(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    
    # Extract sign, exponent, and mantissa directly from bfloat16
    t_bits = t.view(torch.int16)
    s = (t_bits & 0x8000) >> 15
    e = (t_bits & 0x7F80) >> 7
    m = t_bits & 0x007F
    
    # Quantize to e5m2
    # Subtract bfloat16 bias and add e5m2 bias
    e = (e - 127 + 15) % 0b1_00_000
    
    # Calculate the probability for stochastic rounding
    p = m.float() / 128.0
    
    # Generate random values for stochastic rounding
    random_values = torch.rand_like(p)
    
    # Perform stochastic rounding on mantissa
    m_rounded = (m >> 5) + (random_values < p).to(torch.int32)
    
    # Handle overflow
    overflow = m_rounded == 0b100
    e = torch.where(overflow, e + 1, e)
    m_rounded = torch.where(overflow, 0, m_rounded)
    
    # Compose the result
    return compose_e5m2(s, e, m_rounded)

# @torch.jit.script
def round_to_fp8_represented_as_int8(
        t: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert t.dtype == torch.bfloat16

    if out is None:
        out = torch.zeros_like(t, dtype=torch.uint8)
    
    if n_mantissa == 2:
        # Use encode_as_e5m2_stochastic for E5M2 format
        fp8_int = encode_as_e5m2_stochastic(t)
    else:
        # TODO: Implement E4M3 encoding if needed
        raise NotImplementedError("E4M3 encoding not implemented yet")
    
    # Convert the result to uint8
    out = fp8_int.to(torch.uint8)
    
    return out

# @torch.jit.script
def undo_int8_fp8(
        fp8_tensor: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert fp8_tensor.dtype == torch.uint8

    if out is None:
        out = torch.zeros_like(fp8_tensor, dtype=torch.bfloat16)

    if n_mantissa == 2:
        # Convert uint8 to int32 for decoding
        fp8_int = fp8_tensor.to(torch.int32)
        # Decode E5M2 format
        bfloat16_bits = decode_from_e5m2(fp8_int)
        # Convert bits to bfloat16
        out = bits_to_bfloat16(bfloat16_bits)
    else:
        # TODO: Implement E4M3 decoding if needed
        raise NotImplementedError("E4M3 decoding not implemented yet")

    return out
