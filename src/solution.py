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

def decode_from_e5m2(encoded: torch.Tensor) -> torch.Tensor:
    assert encoded.dtype == torch.int32
    s, e, m = decompose_8bit_e5m2(encoded)
    # Handle zero
    is_zero = (e == 0) & (m == 0)
    # Update exponent
    e = e + (127 - 15) % 0b1_0000_0000
    # Expand mantissa
    m = m << 5
    # Compose the result
    result = (s << 15) + (e << 7) + m
    # Set zero values
    return torch.where(is_zero, 0, result)

def decode_from_e4m3(fp8_int: torch.Tensor) -> torch.Tensor:
    # Extract sign, exponent, and mantissa
    s = (fp8_int >> 7) & 0b1
    e = (fp8_int >> 3) & 0b1111
    m = fp8_int & 0b111

    # Handle special cases
    is_zero = (e == 0) & (m == 0)
    is_inf = (e == 15) & (m == 0)
    is_nan = (e == 15) & (m != 0)

    # Adjust exponent and mantissa
    e_adjusted = torch.where(e != 0, e + 127 - 7, 0)
    m_adjusted = torch.where(e != 0, m | 0b100, m)

    # Shift mantissa to bfloat16 position (3 bits to 7 bits)
    m_shifted = m_adjusted << 4

    # Compose bfloat16 bits
    bfloat16_bits = (s << 15) | (e_adjusted << 7) | m_shifted

    # Handle special cases
    bfloat16_bits = torch.where(is_zero, s << 15, bfloat16_bits)
    bfloat16_bits = torch.where(is_inf, (s << 15) | (0xFF << 7), bfloat16_bits)
    bfloat16_bits = torch.where(is_nan, (s << 15) | (0xFF << 7) | 1, bfloat16_bits)

    return bfloat16_bits
def encode_as_e4m3_round_up(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
    
    # Quantize to e4m3
    e = (e - 127) % 0b1_0000_0000  # Subtract bfloat16 bias
    e = (e + 7) % 0b1_0000  # Add e4m3 bias
    
    # Round up mantissa
    m = (m >> 4) + 1
    
    # Handle overflow
    overflow = m == 0b1000
    e = torch.where(overflow, e + 1, e)
    m = torch.where(overflow, 0, m)
    
    # Compose the result
    result = (s << 7) | (e << 3) | m
    
    # Set zero values
    return torch.where(is_zero, 0, result)

def encode_as_e4m3_trunc(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
    
    # Quantize to e4m3
    e = (e - 127) % 0b1_0000_0000  # Subtract bfloat16 bias
    e = (e + 7) % 0b1_0000  # Add e4m3 bias
    
    # Truncate mantissa
    m = m >> 4
    
    # Compose the result
    result = (s << 7) | (e << 3) | m
    
    # Set zero values
    return torch.where(is_zero, 0, result)

def encode_as_e5m2_stochastic(t: torch.Tensor, mode: str = 'trunc') -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    
    # Handle zero values
    is_zero = (e == 0) & (m == 0)

    # Quantize to e5m2
    e_unbiased = (e.to(torch.int32) - 127).to(torch.int32)  # Unbias bfloat16 exponent
    
    # Identify subnormal values in E5M2 range
    is_subnormal = e_unbiased < -14  # E5M2 has a bias of 15, so anything less than 2^-14 is subnormal
    
    # Adjust exponent and mantissa for E5M2
    e_e5m2 = torch.clamp(e_unbiased + 15, min=0, max=31).to(torch.int32)  # Add E5M2 bias and clamp
    m_e5m2 = m

    # Handle subnormal values
    subnormal_shift = torch.clamp(-14 - e_unbiased, min=0, max=7)
    m_e5m2 = torch.where(is_subnormal, (m_e5m2 | (1 << 7)) >> subnormal_shift, m_e5m2)
    e_e5m2 = torch.where(is_subnormal, 0, e_e5m2)
    
    # Calculate the distance to the next representable number
    distance = m & 0b11111  # Last 5 bits of mantissa
    
    if mode == 'stochastic':
        # Calculate probability of rounding up
        prob_round_up = distance.float() / 32.0
        
        # Generate random values for stochastic rounding
        random_values = torch.rand_like(t, dtype=torch.float32)
        
        # Determine whether to round up or truncate
        round_up = random_values < prob_round_up
    elif mode == 'trunc':
        round_up = torch.zeros_like(t, dtype=torch.bool)
    elif mode == 'round_up':
        round_up = torch.ones_like(t, dtype=torch.bool)
    else:
        raise ValueError("Invalid mode. Choose 'stochastic', 'trunc', or 'round_up'.")
    
    # Truncate mantissa to 2 bits
    m_truncated = m_e5m2 >> 5

    m_rounded_up = m_truncated + 1
    
    m_e5m2 = torch.where(round_up, m_rounded_up, m_truncated)
    
    # Handle overflow
    overflow = m_e5m2 == 0b100
    e_e5m2 = torch.where(overflow & ~is_subnormal, e_e5m2 + 1, e_e5m2)
    m_e5m2 = torch.where(overflow, 0, m_e5m2)
    
    # Compose the result
    result = compose_e5m2(s, e_e5m2, m_e5m2)
    import pdb; pdb.set_trace()
    
    # Set zero values
    return torch.where(is_zero, 0, result)

def encode_as_e4m3_stochastic(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32

    s, e, m = decompose_16bit(t)
    
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
    
    # Quantize to e4m3
    e = (e - 127) % 0b1_0000_0000  # Subtract bfloat16 bias
    e = (e + 7) % 0b1_0000  # Add e4m3 bias
    
    # Calculate the distance to the next representable number
    distance = m & 0b1111  # Last 4 bits of mantissa
    
    # Calculate probability of rounding up
    prob_round_up = distance.float() / 16.0
    
    # Generate random values for stochastic rounding
    random_values = torch.rand_like(t, dtype=torch.float32)
    
    # Determine whether to round up or truncate
    round_up = random_values < prob_round_up
    
    # Perform rounding
    m_truncated = m >> 4
    m_rounded_up = m_truncated + 1
    
    m = torch.where(round_up, m_rounded_up, m_truncated)
    
    # Handle overflow
    overflow = m == 0b1000
    e = torch.where(overflow, e + 1, e)
    m = torch.where(overflow, 0, m)
    
    # Compose the result
    result = (s << 7) | (e << 3) | m
    
    # Handle special cases
    is_inf = (e == 15) & (m == 0)
    is_nan = (e == 15) & (m != 0)
    result = torch.where(is_inf, (s << 7) | 0b01111000, result)
    result = torch.where(is_nan, (s << 7) | 0b01111111, result)
    
    # Set zero values
    result = torch.where(is_zero, 0, result)
    
    return result.to(torch.uint8)

def bfloat16_to_fp8(t: torch.Tensor, mantissa_bits: int, rounding: str = 'trunc') -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    assert mantissa_bits in [2, 3]

    # Convert bfloat16 to int32 bits representation
    t_bits = bfloat16_to_bits(t)
    import pdb; pdb.set_trace()

    # Choose rounding method
    if mantissa_bits == 2:
        fp8_int = encode_as_e5m2_stochastic(t_bits, rounding)
    elif mantissa_bits == 3:
        if rounding == 'trunc':
            fp8_int = encode_as_e4m3_trunc(t_bits)
        elif rounding == 'roundup':
            fp8_int = encode_as_e4m3_round_up(t_bits)
        elif rounding == 'stochastic':
            fp8_int = encode_as_e4m3_stochastic(t_bits)
    else:
        raise ValueError("Invalid rounding method. Choose 'trunc', 'roundup', or 'stochastic'.")

    # Convert the result to uint8
    return fp8_int.to(torch.uint8)


def fp8_to_bfloat16(t: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    assert t.dtype == torch.uint8
    assert mantissa_bits in [2, 3]

    # Convert uint8 to int32 for decoding
    fp8_int = t.to(torch.int32)

    # Decode E5M2 format
    if mantissa_bits == 2:
        bfloat16_bits = decode_from_e5m2(fp8_int)
    elif mantissa_bits == 3:
        bfloat16_bits = decode_from_e4m3(fp8_int)
    else:
        raise ValueError("Invalid mantissa bits. Choose 2 or 3.")

    # Convert bits to bfloat16
    return bits_to_bfloat16(bfloat16_bits)

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
    
    if n_mantissa in [2, 3]:
        out = bfloat16_to_fp8(t, n_mantissa, rounding='stochastic')
    else:
        raise NotImplementedError("Unsupported mantissa bits")
    
    return out

@torch.jit.script
def undo_int8_fp8(
        fp8_tensor: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert fp8_tensor.dtype == torch.uint8

    if out is None:
        out = torch.zeros_like(fp8_tensor, dtype=torch.bfloat16)

    out = fp8_to_bfloat16(fp8_tensor, n_mantissa)

    return out
