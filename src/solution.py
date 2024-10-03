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
def encode_as_e5m2_trunc(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
    # Quantize to e5m2
    # Subtract bfloat16 bias
    e = (e - 127) % 0b1_0000_0000
    # Add e5m2 bias
    e = (e + 15) % 0b1_00_000
    # chop mantissa
    m = m >> 5
    # Compose the result
    result = (s << 7) + (e << 2) + m
    # Set zero values
    return torch.where(is_zero, 0, result)

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

def encode_as_e5m2_round_up(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
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
    result = compose_e5m2(s, e, m)
    # Set zero values
    return torch.where(is_zero, 0, result)

def encode_as_e5m2_stochastic(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.int32
    s, e, m = decompose_16bit(t)
    
    # Handle zero values
    is_zero = (e == 0) & (m == 0)
    
    # Quantize to e5m2
    e = (e - 127) % 0b1_0000_0000  # Subtract bfloat16 bias
    e = (e + 15) % 0b1_00_000  # Add e5m2 bias
    
    # Calculate the distance to the next representable number
    distance = m & 0b11111  # Last 5 bits of mantissa
    
    # Calculate probability of rounding up
    prob_round_up = distance.float() / 32.0
    
    # Generate random values for stochastic rounding
    random_values = torch.rand_like(t, dtype=torch.float32)
    
    # Determine whether to round up or truncate
    round_up = random_values < prob_round_up
    
    # Perform rounding
    m_truncated = m >> 5
    m_rounded_up = m_truncated + 1
    
    m = torch.where(round_up, m_rounded_up, m_truncated)
    
    # Handle overflow
    overflow = m == 0b100
    e = torch.where(overflow, e + 1, e)
    m = torch.where(overflow, 0, m)
    
    # Compose the result
    result = compose_e5m2(s, e, m)
    
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

    # Choose rounding method
    if rounding == 'trunc':
        if mantissa_bits == 2:
            fp8_int = encode_as_e5m2_trunc(t_bits)
        elif mantissa_bits == 3:
            fp8_int = encode_as_e4m3_trunc(t_bits)
    elif rounding == 'roundup':
        if mantissa_bits == 2:
            fp8_int = encode_as_e5m2_round_up(t_bits)
        elif mantissa_bits == 3:
            fp8_int = encode_as_e4m3_round_up(t_bits)
    elif rounding == 'stochastic':
        if mantissa_bits == 2:
            fp8_int = encode_as_e5m2_stochastic(t_bits)
        elif mantissa_bits == 3:
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
