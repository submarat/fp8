from typing import Optional

import torch

def bfloat16_to_fp8(t: torch.Tensor, n_mantissa: int):
    # Define constants based on n_mantissa
    if n_mantissa == 2:
        exponent_bits = 5
        max_exponent = 32
        bias = 15
    else:  # n_mantissa == 3
        exponent_bits = 4
        max_exponent = 16
        bias = 7

    # Extract sign, exponent, and mantissa
    sign = torch.sign(t)
    abs_t = torch.abs(t).to(torch.float32)  # Cast to float32 for higher precision in calculations
    exponent = torch.floor(torch.log2(abs_t)).to(torch.int32)
    mantissa = ((abs_t / (2.0 ** (exponent))) - 1.0).to(torch.bfloat16)
    # import pdb; pdb.set_trace()

    # Handle subnormal numbers
    subnormal_mask = (exponent + bias) <= 0
    exponent[subnormal_mask] = 0
    mantissa[subnormal_mask] = (abs_t[subnormal_mask] / (2.0 ** (-bias + 1))).to(torch.bfloat16)

    # Bias exponent
    exponent[~subnormal_mask] = (exponent[~subnormal_mask] + bias) % max_exponent
    
    # Stochastic rounding for mantissa
    mantissa_scaled = mantissa * (2 ** n_mantissa)
    mantissa_floor = torch.floor(mantissa_scaled)
    # mantissa_prob = mantissa_scaled - mantissa_floor
    # random_values = torch.rand_like(mantissa_prob)
    # mantissa_rounded = mantissa_floor + (random_values < mantissa_prob).to(torch.float32)
    mantissa_rounded = mantissa_floor
    
    # Combine components
    result = (sign < 0).to(torch.uint8) << 7
    result |= (exponent.to(torch.uint8) << n_mantissa)
    result |= mantissa_rounded.to(torch.uint8)
    
    # Handle special cases
    result[torch.isnan(t)] = 0b01111111 if n_mantissa == 2 else 0b01111111
    result[torch.isinf(t) & (t > 0)] = 0b01111100 if n_mantissa == 2 else 0b01111000
    result[torch.isinf(t) & (t < 0)] = 0b11111100 if n_mantissa == 2 else 0b11111000
    
    return result

def fp8_to_bfloat16(fp8_tensor: torch.Tensor, n_mantissa: int):
    # Define constants based on n_mantissa
    if n_mantissa == 2:
        exponent_bits = 5
        bias = 15
    else:  # n_mantissa == 3
        exponent_bits = 4
        bias = 7
    
    # Extract sign, exponent, and mantissa
    sign = ((fp8_tensor & 0b10000000) != 0).to(torch.float32) * -2 + 1
    exponent = (fp8_tensor >> n_mantissa) & ((1 << exponent_bits) - 1)
    mantissa = fp8_tensor & ((1 << n_mantissa) - 1)
    
    # Convert to float
    result = sign * (1 + mantissa.to(torch.float32) / (2 ** n_mantissa)) * (2.0 ** (exponent.to(torch.float32) - bias))
    
    # Handle subnormal numbers
    subnormal_mask = exponent == 0
    result[subnormal_mask] = sign[subnormal_mask] * (mantissa[subnormal_mask].to(torch.float32) / (2 ** n_mantissa)) * (2.0 ** (-bias + 1))
    
    # Handle special cases
    result[fp8_tensor == 0] = 0.0
    if n_mantissa == 2:
        result[(fp8_tensor & 0b01111100) == 0b01111100] = float('inf')
        result[(fp8_tensor & 0b11111100) == 0b11111100] = float('-inf')

        result[(fp8_tensor & 0b01111101) == 0b01111101] = float('nan')
        result[(fp8_tensor & 0b01111110) == 0b01111110] = float('nan')
        result[(fp8_tensor & 0b01111111) == 0b01111111] = float('nan')
    else:  # n_mantissa == 3
        result[(fp8_tensor & 0b01111111) == 0b11111000] = float('nan')
    return result

# @torch.jit.script
def round_to_fp8_represented_as_int8(
        t: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert n_mantissa == 2 or n_mantissa == 3
    assert t.dtype == torch.bfloat16

    if out is None:
        out = torch.empty_like(t, dtype=torch.uint8)

    # Define constants based on n_mantissa
    if n_mantissa == 2:
        exponent_bits = 5
        max_exponent = 15
        bias = 15
    else:  # n_mantissa == 3
        exponent_bits = 4
        max_exponent = 7
        bias = 7

    # Extract sign, exponent, and mantissa
    sign = torch.sign(t)
    abs_t = torch.abs(t).to(torch.float32)  # Cast to float32 for higher precision in calculations
    exponent = torch.floor(torch.log2(abs_t)).to(torch.int32) + bias
    mantissa = ((abs_t / (2.0 ** (exponent - bias))) - 1.0).to(torch.bfloat16)

    # Handle subnormal numbers
    subnormal_mask = exponent <= 0
    exponent[subnormal_mask] = 0
    mantissa[subnormal_mask] = (abs_t[subnormal_mask] / (2.0 ** (-bias + 1))).to(torch.bfloat16)

    # Clamp exponent
    exponent = torch.clamp(exponent, 0, max_exponent)
    
    # Stochastic rounding for mantissa
    mantissa_scaled = mantissa * (2 ** n_mantissa)
    mantissa_floor = torch.floor(mantissa_scaled)
    mantissa_prob = mantissa_scaled - mantissa_floor
    random_values = torch.rand_like(mantissa_prob)
    mantissa_rounded = mantissa_floor + (random_values < mantissa_prob).to(torch.float32)
    
    # Combine components
    result = (sign < 0).to(torch.uint8) << 7
    result |= (exponent.to(torch.uint8) << n_mantissa)
    result |= mantissa_rounded.to(torch.uint8)
    
    # Handle special cases
    result[torch.isnan(t)] = 0b01111111 if n_mantissa == 2 else 0b01111111
    result[torch.isinf(t) & (t > 0)] = 0b01111100 if n_mantissa == 2 else 0b01111000
    result[torch.isinf(t) & (t < 0)] = 0b11111100 if n_mantissa == 2 else 0b11111000
    
    out.copy_(result)
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
        out = torch.empty_like(fp8_tensor, dtype=torch.bfloat16)
    
    result = fp8_to_bfloat16(fp8_tensor, n_mantissa)
    out.copy_(result.to(torch.bfloat16))
    return out
