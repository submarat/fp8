# %%
from src.solution import *

import pytest
import torch._prims as prims

e5m2 = {
    "0":                  0b0_00000_00,
    "smallest_subnormal": 0b0_00000_01,
    "largest_subnormal":  0b0_00000_11,
    "smallest_normal":    0b0_00001_00,
    "largest_normal":     0b0_11110_11,
    "inf":                0b0_11111_00,
    "nan1":               0b0_11111_01,
    "nan2":               0b0_11111_10,
    "nan3":               0b0_11111_11,
    "-0":                 0b1_00000_00,
    "-smallest_subnormal":0b1_00000_01,
    "-largest_subnormal": 0b1_00000_11,
    "-smallest_normal":   0b1_00001_00,
    "-largest_normal":    0b1_11110_11,
    "-inf":               0b1_11111_00,
    "nan4":               0b1_11111_01,
    "nan5":               0b1_11111_10,
    "nan6":               0b1_11111_11,
}

e4m3 = {
    "0":                  0b0_0000_000,
    "smallest_subnormal": 0b0_0000_001,
    "largest_subnormal":  0b0_0000_111,
    "smallest_normal":    0b0_0001_000,
    "largest_normal":     0b0_1110_111,
    "largest_normal_ext": 0b0_1111_110,
    "nan1":               0b0_1111_111,
    "-0":                 0b1_0000_000,
    "-smallest_subnormal":0b1_0000_001,
    "-largest_subnormal": 0b1_0000_111,
    "-smallest_normal":   0b1_0001_000,
    "-largest_normal":    0b1_1110_111,
    "-largest_normal_ext":0b1_1111_110,
    "nan2":               0b1_1111_111,
}

nan_values = {
    2: {
        e5m2['nan1'],
        e5m2['nan2'],
        e5m2['nan3'],
        e5m2['nan4'],
        e5m2['nan5'],
        e5m2['nan6'],
    },
    3: {
        e4m3['nan1'],
        e4m3['nan2'],
    }
}

def bfloat16_to_bits(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    x = prims.view_element_type(x, torch.uint16)
    return x.to(dtype=torch.int32)

def bits_to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int32
    x = x.to(dtype=torch.uint16)
    return prims.view_element_type(x, torch.bfloat16)

# Compositions functions for bfloat16, e4m3, e5m2
def compose_16bit(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 15) | (exponent << 7) | mantissa

def compose_e4m3(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 7) | (exponent << 3) | mantissa

def compose_e5m2(sign: torch.Tensor, exponent: torch.Tensor, mantissa: torch.Tensor) -> torch.Tensor:
    return (sign << 7) | (exponent << 2) | mantissa


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

@pytest.mark.parametrize("test_case, n_mantissa, start_value, end_value",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-largest_normal"], e5m2["-0"]),
    ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    ("e4m3 negative", 3, e4m3["-largest_normal_ext"], e4m3["-0"]),
])
def test_round(test_case: str, n_mantissa: int, start_value: int, end_value):
    prev = start_value
    for curr in range(start_value, end_value + 1):
        prev_tensor = torch.tensor(prev, dtype=torch.uint8)
        curr_tensor = torch.tensor(curr, dtype=torch.uint8)
        prev_ = fp8_to_bfloat16(prev_tensor, n_mantissa)
        curr_ = fp8_to_bfloat16(curr_tensor, n_mantissa)

        # mid point is higher precision than curr and prev
        mid_point = (curr_ + prev_)/2

        # test that we correctly round up and down
        chopped = bfloat16_to_fp8(mid_point, n_mantissa)

        assert chopped == prev_tensor, f"Failed for {test_case}: input {mid_point} expected {prev_tensor}, got {chopped}"

        prev = curr

@pytest.mark.parametrize("test_case, n_mantissa",
[
    ("e5m2", 2), 
    ("e4m3", 3), 
])
def test_lossless_quantization_cases(test_case, n_mantissa):

    input = torch.arange(0, 256, dtype=torch.uint8)

    quantized_bfloat16 = fp8_to_bfloat16(input, n_mantissa)
    quantized_fp8 = bfloat16_to_fp8(quantized_bfloat16, n_mantissa)

    isnan = quantized_bfloat16.isnan()
    isinf = quantized_bfloat16.isinf()
    mask = ~(isnan | isinf)

    torch.testing.assert_close(input[mask], quantized_fp8[mask])

@pytest.mark.parametrize("test_case, n_mantissa, num, offset, expected",
[
    ("e5m2 clamp if > largest positive", 2, e5m2["largest_normal"], 128, e5m2["largest_normal"]),
    ("e5m2 clamp if < smallest negative", 2, e5m2["-largest_normal"], -128, e5m2["-largest_normal"]),
    ("e4m3 clamp if > largest positive", 3, e4m3["largest_normal_ext"], 128, e4m3["largest_normal_ext"]),
    ("e4m3 clamp if < smallest negative", 3, e4m3["-largest_normal_ext"], -128, e4m3["-largest_normal_ext"]),
])
def test_clamp(test_case: str, n_mantissa, num, offset, expected):
    fp8_tensor = torch.tensor(num, dtype=torch.uint8)
    expected_fp8 = torch.tensor(expected, dtype=torch.uint8)

    original = fp8_to_bfloat16(fp8_tensor, n_mantissa)
    unclamped_bfloat16 = original + torch.tensor((offset,), dtype=torch.bfloat16)
    rounded_fp8 = round_to_fp8_represented_as_int8(unclamped_bfloat16, n_mantissa)

    assert rounded_fp8 == expected_fp8

@pytest.mark.parametrize("test_case, n_mantissa, input_value, expected_output", [
    ("e5m2 subnormal 1 (smallest positive)", 2, 1.5259e-05, 0b00000001),
    ("e5m2 subnormal 2", 2, 3.0518e-05, 0b00000010),
    ("e5m2 subnormal 3 (largest positive)", 2, 4.5776e-05, 0b00000011),
    ("e5m2 subnormal -1 (smallest negative)", 2, -1.5259e-05, 0b10000001),
    ("e5m2 subnormal -2", 2, -3.0518e-05, 0b10000010),
    ("e5m2 subnormal -3 (largest negative)", 2, -4.5776e-05, 0b10000011),
    ("e4m3 subnormal 1 (smallest positive)", 3, 1.9531e-03, 0b00000001),
    ("e4m3 subnormal 2", 3, 3.9062e-03, 0b00000010),
    ("e4m3 subnormal 3", 3, 5.8594e-03, 0b00000011),
    ("e4m3 subnormal 4", 3, 7.8125e-03, 0b00000100),
    ("e4m3 subnormal 5", 3, 9.7656e-03, 0b00000101),
    ("e4m3 subnormal 6", 3, 1.1719e-02, 0b00000110),
    ("e4m3 subnormal 7 (largest positive)", 3, 1.3672e-02, 0b00000111),
    ("e4m3 subnormal -1 (smallest negative)", 3, -1.9531e-03, 0b10000001),
    ("e4m3 subnormal -2", 3, -3.9062e-03, 0b10000010),
    ("e4m3 subnormal -3", 3, -5.8594e-03, 0b10000011),
    ("e4m3 subnormal -4", 3, -7.8125e-03, 0b10000100),
    ("e4m3 subnormal -5", 3, -9.7656e-03, 0b10000101),
    ("e4m3 subnormal -6", 3, -1.1719e-02, 0b10000110),
    ("e4m3 subnormal -7 (largest negative)", 3, -1.3672e-02, 0b10000111),
])
def test_bfloat16_to_fp8_subnormals(test_case, n_mantissa, input_value, expected_output):
    input_tensor = torch.tensor([input_value], dtype=torch.bfloat16)
    result = bfloat16_to_fp8(input_tensor, n_mantissa)
    expected = torch.tensor([expected_output], dtype=torch.uint8)
    
    assert torch.all(result == expected), f"Failed for {test_case}: expected {expected}, got {result}"

@pytest.mark.parametrize("test_case, n_mantissa, input_value, expected_output", [
    ("e5m2 positive infinity", 2, float('inf'), 0b01111100),
    ("e5m2 negative infinity", 2, float('-inf'), 0b11111100),
    ("e5m2 NaN", 2, float('nan'), 0b01111111),
    ("e4m3 positive infinity", 3, float('inf'), 0b01111000),
    ("e4m3 negative infinity", 3, float('-inf'), 0b11111000),
    ("e4m3 NaN", 3, float('nan'), 0b01111111),
])
def test_bfloat16_to_fp8_inf_nan(test_case, n_mantissa, input_value, expected_output):
    input_tensor = torch.tensor([input_value], dtype=torch.bfloat16)
    result = round_to_fp8_represented_as_int8(input_tensor, n_mantissa)
    expected = torch.tensor([expected_output], dtype=torch.uint8)
    
    if torch.isnan(input_tensor[0]):
        assert result[0] & 0b01111000 == 0b01111000, f"Failed for {test_case}: NaN not correctly represented"
    else:
        assert torch.all(result == expected), f"Failed for {test_case}: expected {expected}, got {result}"


@pytest.mark.parametrize("scale", [0.1, 1.0, 2.0, 10.0, 100.0])
@pytest.mark.parametrize("shift", [-2.0, -1.0, -1.5259e-05, 0.0, 1.0, 1.5259e-05, 2.0])
@pytest.mark.parametrize("n_mantissa", [2, 3])
def test_avg(n_mantissa, scale, shift):
    for i in range(1):
        input = torch.rand((1024, 1024), dtype=torch.bfloat16)
        input = input * scale + shift
        fp8 = round_to_fp8_represented_as_int8(input, n_mantissa, None)
        output = undo_int8_fp8(fp8, n_mantissa)

        assert torch.allclose(input.mean(), output.mean(), rtol=1e-02, atol=1e-02), f"input mean {input.mean()} != output mean {output.mean()}"

@pytest.mark.parametrize("n_mantissa", [2, 3])
def test_all_finite_bfloat16(n_mantissa):
    # Generate all possible 16-bit values
    all_bits = torch.arange(0, 2**16, dtype=torch.int16)
    
    # Convert bits to bfloat16
    all_bfloat16 = torch.empty(all_bits.shape, dtype=torch.bfloat16)
    all_bfloat16.view(torch.int16)[:] = all_bits
    
    # Filter out NaN and Inf values
    input = all_bfloat16[torch.isfinite(all_bfloat16)]
    
    import pdb; pdb.set_trace()
    fp8 = round_to_fp8_represented_as_int8(input, n_mantissa, None)
    output = undo_int8_fp8(fp8, n_mantissa)

    assert torch.allclose(input.mean(), output.mean(), rtol=1e-02, atol=1e-02), f"input mean {input.mean()} != output mean {output.mean()}"