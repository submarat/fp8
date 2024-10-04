# %%
from src.solution import *

import pytest

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

@pytest.mark.parametrize("test_case, n_mantissa, start_value, end_value",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-largest_normal"], e5m2["-0"]),
    # ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    # ("e4m3 negative", 3, e4m3["-largest_normal_ext"], e4m3["-0"]),
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

def test_failing_input():
    input_value = 2.25
    n_mantissa = 2  # for e5m2
    input_tensor = torch.tensor([input_value], dtype=torch.bfloat16)
    result = bfloat16_to_fp8(input_tensor, n_mantissa)
    expected = torch.tensor([64], dtype=torch.uint8)
    assert torch.all(result == expected), f"Failed for e5m2 positive: input {input_value} expected {expected}, got {result}"

@pytest.mark.parametrize("test_case, n_mantissa, start_bin, max_bin",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-0"] + 1, e5m2["-largest_normal"]-1),
    ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    ("e4m3 negative", 3, e4m3["-0"] + 1, e4m3["-largest_normal_ext"]),
])
def test_lossless_quantization_cases(test_case, n_mantissa, start_bin, max_bin):

    def generate_input(start_bin, end_bin, shape: tuple = (1, 1)) -> torch.Tensor:
        range_values = torch.arange(start_bin, end_bin, dtype=torch.uint8)
        num_elements = torch.prod(torch.tensor(shape)).item()
        repeated_values = range_values.repeat((num_elements // range_values.size(0)) + 1)[:num_elements]
        result_tensor = repeated_values.view(shape)
        return result_tensor.to(dtype=torch.uint8)
    
    input = generate_input(start_bin, max_bin)

    quantized_bfloat16 = undo_int8_fp8(input, n_mantissa, None)
    quantized_fp8 = round_to_fp8_represented_as_int8(quantized_bfloat16, n_mantissa, None)

    torch.testing.assert_close(input, quantized_fp8.to(torch.uint8))

@pytest.mark.parametrize("test_case, n_mantissa, num, offset, expected",
[
    ("e5m2 clamp if > largest positive", 2, e5m2["largest_normal"], 1, e5m2["largest_normal"]),
    ("e5m2 clamp if < smallest negative", 2, e5m2["-largest_normal"], -1, e5m2["-largest_normal"]),
    ("e4m3 clamp if > largest positive", 3, e4m3["largest_normal"], 1, e4m3["largest_normal"]),
    ("e4m3 clamp if < smallest negative", 3, e4m3["-largest_normal"], -1, e4m3["-largest_normal"]),
])
def test_clamp(test_case: str, n_mantissa, num, offset, expected):
    fp8_tensor = torch.tensor(num, dtype=torch.uint8)
    expected_fp8 = torch.tensor(expected, dtype=torch.uint8)
    bfloat16_tensor = fp8_to_bfloat16(fp8_tensor, n_mantissa) + offset

    chopped_fp8 = bfloat16_to_fp8(bfloat16_tensor, n_mantissa)

    assert chopped_fp8 == expected_fp8

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
    result = bfloat16_to_fp8(input_tensor, n_mantissa)
    expected = torch.tensor([expected_output], dtype=torch.uint8)
    
    if torch.isnan(input_tensor[0]):
        assert result[0] & 0b01111000 == 0b01111000, f"Failed for {test_case}: NaN not correctly represented"
    else:
        assert torch.all(result == expected), f"Failed for {test_case}: expected {expected}, got {result}"

@pytest.mark.parametrize("n_mantissa", [2, 3])
def test_fp8_to_bfloat16_inf_nan(n_mantissa):
    # Test positive infinity
    pos_inf_fp8 = torch.tensor([0b01111000 if n_mantissa == 3 else 0b01111100], dtype=torch.uint8)
    pos_inf_result = fp8_to_bfloat16(pos_inf_fp8, n_mantissa)
    assert torch.isinf(pos_inf_result) and pos_inf_result > 0, f"Failed for positive infinity with {n_mantissa} mantissa bits"

    # Test negative infinity
    neg_inf_fp8 = torch.tensor([0b11111000 if n_mantissa == 3 else 0b11111100], dtype=torch.uint8)
    neg_inf_result = fp8_to_bfloat16(neg_inf_fp8, n_mantissa)
    assert torch.isinf(neg_inf_result) and neg_inf_result < 0, f"Failed for negative infinity with {n_mantissa} mantissa bits"

    # Test NaN
    nan_fp8 = torch.tensor([0b01111111], dtype=torch.uint8)
    nan_result = fp8_to_bfloat16(nan_fp8, n_mantissa)
    assert torch.isnan(nan_result), f"Failed for NaN with {n_mantissa} mantissa bits"


@pytest.mark.parametrize("n_mantissa, format_name", [
    (2, "e5m2"),
    (3, "e4m3")
])
def test_bfloat16_to_fp8_zero(n_mantissa, format_name):
    zero_tensor = torch.tensor([0.0], dtype=torch.bfloat16)
    
    result = bfloat16_to_fp8(zero_tensor, n_mantissa)
    expected = torch.tensor([0], dtype=torch.uint8)
    assert torch.all(result == expected), f"Failed for {format_name}: expected {expected}, got {result}"


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("shift", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("n_mantissa", [2, 3])
def test_avg(n_mantissa, scale, shift):
    for i in range(1):
        input = torch.rand((1024, 1024), dtype=torch.bfloat16)
        input = input * scale + shift
        fp8 = round_to_fp8_represented_as_int8(input, n_mantissa, None)
        output = undo_int8_fp8(fp8, n_mantissa)

        assert torch.allclose(input.mean(), output.mean(), rtol=1e-02, atol=1e-02), f"input mean {input.mean()} != output mean {output.mean()}"