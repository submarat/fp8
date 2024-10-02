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
    for curr in range(start_value + 1, end_value + 1):
        prev_tensor = torch.tensor(prev, dtype=torch.uint8)
        curr_tensor = torch.tensor(curr, dtype=torch.uint8)
        prev_ = fp8_to_bfloat16(prev_tensor, n_mantissa)
        curr_ = fp8_to_bfloat16(curr_tensor, n_mantissa)

        # mid point is higher precision than curr and prev
        mid_point = (curr_ + prev_)/2

        # test that we correctly round up and down
        chopped = bfloat16_to_fp8(mid_point, n_mantissa)

        assert chopped == prev_tensor

        prev = curr

@pytest.mark.parametrize("test_case, n_mantissa, start_bin, max_bin",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-0"] + 1, e5m2["-largest_normal"]-1),
    # ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    # ("e4m3 negative", 3, e4m3["-0"] + 1, e4m3["-largest_normal_ext"]),
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

# @pytest.mark.parametrize("n_mantissa", [ 2, 3 ])
@pytest.mark.parametrize("n_mantissa", [ 2 ])
def test_avg(n_mantissa):
    for i in range(100):
        input = torch.rand((1024, 1024), dtype=torch.bfloat16)
        fp8 = round_to_fp8_represented_as_int8(input, n_mantissa, None)
        output = undo_int8_fp8(fp8, n_mantissa)

        assert torch.allclose(input.mean(), output.mean(), rtol=1e-02)