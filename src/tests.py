from src.solution import *

import pytest
import numpy as np
import torch as t

def test_basic():
    x = t.tensor([1.0,2.0,3.0], dtype=t.bfloat16)
    n_mantissa = 3
    x_int8 = round_to_fp8_represented_as_int8(x, n_mantissa)

    x_bfloat16 = undo_int8_fp8(x_int8, n_mantissa)
