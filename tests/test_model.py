import pytest
import torch

from src.models.model import NEOModel


def test_model_forward_shape():
    input_len = 4
    model = NEOModel(input_len, 8)
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_len) 
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)

