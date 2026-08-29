import pytest
import torch
from cinematch.models.ncf import NCF
from cinematch.data.ncf_dataset import NCFDataset

def test_ncf_forward_and_shapes():
    num_users, num_items, batch_size = 100, 50, 16
    model = NCF(num_users=num_users, num_items=num_items)
    
    users = torch.randint(0, num_users, (batch_size,))
    items = torch.randint(0, num_items, (batch_size,))
    
    outputs = model(users, items)
    
    assert outputs.shape == (batch_size,), f"Shape mismatch error: {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Error: Output contains NaN"
    assert not torch.isinf(outputs).any(), "Error: Output contains Inf"

def test_ncf_dataset_and_backward():
    model = NCF(num_users=20, num_items=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    users = torch.tensor([0, 1])
    items = torch.tensor([1, 2])
    ratings = torch.tensor([4.0, 5.0])
    
    optimizer.zero_grad()
    preds = model(users, items)
    loss = criterion(preds, ratings)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing at layer: {name}"