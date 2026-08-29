import torch

from cinematch.data.ncf_dataset import NCFDataset
from cinematch.models.ncf import NCF


def test_ncf_forward_and_shapes():
    num_users, num_items, batch_size = 100, 50, 16
    model = NCF(num_users=num_users, num_items=num_items)
    
    users = torch.randint(0, num_users, (batch_size,))
    items = torch.randint(0, num_items, (batch_size,))
    
    outputs = model(users, items)
    
    assert outputs.shape == (batch_size,), f"Shape mismatch error: {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Error: Output contains NaN"
    assert not torch.isinf(outputs).any(), "Error: Output contains Inf"


def test_ncf_output_stays_in_explicit_rating_range():
    model = NCF(num_users=20, num_items=30)
    model.eval()
    users = torch.tensor([0, 1, 2, 3])
    items = torch.tensor([4, 5, 6, 7])

    with torch.no_grad():
        predictions = model(users, items)

    assert torch.all(predictions >= 1.0)
    assert torch.all(predictions <= 5.0)


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


def test_ncf_dataset_returns_model_ready_dtypes():
    dataset = NCFDataset(
        users=torch.tensor([0, 1], dtype=torch.int32),
        items=torch.tensor([2, 3], dtype=torch.int32),
        ratings=torch.tensor([4, 5], dtype=torch.int32),
    )

    user, item, rating = dataset[0]
    assert len(dataset) == 2
    assert user.dtype == torch.long
    assert item.dtype == torch.long
    assert rating.dtype == torch.float32


def test_ncf_training_supports_a_single_row_batch():
    model = NCF(num_users=2, num_items=2)
    prediction = model(torch.tensor([0]), torch.tensor([1]))
    prediction.sum().backward()

    assert prediction.shape == (1,)
