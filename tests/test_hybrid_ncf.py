import torch

from cinematch.models.hybrid_ncf import HybridNCF


def test_hybrid_ncf_forward_shape():
    num_users, num_items, side_dim = 100, 50, 10
    model = HybridNCF(num_users=num_users, num_items=num_items, side_feature_dim=side_dim)

    batch_size = 16
    users = torch.randint(0, num_users, (batch_size,))
    items = torch.randint(0, num_items, (batch_size,))
    side_feats = torch.randn(batch_size, side_dim)

    outputs = model(users, items, side_feats)

    assert outputs.shape == (batch_size,), f"Expected shape ({batch_size},), got {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Output contains NaN"
    assert torch.all(outputs >= 1.0)
    assert torch.all(outputs <= 5.0)


def test_hybrid_ncf_rejects_wrong_side_feature_dimension():
    model = HybridNCF(num_users=5, num_items=5, side_feature_dim=10)

    try:
        model(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.randn(2, 9),
        )
    except ValueError as error:
        assert "expected 10 side features" in str(error)
    else:
        raise AssertionError("HybridNCF accepted an invalid side-feature shape")
