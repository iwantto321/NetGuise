import torch

from netguise.models import FastFlowClassifier, UniNetClassifier


def traffic_batch():
    x = torch.zeros((3, 8, 3))
    x[:, :5, 0] = torch.tensor([100.0, 200.0, 300.0, 250.0, 180.0])
    x[:, :5, 1] = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0])
    x[:, :5, 2] = torch.tensor([0.0, 0.02, 0.04, 0.15, 0.01])
    return x


def test_uninet_packet_flow_adaptation_shape():
    model = UniNetClassifier(
        num_classes=4,
        max_packets=8,
        d_model=16,
        num_heads=2,
        feedforward_dim=32,
        num_layers=1,
        dropout=0.0,
    ).eval()
    assert model(traffic_batch()).shape == (3, 4)


def test_fastflow_dual_granularity_and_early_decision():
    model = FastFlowClassifier(
        num_classes=4, max_packets=8, hidden_size=16, dropout=0.0
    ).eval()
    x = traffic_batch()
    assert model(x).shape == (3, 4)
    decision = model.predict_early(
        x, decision_points=(3, 5, 8), confidence_threshold=0.0
    )
    assert torch.all(decision.packets_used == 3)
    assert decision.label.shape == (3,)
