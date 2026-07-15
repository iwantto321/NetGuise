import torch
from torch import nn

from netguise.models import NetGuiseMorpher
from netguise.online import CausalMorpherSession
from netguise.validation import validate_features


def make_model() -> NetGuiseMorpher:
    torch.manual_seed(7)
    model = NetGuiseMorpher(
        max_packets=4,
        d_model=8,
        num_heads=2,
        feedforward_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    return model.eval()


def sample_windows():
    source = torch.tensor(
        [[[1000.0, 1.0, 0.0], [1200.0, -1.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    target = torch.tensor(
        [[[1100.0, 1.0, 0.0], [1300.0, -1.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    return source, target


def test_sigmoid_projection_respects_constraints():
    model = make_model()
    source, _ = sample_windows()
    actions = model.project_full(source, torch.zeros((1, 4, 2)))
    morphed = model._compose(source, actions)

    assert not any(isinstance(module, nn.ReLU) for module in model.modules())
    assert torch.all(actions >= 0)
    assert actions[0, :, 0].sum() <= source[0, :, 0].sum() * 0.30 + 1e-5
    assert actions[0, :, 1].sum() <= source[0, :, 2].sum() * 0.50 + 1e-5
    assert torch.all(morphed[0, :2, 0] <= 1500.0)

    report = validate_features(source[0].numpy(), morphed[0].numpy())
    assert report.passed


def test_future_packets_cannot_change_committed_actions():
    model = make_model()
    source, target = sample_windows()
    variant = source.clone()
    variant[0, 2:] = torch.tensor([[1490.0, 1.0, 8.0], [1400.0, -1.0, 4.0]])

    first = model.forward_causal(source, target, decision_point=2).actions[:, :2]
    second = model.forward_causal(variant, target, decision_point=2).actions[:, :2]
    torch.testing.assert_close(first, second)


def test_stateful_session_commits_one_action_per_packet():
    model = make_model()
    source, target = sample_windows()
    session = CausalMorpherSession(model, target[0])
    first = session.observe(source[0, 0])
    second = session.observe(source[0, 1])

    assert first.index == 0
    assert second.index == 1
    assert len(session.actions) == 2
    assert first.padding_bytes >= 0
    assert second.delay_seconds >= 0
