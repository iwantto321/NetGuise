from netguise.checkpoints import load_checkpoint, save_checkpoint
from netguise.models import NetGuiseMorpher


def test_checkpoint_is_weights_only_safe(tmp_path):
    config = {
        "max_packets": 4,
        "d_model": 8,
        "num_heads": 2,
        "feedforward_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "mtu": 1500.0,
        "bandwidth_ratio": 0.30,
        "latency_ratio": 0.50,
    }
    model = NetGuiseMorpher(**config)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        "netguise",
        model,
        config,
        ["source", "target"],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    loaded, payload = load_checkpoint(path)
    assert isinstance(loaded, NetGuiseMorpher)
    assert payload["classes"] == ["source", "target"]
    assert all(isinstance(value, float) for value in payload["feature_mean"])
