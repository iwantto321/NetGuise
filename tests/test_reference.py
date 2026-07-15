import numpy as np

from netguise.reference import ReferencePool, morphability_score


def flow(directions, size_offset=0.0):
    value = np.zeros((4, 3), dtype=np.float32)
    value[:, 0] = np.asarray([100, 110, 120, 130]) + size_offset
    value[:, 1] = directions
    value[:, 2] = [0.0, 0.01, 0.02, 0.03]
    return value


def test_morphability_prefers_direction_compatible_reference():
    source = flow([1, -1, 1, -1])
    compatible = flow([1, -1, 1, -1], size_offset=10)
    incompatible = flow([-1, 1, -1, 1], size_offset=10)
    assert morphability_score(source, compatible) > morphability_score(source, incompatible)


def test_reference_selection_stays_within_target_class():
    source = flow([1, -1, 1, -1])
    features = np.stack(
        [source, flow([1, -1, 1, -1], 10), flow([-1, 1, -1, 1], 10)]
    )
    labels = np.asarray([0, 1, 1])
    _, index, _ = ReferencePool(features, labels).select(
        source, target_class=1, candidate_count=2, seed=4
    )
    assert labels[index] == 1


def test_causal_reference_selection_ignores_future_source_packets():
    source = flow([1, -1, 1, -1])
    variant = source.copy()
    variant[1:, 0] = 1400
    variant[1:, 1] *= -1
    features = np.stack(
        [source, flow([1, -1, 1, -1], 10), flow([-1, 1, -1, 1], 10)]
    )
    labels = np.asarray([0, 1, 1])
    pool = ReferencePool(features, labels)
    _, first_index, first_score = pool.select(
        source, 1, candidate_count=2, seed=9, observed_packets=1
    )
    _, second_index, second_score = pool.select(
        variant, 1, candidate_count=2, seed=9, observed_packets=1
    )
    assert first_index == second_index
    assert first_score == second_score
