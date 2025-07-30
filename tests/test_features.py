import os
import json
import numpy as np
from scmr_embed.embedding_types import parse_qasm, Features
from scmr_embed.feature_extractor import FeatureExtractor

TEST_QASM_PATH = "./tests/test_qasm/test_qasm.qasm"
CHECK_FEATURES_PATH = "./tests/test_qasm/test_features.json"


def test_check_files():
    # Path to test qasm
    assert os.path.exists(TEST_QASM_PATH), (
        "Must be running in the wrong directory, try running in the root?"
    )
    assert os.path.exists(CHECK_FEATURES_PATH)


def test_cx_layer_ratio():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    # Truncate floats to two decimal places, can't represent 1/3 in json easily
    assert check_features["cx_layer_ratio"] == [
        int(x * 100) / 100 for x in features.cx_layer_ratio()
    ]


def test_qubit_degrees():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    assert set(check_features["qubit_degrees"]) == set(features.qubit_degrees())


def test_parallelism():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    assert check_features["parallelism"] == list(features.parallelism())


def test_cx_parallelism():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    assert check_features["cx_parallelism"] == list(features.cx_parallelism())


def test_t_parallelism():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    assert check_features["t_parallelism"] == list(features.t_parallelism())


def test_mean_time_between_t():
    qasm = parse_qasm(TEST_QASM_PATH)
    with open(CHECK_FEATURES_PATH, "r") as f:
        check_features = json.loads(f.read())
    features = FeatureExtractor(qasm)
    # Truncate floats to two decimal places, can't represent 1/3 in json easily
    assert set(np.array(check_features["mean_time_between_t"])) == set(
        [int(x * 100) / 100 for x in features.mean_time_between_t()]
    )


def test_all_features_coherence():
    qasm = parse_qasm(TEST_QASM_PATH)
    features = FeatureExtractor(qasm)
    all_features = features.all_features()
    # Test coherence between all_features() method and individual feature methods
    assert list(all_features[Features.cx_layer_ratio]) == list(
        features.cx_layer_ratio()
    )
    assert set(list(all_features[Features.qubit_degrees])) == set(
        list(features.qubit_degrees())
    )
    assert list(all_features[Features.parallelism]) == list(features.parallelism())
    assert list(all_features[Features.cx_parallelism]) == list(
        features.cx_parallelism()
    )
    assert list(all_features[Features.t_parallelism]) == list(features.t_parallelism())
    assert set(list(all_features[Features.mean_time_between_t])) == set(
        list(features.mean_time_between_t())
    )
