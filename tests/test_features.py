import os
import json

TEST_QASM_PATH = "./tests/test_qasm.qasm"
CHECK_FEATURES_PATH = "./tests/test_features.json"


def test_check_test_files():
    # Path to test qasm
    assert os.path.exists(TEST_QASM_PATH), (
        "Must be running in the wrong directory, try running in the root?"
    )
    assert os.path.exists(CHECK_FEATURES_PATH)
