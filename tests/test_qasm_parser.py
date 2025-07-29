import os
import json
from scmr_embed.embedding_types import parse_qasm, GateType

TEST_QASM_PATH = "./tests/test_qasm.qasm"
CHECK_PARSED_PATH = "./tests/test_qasm_parsed.json"


def test_check_files():
    # Path to test qasm
    assert os.path.exists(TEST_QASM_PATH), (
        "Must be running in the wrong directory, try running in the root?"
    )
    assert os.path.exists(CHECK_PARSED_PATH)


def test_parse():
    with open(CHECK_PARSED_PATH, "r") as f:
        check_parsed = json.loads(f.read())
    check_layers = check_parsed["layers"]
    parsed_qasm = parse_qasm(TEST_QASM_PATH)
    num_gates = len([gate for layer in parsed_qasm.layers for gate in layer])
    num_layers = len(parsed_qasm.layers)
    # Check the number of layers and gates
    assert num_gates == len([gate for layer in check_layers for gate in layer])
    assert num_layers == len(check_layers)
    # Check each layer individually
    for i in range(len(check_layers)):
        for j in range(len(check_layers[i])):
            check_qubits = set(check_layers[i][j]["data"])
            parsed_qubits = set(
                filter(lambda q: q != -1, parsed_qasm.layers[i][j].data)
            )
            assert check_qubits == parsed_qubits
            assert (
                check_layers[i][j]["type"] == "cx"
                if parsed_qasm.layers[i][j].type is GateType.CX
                else check_layers[i][j]["type"] == "t"
            )
    # Check qubits
    assert parsed_qasm.qubits == set(check_parsed["qubits"])
