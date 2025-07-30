from enum import Enum
from dataclasses import dataclass
import re


class Features(Enum):
    cx_layer_ratio = "cx_layer_ratio"
    qubit_degrees = "qubit_degrees"
    parallelism = "parallelism"
    cx_parallelism = "cx_parallelism"
    t_parallelism = "t_parallelism"
    mean_time_between_t = "mean_time_between_t"


class GateType(Enum):
    T = 0
    CX = 1


@dataclass
class Gate:
    type: GateType
    data: tuple[int, int]


type Layer = list[Gate]


@dataclass
class QASM:
    layers: list[Layer]
    qubits: set[int]


def parse_qasm(qasm_path: str) -> QASM:
    qubit_next_layer: dict[int, int] = {}
    layers: list[Layer] = []
    offset = 0
    register_groups: dict[str, int] = {}
    with open(qasm_path, "r") as qf:
        for line in qf:
            reg_match = re.match(r"(qreg)\s+([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\];", line)
            if reg_match:
                register_groups[reg_match.group(2)] = offset
                offset += int(reg_match.group(3))
                continue
            gate: Gate | None = None
            cx_match = re.match(
                rf"(cx)\s+({'|'.join(register_groups.keys())})\[(\d+)\],\s*({'|'.join(register_groups.keys())})\[(\d+)\];",
                line,
            )
            t_match = re.match(
                rf"(t|tdg)\s+({'|'.join(register_groups.keys())})\[(\d+)\];", line
            )
            if cx_match:
                gate = Gate(
                    type=GateType.CX,
                    data=(
                        int(cx_match.group(3)) + register_groups[cx_match.group(2)],
                        int(cx_match.group(5)) + register_groups[cx_match.group(4)],
                    ),
                )
            elif t_match:
                gate = Gate(
                    type=GateType.T,
                    data=(
                        (int(t_match.group(3)) + register_groups[t_match.group(2)], -1)
                    ),
                )
            if gate is None:
                continue
            gate_qubits = list(filter(lambda x: x != -1, gate.data))
            max_layer = 0
            for qubit in gate_qubits:
                if qubit not in qubit_next_layer:
                    qubit_next_layer[qubit] = 0
                max_layer = max(max_layer, qubit_next_layer[qubit])
            # Updating qubit layers must be done seperately for an accurate max_layer
            for qubit in gate_qubits:
                qubit_next_layer[qubit] = max_layer + 1
            # If the gate needs to start a new layer
            if len(layers) == max_layer:
                layers.append([])
            layers[max_layer].append(gate)
    return QASM(layers=layers, qubits=set(qubit_next_layer.keys()))
