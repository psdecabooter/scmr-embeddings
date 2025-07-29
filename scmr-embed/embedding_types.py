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
    qubit_layers: dict[int, int] = {}
    layers: list[Layer] = []
    with open(qasm_path, "r") as qf:
        for line in qf:
            gate: Gate | None = None
            cx_match = re.match(r"(cx)\s+q\[(\d+)\],\s*q\[(\d+)\];", line)
            t_match = re.match(r"(t|tdg)\s+q\[(\d+)\];", line)
            if cx_match:
                gate = Gate(
                    type=GateType.CX,
                    data=(int(cx_match.group(2)), int(cx_match.group(3))),
                )
            elif t_match:
                gate = Gate(type=GateType.T, data=((int(t_match.group(2)), -1)))
            if gate is None:
                continue
            gate_qubits = filter(lambda x: x != -1, gate.data)
            max_layer = 0
            for qubit in gate_qubits:
                if qubit not in qubit_layers:
                    qubit_layers[qubit] = 0
                max_layer = max(max_layer, qubit_layers[qubit])
                qubit_layers[qubit] += 1
            # If the gate needs to start a new layer
            if len(layers) == max_layer:
                layers.append([])
            layers[max_layer].append(gate)
    return QASM(layers=layers, qubits=set(qubit_layers.keys()))
