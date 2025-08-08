import math
from enum import Enum
from dataclasses import dataclass
import random
import pandas as pd
import numpy as np
from scmr_embed.embedding_types import QASM, Gate, GateType, Layer
from scmr_embed.feature_extractor import FeatureExtractor

type RatioLayers = list[tuple[int, int]]


class QasmGenerator:
    def __init__(
        self, features: pd.DataFrame, num_qubits: int, num_layers: int
    ) -> None:
        self.features = features
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def save_qasm(self, qasm: QASM, path: str):
        qasm_lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.num_qubits}];",
        ]
        t_gate_line = "t q[{}];"
        cx_gate_line = "cx q[{}], q[{}];"
        for i, layer in enumerate(qasm.layers):
            qasm_lines.append(f"// Layer {i}")
            for gate in layer:
                if gate.type is GateType.CX:
                    qasm_lines.append(cx_gate_line.format(gate.data[0], gate.data[1]))
                elif gate.type is GateType.T:
                    qasm_lines.append(t_gate_line.format(gate.data[0]))
        with open(path, "w") as f:
            f.write("\n".join(qasm_lines))

    def validate_qasm(self, qasm: QASM):
        qubit_layers = {q: 0 for q in qasm.qubits}
        qubits_used: set[int] = set()
        print("validating")
        for i, layer in enumerate(qasm.layers):
            print(i, [gate.data for gate in layer])
            layer_qubits: set[int] = set()
            valid_starts = set([q for q, layer in qubit_layers.items() if layer == i])
            for gate in layer:
                # Ensure a valid start is used
                assert gate.data[0] in valid_starts or gate.data[1] in valid_starts
                # Ensure the qubit hasn't been used already in the layer
                assert gate.data[0] not in layer_qubits
                layer_qubits.add(gate.data[0])
                qubit_layers[gate.data[0]] = i + 1
                if gate.type is GateType.CX:
                    # Ensure the qubit hasn't been used already in the layer
                    assert gate.data[1] not in layer_qubits
                    layer_qubits.add(gate.data[1])
                    qubit_layers[gate.data[1]] = i + 1
                elif gate.type is GateType.T:
                    # Ensure the second gate data is -1 for t gates
                    assert gate.data[1] == -1
            qubits_used.update(layer_qubits)

        # Ensure only/all of the qasm qubits are used
        assert qubits_used == qasm.qubits

    def sample_features(self, feature: str, num_samples: int):
        """
        Gausian Mixture Model Sampling
        """
        means = np.array(self.features[feature + ".mean"])
        stds = np.array(self.features[feature + ".std"])
        assert len(means) == len(stds)
        # Randomly choose components with equal probability
        components = np.random.choice(len(means), size=num_samples)
        # Generate samples
        return np.random.normal(loc=means[components], scale=stds[components])

    def construct_perfect_parallel_qasm(self, num_layers: int) -> QASM:
        qubits = set(range(self.num_qubits))
        layers: list[Layer] = []
        for i in range(num_layers):
            layer: list[Gate] = []
            for q in range(self.num_qubits):
                layer.append(Gate(type=GateType.T, data=(q, -1)))
            layers.append(layer)
        return QASM(layers=layers, qubits=qubits)

    def construct_parallel(self, parallel: float) -> QASM:
        gate_layer_ratio = (self.num_qubits * parallel).as_integer_ratio()
        print(f"Gate:Layer = {gate_layer_ratio}")
        necessary_padding = math.ceil(5 / (gate_layer_ratio[0] - gate_layer_ratio[1]))
        gate_layer_ratio = (
            gate_layer_ratio[0] * necessary_padding,
            gate_layer_ratio[1] * necessary_padding,
        )
        print(f"Gate:Layer post padding = {gate_layer_ratio}")
        qasm = self.construct_perfect_parallel_qasm(gate_layer_ratio[1])

        # One qubit needs to be maintained
        persistent_qubit = random.randrange(self.num_qubits)

        layer_track = len(qasm.layers) - 1
        num_remove_gates = (gate_layer_ratio[1] * self.num_qubits) - gate_layer_ratio[0]
        for i in range(num_remove_gates):
            if len(qasm.layers[layer_track]) == 1:
                layer_track -= 1
            candidate_removals = [
                i
                for i, gate in enumerate(qasm.layers[layer_track])
                if persistent_qubit not in gate.data
            ]
            qasm.layers[layer_track].pop(random.choice(candidate_removals))

        return qasm

    def arrange_cx_ratio(self, cx_ratio: float, precision: float, limit: int = 1000):
        @dataclass
        class OP:
            class OP_TYPE(Enum):
                add_layer = 0
                add_cx = 1
                remove_t = 2

            t: OP_TYPE
            i: int

        cx_gate_ratios = [(0, self.num_qubits)]
        cur_ratio = 0

        loops = 0
        while abs(cur_ratio - cx_ratio) > precision and loops < limit:
            loops += 1
            best_ratio = cur_ratio * (len(cx_gate_ratios) / (len(cx_gate_ratios) + 1))
            best_diff = abs(best_ratio - cx_ratio)
            best_op = OP(t=OP.OP_TYPE.add_layer, i=0)
            for i, (cx, count) in enumerate(cx_gate_ratios):
                if cx < count - 1:
                    add_cx_ratio = cur_ratio + (
                        (cx + 1) / (count - 1) - cx / count
                    ) / len(cx_gate_ratios)

                    add_cx_diff = abs(add_cx_ratio - cx_ratio)
                    if add_cx_diff < best_diff:
                        best_diff = add_cx_diff
                        best_ratio = add_cx_ratio
                        best_op = OP(t=OP.OP_TYPE.add_cx, i=i)

                if cx == count - 1:
                    remove_t_ratio = cur_ratio + (cx / (count - 1) - cx / count) / len(
                        cx_gate_ratios
                    )

                    remove_t_diff = abs(remove_t_ratio - cx_ratio)
                    if remove_t_diff < best_diff:
                        best_diff = remove_t_diff
                        best_ratio = remove_t_ratio
                        best_op = OP(t=OP.OP_TYPE.remove_t, i=i)

            # print(cur_ratio, best_ratio, best_op)

            cur_ratio = best_ratio
            if best_op.t is OP.OP_TYPE.add_layer:
                cx_gate_ratios.append((0, self.num_qubits))
            elif best_op.t is OP.OP_TYPE.add_cx:
                layer = cx_gate_ratios[best_op.i]
                cx_gate_ratios[best_op.i] = (layer[0] + 1, layer[1] - 1)
            elif best_op.t is OP.OP_TYPE.remove_t:
                layer = cx_gate_ratios[best_op.i]
                cx_gate_ratios[best_op.i] = (layer[0], layer[1] - 1)

        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))
        missing = random.randrange(self.num_qubits)
        for cxs, gates in cx_gate_ratios:
            layer = []
            available_qubits = set(range(self.num_qubits))
            cx_remaining = cxs
            new_missing = missing
            # If a missing is required, remove it from the available qubits
            if cxs == gates and self.num_qubits % 2 == 1:
                new_missing = (
                    missing + random.randrange(self.num_qubits - 1)
                ) % self.num_qubits
                available_qubits.remove(new_missing)
            for _ in range(gates):
                # Place ts if all of the cxs have been placed
                if cx_remaining == 0:
                    layer.append(
                        Gate(type=GateType.T, data=(available_qubits.pop(), -1))
                    )
                    continue
                cx_remaining -= 1
                # Try to connect the previous missing if able
                if missing in available_qubits:
                    available_qubits.remove(missing)
                    layer.append(
                        Gate(type=GateType.CX, data=(missing, available_qubits.pop()))
                    )
                    continue
                # Connect cx gates
                layer.append(
                    Gate(
                        type=GateType.CX,
                        data=(available_qubits.pop(), available_qubits.pop()),
                    )
                )

            missing = new_missing
            qasm.layers.append(layer)
        return qasm

    def cx_ratio_and_parallel(
        self,
        cx_ratio: float,
        parallelism: float,
        precision: float,
        limit: int = 1000,
        min_gates: int = 0,
    ) -> RatioLayers:
        @dataclass
        class OP:
            class OP_TYPE(Enum):
                add_layer = 0
                merge_ts = 1
                add_t = 2

            t: OP_TYPE
            i: int

        gate_layer_ratio = parallelism * self.num_qubits
        budget = gate_layer_ratio - 1  # starts out with -1 since we route a gate first
        # (#cxs, #gates)
        ratio_layers = [(1, 1)]
        cur_ratio = 1

        loops = 0
        total_gates = 0
        while (
            math.floor(budget) != 0
            or (abs(cur_ratio - cx_ratio) > precision and loops < limit)
            or total_gates < min_gates
        ):
            loops += 1
            best_ratio = 1  # Placeholder ratio, doesn't effect anything
            best_diff = 1  # Worst possible diff
            best_op: None | OP = None

            for i, (cxs, gates) in enumerate(ratio_layers):
                # If a layer can be added
                if math.floor(budget) == 0:
                    layer_ratio = cur_ratio * (
                        len(ratio_layers) / (len(ratio_layers) + 1)
                    )
                    layer_diff = abs(layer_ratio - cx_ratio)

                    if layer_diff < best_diff:
                        best_ratio = layer_ratio
                        best_diff = layer_diff
                        best_op = OP(t=OP.OP_TYPE.add_layer, i=len(ratio_layers))

                # If two t gates can be merged
                if gates - cxs >= 2:
                    merge_ratio = cur_ratio + (
                        (cxs + 1) / (gates - 1) - cxs / gates
                    ) / len(ratio_layers)
                    merge_diff = abs(merge_ratio - cx_ratio)

                    if merge_diff < best_diff:
                        best_ratio = merge_ratio
                        best_diff = merge_diff
                        best_op = OP(t=OP.OP_TYPE.merge_ts, i=i)

                # If a t can be added
                if gates - cxs + cxs * 2 < self.num_qubits and math.floor(budget) > 0:
                    add_ratio = cur_ratio + (cxs / (gates + 1) - cxs / gates) / len(
                        ratio_layers
                    )
                    add_diff = abs(add_ratio - cx_ratio)

                    if add_diff < best_diff:
                        best_ratio = add_ratio
                        best_diff = add_diff
                        best_op = OP(t=OP.OP_TYPE.add_t, i=i)

            cur_ratio = best_ratio
            assert best_op is not None  # something messed up if its None
            if best_op.t is OP.OP_TYPE.add_layer:
                ratio_layers.append((0, 1))  # add a layer with 1 t gate
                budget += gate_layer_ratio - 1  # add to budget, subtract 1 for the t
                total_gates += 1
            elif best_op.t is OP.OP_TYPE.merge_ts:
                layer = ratio_layers[best_op.i]
                ratio_layers[best_op.i] = (layer[0] + 1, layer[1] - 1)
                budget += 1  # add to budget, because removing a gate
                total_gates -= 1
            elif best_op.t is OP.OP_TYPE.add_t:
                layer = ratio_layers[best_op.i]
                ratio_layers[best_op.i] = (layer[0], layer[1] + 1)
                budget -= 1  # remove from budget, because adding a gate
                total_gates += 1
        print("Summary")
        print(cur_ratio, budget)
        return ratio_layers

    def from_ordered_layers(self, ratio_layers: RatioLayers):
        ordered_layers = sorted(ratio_layers, key=lambda x: x[1], reverse=True)
        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))
        qubit_layers = {q: 0 for q in range(self.num_qubits)}
        # Layers are oredered by number of gates so they can be valid
        for i, (cxs, gates) in enumerate(ordered_layers):
            remaining_qubits = set(range(self.num_qubits))
            caught_up_qubits = [q for q, layer in qubit_layers.items() if layer == i]

            # Prioritize ts so caught_up_qubits don't overlap
            t_remaining = gates - cxs
            layer = []
            for _ in range(gates):
                # Place cxs if all of the ts have been placed
                if t_remaining == 0:
                    assert len(caught_up_qubits) > 0
                    source_qubit = caught_up_qubits.pop(
                        random.randrange(len(caught_up_qubits))
                    )
                    remaining_qubits.remove(source_qubit)
                    # Prioritize ensuring all qubits are used
                    unused_qubits = [
                        q
                        for q, layer in qubit_layers.items()
                        if layer == 0 and q != source_qubit
                    ]
                    target_qubit = (
                        random.choice(list(remaining_qubits))
                        if len(unused_qubits) == 0
                        else unused_qubits.pop(random.randrange(len(unused_qubits)))
                    )
                    remaining_qubits.remove(target_qubit)
                    if target_qubit in caught_up_qubits:
                        caught_up_qubits.remove(target_qubit)
                    layer.append(
                        Gate(
                            type=GateType.CX,
                            data=(source_qubit, target_qubit),
                        )
                    )
                    # Update layers
                    qubit_layers[source_qubit] = i + 1
                    qubit_layers[target_qubit] = i + 1
                    continue
                t_remaining -= 1
                assert len(caught_up_qubits) > 0
                qubit = caught_up_qubits.pop(random.randrange(len(caught_up_qubits)))
                remaining_qubits.remove(qubit)
                layer.append(Gate(type=GateType.T, data=(qubit, -1)))
                qubit_layers[qubit] = i + 1
            qasm.layers.append(layer)
        return qasm

    def from_degree_layers(
        self, ratio_layers: RatioLayers, target_degree: float
    ) -> QASM:
        """ """
        print("Target degree=", target_degree)
        qubit_interactions: dict[int, dict[int, int]] = {
            q: {} for q in range(self.num_qubits)
        }
        current_degree = 0

        ordered_layers = sorted(ratio_layers, key=lambda x: x[1], reverse=True)
        qubit_layers = {q: 0 for q in range(self.num_qubits)}
        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))

        for i, (cxs, gates) in enumerate(ordered_layers):
            new_qubit_layers = qubit_layers.copy()
            remaining_qubits = set(range(self.num_qubits))
            caught_up_qubits = [
                q for q, layer in new_qubit_layers.items() if layer == i
            ]
            lagging_qubits = remaining_qubits.difference(set(caught_up_qubits))
            print(i, caught_up_qubits, lagging_qubits)

            # Prioritize cxs to find valid connections
            cxs_remaining = cxs
            layer: list[Gate] = []
            for _ in range(gates):
                # Place cxs if not all have been placed
                if cxs_remaining != 0:
                    cxs_remaining -= 1
                    assert len(caught_up_qubits) > 0

                    source_qubit = caught_up_qubits.pop(
                        random.randrange(len(caught_up_qubits))
                    )
                    remaining_qubits.remove(source_qubit)
                    # Prioritize ensuring all qubits are used
                    unused_qubits = [
                        q
                        for q, layer in new_qubit_layers.items()
                        if layer == 0 and q != source_qubit
                    ]
                    # Choose target qubit in a way which minimizes connections if possible
                    unconnected_qubits = remaining_qubits.difference(
                        qubit_interactions[source_qubit]
                    )
                    connected_qubits = remaining_qubits.intersection(
                        qubit_interactions[source_qubit]
                    )

                    target_qubit = (
                        unused_qubits.pop(random.randrange(len(unused_qubits)))
                        if len(unused_qubits) != 0
                        else random.choice(list(connected_qubits))
                        if len(connected_qubits) != 0
                        else random.choice(list(unconnected_qubits))
                    )

                    # Remove target qubit from set
                    remaining_qubits.remove(target_qubit)
                    if target_qubit in caught_up_qubits:
                        caught_up_qubits.remove(target_qubit)
                    layer.append(
                        Gate(
                            type=GateType.CX,
                            data=(source_qubit, target_qubit),
                        )
                    )
                    if target_qubit not in qubit_interactions[source_qubit]:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                        qubit_interactions[source_qubit][target_qubit] = 0
                    else:
                        qubit_interactions[source_qubit][target_qubit] += 1
                    if source_qubit not in qubit_interactions[target_qubit]:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                        qubit_interactions[target_qubit][source_qubit] = 0
                    else:
                        qubit_interactions[target_qubit][source_qubit] += 1
                    # Update layers
                    new_qubit_layers[source_qubit] = i + 1
                    new_qubit_layers[target_qubit] = i + 1
                    continue
                # Place t gates
                assert len(caught_up_qubits) > 0
                qubit = caught_up_qubits.pop(random.randrange(len(caught_up_qubits)))
                remaining_qubits.remove(qubit)
                layer.append(Gate(type=GateType.T, data=(qubit, -1)))
                new_qubit_layers[qubit] = i + 1
            # Swap qubits in layers to get the best diff
            invalid_swap_diff = 100  # Used for invalid swaps
            swap_diffs = [
                [1.0 for _ in range(self.num_qubits)] for _ in range(self.num_qubits)
            ]
            qubit_pairs: dict[int, int] = {}
            qubit_gates: dict[int, Gate] = {}
            # Fill in current cx
            for gate in layer:
                # Swapping current gates keeps the diff
                if gate.type is GateType.CX:
                    swap_diffs[gate.data[0]][gate.data[1]] = abs(
                        current_degree - target_degree
                    )
                    swap_diffs[gate.data[1]][gate.data[0]] = abs(
                        current_degree - target_degree
                    )
                    qubit_pairs[gate.data[0]] = gate.data[1]
                    qubit_pairs[gate.data[1]] = gate.data[0]
                    qubit_gates[gate.data[0]] = gate
                    qubit_gates[gate.data[1]] = gate
                    for qubit in gate.data:
                        if (
                            qubit not in lagging_qubits
                            and qubit_pairs[qubit] in lagging_qubits
                        ):
                            for lqubit in lagging_qubits:
                                swap_diffs[lqubit][qubit] = invalid_swap_diff
                                swap_diffs[qubit][lqubit] = invalid_swap_diff

                # A qubit w/ a t gate cannot swap to a lagging qubit
                if gate.type is GateType.T:
                    for lqubit in lagging_qubits:
                        swap_diffs[gate.data[0]][lqubit] = invalid_swap_diff
                        swap_diffs[lqubit][gate.data[0]] = invalid_swap_diff
                    qubit_pairs[gate.data[0]] = gate.data[1]
                    qubit_gates[gate.data[0]] = gate

            for source in range(self.num_qubits):
                # Other qubit is the qubit the current one is connected to
                # Other == -1 if the current is in a t gate
                # Other == -2 if the current is not in the layer
                source_other = qubit_pairs[source] if source in qubit_pairs else -2
                for target in range(source, self.num_qubits):
                    if swap_diffs[source][target] != 1:
                        continue
                    if source not in qubit_gates and target not in qubit_gates:
                        swap_diffs[source][target] = invalid_swap_diff
                        continue
                    if source == target:
                        swap_diffs[source][target] = abs(current_degree - target_degree)
                        continue
                    target_other = qubit_pairs[target] if target in qubit_pairs else -2

                    # +1 for adding an interaction, -1 for removing an interaction
                    source_change = (
                        1
                        if target_other >= 0
                        and target_other not in qubit_interactions[source]
                        else 0
                    ) + (
                        -1
                        if source_other >= 0 and qubit_interactions[source_other] == 1
                        else 0
                    )
                    target_change = (
                        1
                        if source_other >= 0
                        and source_other not in qubit_interactions[target]
                        else 0
                    ) + (
                        -1
                        if target_other >= 0 and qubit_interactions[target_other] == 1
                        else 0
                    )
                    # print(target_change + source_change)

                    swap_degree: float = current_degree + (
                        (source_change + target_change)
                        / (self.num_qubits - 1)
                        / self.num_qubits
                    )
                    swap_diff = abs(swap_degree - target_degree)

                    swap_diffs[source][target] = swap_diff
            # swap smallest diff
            flat_diffs = [
                (i, j, diff)
                for i, sources in enumerate(swap_diffs)
                for j, diff in enumerate(sources)
            ]
            best_diff = min(flat_diffs, key=lambda x: x[2])[2]
            print(best_diff)
            best_diffs = list(filter(lambda x: x[2] == best_diff, flat_diffs))
            print(", ".join([f"({s},{t},{d})" for s, t, d in best_diffs]))
            (source, target, _) = random.choice(list(best_diffs))
            source_gate = qubit_gates[source] if source in qubit_gates else None
            target_gate = qubit_gates[target] if target in qubit_gates else None
            print(target, source, swap_diffs[target][source])
            print("before", source_gate, target_gate)
            if source_gate is None:
                new_qubit_layers[source] = new_qubit_layers[target]
                new_qubit_layers[target] = qubit_layers[target]
                pass
            elif source_gate.data[0] == source:
                source_gate.data = (target, source_gate.data[1])
            else:
                source_gate.data = (source_gate.data[0], target)

            if target_gate is None:
                new_qubit_layers[target] = new_qubit_layers[source]
                new_qubit_layers[source] = qubit_layers[source]
            elif target_gate.data[0] == target:
                target_gate.data = (source, target_gate.data[1])
            else:
                target_gate.data = (target_gate.data[0], source)
            print("after", source_gate, target_gate)

            print(current_degree)
            print(layer)
            print("\n".join([str(x) for x in swap_diffs]))
            # print(list(best_diffs))
            # input()
            print()

            qasm.layers.append(layer)
            qubit_layers = new_qubit_layers
        print("sparse degree=", current_degree)

        return qasm

    def loop_from_degree_layers(
        self, ratio_layers: RatioLayers, target_degree: float
    ) -> QASM:
        """ """
        print("Target degree=", target_degree)
        qubit_interactions: dict[int, dict[int, int]] = {
            q: {t: 0 for t in range(self.num_qubits)} for q in range(self.num_qubits)
        }
        current_degree = 0

        ordered_layers = sorted(ratio_layers, key=lambda x: x[1], reverse=True)
        qubit_layers = {q: 0 for q in range(self.num_qubits)}
        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))

        # Layers are oredered by number of gates so they can be valid
        for i, (cxs, gates) in enumerate(ordered_layers):
            new_qubit_layers = qubit_layers.copy()

            # Route gates
            remaining_qubits = set(range(self.num_qubits))
            caught_up_remaining = [q for q, layer in qubit_layers.items() if layer == i]
            cx_remaining = cxs
            layer: list[Gate] = []
            for _ in range(gates):
                # Place cxs if all of the ts have been placed
                if cx_remaining > 0:
                    cx_remaining -= 1
                    assert len(caught_up_remaining) > 0

                    source_qubit = caught_up_remaining.pop(
                        random.randrange(len(caught_up_remaining))
                    )
                    remaining_qubits.remove(source_qubit)

                    # Prioritize ensuring all qubits are used
                    unused_qubits = [
                        q
                        for q, layer in new_qubit_layers.items()
                        if layer == 0 and q != source_qubit
                    ]

                    # Choose target qubit in a way which minimizes connections if possible
                    unconnected_qubits = remaining_qubits.difference(
                        qubit_interactions[source_qubit]
                    )
                    connected_qubits = remaining_qubits.intersection(
                        qubit_interactions[source_qubit]
                    )

                    # Choose an unused first, then an already connected, then an unconnected
                    target_qubit = (
                        unused_qubits.pop(random.randrange(len(unused_qubits)))
                        if len(unused_qubits) != 0
                        else random.choice(list(connected_qubits))
                        if len(connected_qubits) != 0
                        else random.choice(list(unconnected_qubits))
                    )

                    # Remove target qubit from set
                    remaining_qubits.remove(target_qubit)
                    if target_qubit in caught_up_remaining:
                        caught_up_remaining.remove(target_qubit)
                    layer.append(
                        Gate(
                            type=GateType.CX,
                            data=(source_qubit, target_qubit),
                        )
                    )

                    # Update interactions
                    if qubit_interactions[source_qubit][target_qubit] == 0:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                    qubit_interactions[source_qubit][target_qubit] += 1
                    if qubit_interactions[target_qubit][source_qubit] == 0:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                    qubit_interactions[target_qubit][source_qubit] += 1

                    # Update layers
                    new_qubit_layers[source_qubit] = i + 1
                    new_qubit_layers[target_qubit] = i + 1
                    continue
                # Place t gates
                assert len(caught_up_remaining) > 0
                qubit = caught_up_remaining.pop(
                    random.randrange(len(caught_up_remaining))
                )
                remaining_qubits.remove(qubit)
                layer.append(Gate(type=GateType.T, data=(qubit, -1)))
                new_qubit_layers[qubit] = i + 1

            # Rearrange
            lagging_qubits = [q for q, layer in qubit_layers.items() if layer != i]
            invalid_swap = 100.0
            temp_swap = 10.0
            # Loop until no better changes can be made
            while True:
                swap_degrees = [
                    [temp_swap for _ in range(self.num_qubits)]
                    for _ in range(self.num_qubits)
                ]
                qubit_pairs = {}
                qubit_gates: dict[int, Gate] = {}
                # Fill in special layers
                # 1. non-swaps
                # 2. hanging cxs
                # 3. hanging ts
                for gate in layer:
                    if gate.type == GateType.CX:
                        # Save pairs
                        qubit_pairs[gate.data[0]] = gate.data[1]
                        qubit_pairs[gate.data[1]] = gate.data[0]
                        # Save swaps
                        swap_degrees[gate.data[0]][gate.data[1]] = current_degree
                        swap_degrees[gate.data[1]][gate.data[0]] = current_degree
                        # Save gates
                        qubit_gates[gate.data[0]] = gate
                        qubit_gates[gate.data[1]] = gate
                        # Haning cxs
                        for qubit in gate.data:
                            if (
                                qubit not in lagging_qubits
                                and qubit_pairs[qubit] in lagging_qubits
                            ):
                                for lqubit in lagging_qubits:
                                    swap_degrees[qubit][lqubit] = invalid_swap
                                    swap_degrees[lqubit][qubit] = invalid_swap

                    elif gate.type == GateType.T:
                        # Save gates
                        qubit_gates[gate.data[0]] = gate
                        # Hanging ts
                        for lqubit in lagging_qubits:
                            swap_degrees[gate.data[0]][lqubit] = invalid_swap
                            swap_degrees[lqubit][gate.data[0]] = invalid_swap

                # print("\n".join([str(x) for x in swap_diffs]))
                # print(qubit_pairs)
                # print(qubit_interactions)
                # Loop and track diffs
                for source in range(self.num_qubits):
                    for target in range(source, self.num_qubits):
                        # Already computed
                        if swap_degrees[source][target] != temp_swap:
                            continue
                        # Invalid swap
                        if target in lagging_qubits and source in lagging_qubits:
                            swap_degrees[source][target] = invalid_swap
                            continue
                        # 1-1 swap
                        if target == source:
                            swap_degrees[source][target] = current_degree
                            continue

                        # Compute swap cost
                        # +1 for adding, -1 for losing qubit interactions
                        source_change = (
                            1
                            if target in qubit_pairs
                            and qubit_pairs[target] not in qubit_interactions[source]
                            else 0
                        ) + (
                            -1
                            if source in qubit_pairs
                            and qubit_interactions[source][qubit_pairs[source]] == 1
                            else 0
                        )
                        target_change = (
                            1
                            if source in qubit_pairs
                            and qubit_pairs[source] not in qubit_interactions[target]
                            else 0
                        ) + (
                            -1
                            if target in qubit_pairs
                            and qubit_interactions[target][qubit_pairs[target]] == 1
                            else 0
                        )

                        degree_change = (
                            (source_change + target_change)
                            / (self.num_qubits + 1)
                            / (self.num_qubits)
                        )
                        # print(degree_change)
                        swap_degrees[source][target] = current_degree + degree_change

                # Flatten and find best
                flattened_diffs = [
                    (source, target, diff)
                    for source, row in enumerate(swap_degrees)
                    for target, diff in enumerate(row)
                ]
                best_degree = min(
                    flattened_diffs, key=lambda x: abs(x[2] - target_degree)
                )
                if best_degree[2] == current_degree:
                    break
                else:
                    current_degree = best_degree[2]
                best_swaps = [
                    swap for swap in flattened_diffs if swap[2] == best_degree[2]
                ]
                best_source, best_target = random.choice(best_swaps)[:2]

                # Change interactions
                if best_source in qubit_pairs:
                    qubit_interactions[best_source][qubit_pairs[best_source]] -= 1
                    qubit_interactions[best_target][qubit_pairs[best_source]] += 1
                if best_target in qubit_pairs:
                    qubit_interactions[best_source][qubit_pairs[best_target]] += 1
                    qubit_interactions[best_target][qubit_pairs[best_target]] -= 1
                # Swap them
                source_gate = (
                    qubit_gates[best_source] if best_source in qubit_gates else None
                )
                target_gate = (
                    qubit_gates[best_target] if best_target in qubit_gates else None
                )
                if source_gate is None:
                    new_qubit_layers[best_source] = i + 1
                    new_qubit_layers[best_target] = qubit_layers[best_target]
                elif source_gate.data[0] == best_source:
                    source_gate.data = (best_target, source_gate.data[1])
                else:
                    source_gate.data = (source_gate.data[0], best_target)

                if target_gate is None:
                    new_qubit_layers[best_target] = i + 1
                    new_qubit_layers[best_source] = qubit_layers[best_source]
                elif target_gate.data[0] == best_target:
                    target_gate.data = (best_source, target_gate.data[1])
                else:
                    target_gate.data = (target_gate.data[0], best_source)

                print("\n".join([str(x) for x in swap_degrees]))
                print(best_degree)
                # print(best_swaps)
                print(best_source, best_target)
                # input()
                # break

            # Shuffle layers
            qasm.layers.append(layer)
            qubit_layers = new_qubit_layers
        return qasm

    def degree_then_t_distance(
        self, ratio_layers: RatioLayers, target_degree: float
    ) -> QASM:
        """ """
        print("Target degree=", target_degree)
        qubit_interactions: dict[int, dict[int, int]] = {
            q: {t: 0 for t in range(self.num_qubits)} for q in range(self.num_qubits)
        }
        current_degree = 0

        ordered_layers = sorted(ratio_layers, key=lambda x: x[1], reverse=True)
        qubit_layers = {q: 0 for q in range(self.num_qubits)}
        qasm = QASM(layers=[], qubits=set(range(self.num_qubits)))

        # Layers are oredered by number of gates so they can be valid
        for i, (cxs, gates) in enumerate(ordered_layers):
            new_qubit_layers = qubit_layers.copy()

            # Route gates
            remaining_qubits = set(range(self.num_qubits))
            caught_up_remaining = [q for q, layer in qubit_layers.items() if layer == i]
            cx_remaining = cxs
            layer: list[Gate] = []
            for _ in range(gates):
                # Place cxs if all of the ts have been placed
                if cx_remaining > 0:
                    cx_remaining -= 1
                    assert len(caught_up_remaining) > 0

                    source_qubit = caught_up_remaining.pop(
                        random.randrange(len(caught_up_remaining))
                    )
                    remaining_qubits.remove(source_qubit)

                    # Prioritize ensuring all qubits are used
                    unused_qubits = [
                        q
                        for q, layer in new_qubit_layers.items()
                        if layer == 0 and q != source_qubit
                    ]

                    # Choose target qubit in a way which minimizes connections if possible
                    unconnected_qubits = remaining_qubits.difference(
                        qubit_interactions[source_qubit]
                    )
                    connected_qubits = remaining_qubits.intersection(
                        qubit_interactions[source_qubit]
                    )

                    # Choose an unused first, then an already connected, then an unconnected
                    target_qubit = (
                        unused_qubits.pop(random.randrange(len(unused_qubits)))
                        if len(unused_qubits) != 0
                        else random.choice(list(connected_qubits))
                        if len(connected_qubits) != 0
                        else random.choice(list(unconnected_qubits))
                    )

                    # Remove target qubit from set
                    remaining_qubits.remove(target_qubit)
                    if target_qubit in caught_up_remaining:
                        caught_up_remaining.remove(target_qubit)
                    layer.append(
                        Gate(
                            type=GateType.CX,
                            data=(source_qubit, target_qubit),
                        )
                    )

                    # Update interactions
                    if qubit_interactions[source_qubit][target_qubit] == 0:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                    qubit_interactions[source_qubit][target_qubit] += 1
                    if qubit_interactions[target_qubit][source_qubit] == 0:
                        current_degree += 1 / (self.num_qubits - 1) / self.num_qubits
                    qubit_interactions[target_qubit][source_qubit] += 1

                    # Update layers
                    new_qubit_layers[source_qubit] = i + 1
                    new_qubit_layers[target_qubit] = i + 1
                    continue
                # Place t gates
                assert len(caught_up_remaining) > 0
                qubit = caught_up_remaining.pop(
                    random.randrange(len(caught_up_remaining))
                )
                remaining_qubits.remove(qubit)
                layer.append(Gate(type=GateType.T, data=(qubit, -1)))
                new_qubit_layers[qubit] = i + 1

            # Rearrange
            lagging_qubits = [q for q, layer in qubit_layers.items() if layer != i]
            invalid_swap = 100.0
            temp_swap = 10.0
            # Loop until no better changes can be made
            while True:
                swap_degrees = [
                    [temp_swap for _ in range(self.num_qubits)]
                    for _ in range(self.num_qubits)
                ]
                qubit_pairs = {}
                qubit_gates: dict[int, Gate] = {}
                # Fill in special layers
                # 1. non-swaps
                # 2. hanging cxs
                # 3. hanging ts
                for gate in layer:
                    if gate.type == GateType.CX:
                        # Save pairs
                        qubit_pairs[gate.data[0]] = gate.data[1]
                        qubit_pairs[gate.data[1]] = gate.data[0]
                        # Save swaps
                        swap_degrees[gate.data[0]][gate.data[1]] = current_degree
                        swap_degrees[gate.data[1]][gate.data[0]] = current_degree
                        # Save gates
                        qubit_gates[gate.data[0]] = gate
                        qubit_gates[gate.data[1]] = gate
                        # Haning cxs
                        for qubit in gate.data:
                            if (
                                qubit not in lagging_qubits
                                and qubit_pairs[qubit] in lagging_qubits
                            ):
                                for lqubit in lagging_qubits:
                                    swap_degrees[qubit][lqubit] = invalid_swap
                                    swap_degrees[lqubit][qubit] = invalid_swap

                    elif gate.type == GateType.T:
                        # Save gates
                        qubit_gates[gate.data[0]] = gate
                        # Hanging ts
                        for lqubit in lagging_qubits:
                            swap_degrees[gate.data[0]][lqubit] = invalid_swap
                            swap_degrees[lqubit][gate.data[0]] = invalid_swap

                # print("\n".join([str(x) for x in swap_diffs]))
                # print(qubit_pairs)
                # print(qubit_interactions)
                # Loop and track diffs
                for source in range(self.num_qubits):
                    for target in range(source, self.num_qubits):
                        # Already computed
                        if swap_degrees[source][target] != temp_swap:
                            continue
                        # Invalid swap
                        if target in lagging_qubits and source in lagging_qubits:
                            swap_degrees[source][target] = invalid_swap
                            continue
                        # 1-1 swap
                        if target == source:
                            swap_degrees[source][target] = current_degree
                            continue

                        # Compute swap cost
                        # +1 for adding, -1 for losing qubit interactions
                        source_change = (
                            1
                            if target in qubit_pairs
                            and qubit_pairs[target] not in qubit_interactions[source]
                            else 0
                        ) + (
                            -1
                            if source in qubit_pairs
                            and qubit_interactions[source][qubit_pairs[source]] == 1
                            else 0
                        )
                        target_change = (
                            1
                            if source in qubit_pairs
                            and qubit_pairs[source] not in qubit_interactions[target]
                            else 0
                        ) + (
                            -1
                            if target in qubit_pairs
                            and qubit_interactions[target][qubit_pairs[target]] == 1
                            else 0
                        )

                        degree_change = (
                            (source_change + target_change)
                            / (self.num_qubits + 1)
                            / (self.num_qubits)
                        )
                        # print(degree_change)
                        swap_degrees[source][target] = current_degree + degree_change

                # Flatten and find best
                flattened_diffs = [
                    (source, target, diff)
                    for source, row in enumerate(swap_degrees)
                    for target, diff in enumerate(row)
                ]
                best_degree = min(
                    flattened_diffs, key=lambda x: abs(x[2] - target_degree)
                )
                if best_degree[2] == current_degree:
                    break
                else:
                    current_degree = best_degree[2]
                best_swaps = [
                    swap for swap in flattened_diffs if swap[2] == best_degree[2]
                ]
                best_source, best_target = random.choice(best_swaps)[:2]

                # Change interactions
                if best_source in qubit_pairs:
                    qubit_interactions[best_source][qubit_pairs[best_source]] -= 1
                    qubit_interactions[best_target][qubit_pairs[best_source]] += 1
                if best_target in qubit_pairs:
                    qubit_interactions[best_source][qubit_pairs[best_target]] += 1
                    qubit_interactions[best_target][qubit_pairs[best_target]] -= 1
                # Swap them
                source_gate = (
                    qubit_gates[best_source] if best_source in qubit_gates else None
                )
                target_gate = (
                    qubit_gates[best_target] if best_target in qubit_gates else None
                )
                if source_gate is None:
                    new_qubit_layers[best_source] = i + 1
                    new_qubit_layers[best_target] = qubit_layers[best_target]
                elif source_gate.data[0] == best_source:
                    source_gate.data = (best_target, source_gate.data[1])
                else:
                    source_gate.data = (source_gate.data[0], best_target)

                if target_gate is None:
                    new_qubit_layers[best_target] = i + 1
                    new_qubit_layers[best_source] = qubit_layers[best_source]
                elif target_gate.data[0] == best_target:
                    target_gate.data = (best_source, target_gate.data[1])
                else:
                    target_gate.data = (target_gate.data[0], best_source)

                print("\n".join([str(x) for x in swap_degrees]))
                print(best_degree)
                # print(best_swaps)
                print(best_source, best_target)
                # input()
                # break

            # Shuffle layers
            qasm.layers.append(layer)
            qubit_layers = new_qubit_layers
        return qasm
