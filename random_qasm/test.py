from random_qasm.qasm_generator import QasmGenerator
import pandas as pd

gen = QasmGenerator(pd.DataFrame(), 4, 10)
qasm = gen.arrange_cx_ratio(0.675, 0.0001)
# qasm = gen.construct_parallel(0.375)
gen.save_qasm(qasm, "out.qasm")
