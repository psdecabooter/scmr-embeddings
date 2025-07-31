#!/bin/bash
uv run -m random_qasm.test
uv run main.py out.qasm -o out.csv