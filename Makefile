.PHONY: install test train eval

install:
	pip install -e ".[dev]"

test:
	pytest -q

train:
	python scripts/train_synthetic.py --steps 800

eval:
	python -m imeanflow_robotics.evaluate --checkpoint checkpoints/imeanflow_synthetic.pt

