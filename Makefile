.PHONY: install test train eval sim mujoco3d pushblock pushblock-viewer

install:
	pip install -e ".[dev]"

test:
	pytest -q

train:
	python scripts/train_synthetic.py --steps 800

eval:
	python -m imeanflow_robotics.evaluate --checkpoint checkpoints/imeanflow_synthetic.pt

sim:
	python scripts/sim_demo.py --train-steps 300

mujoco3d:
	python scripts/mujoco_3d_demo.py --train-steps 1200

pushblock:
	python scripts/mujoco_push_block_demo.py --save-data

pushblock-viewer:
	python scripts/mujoco_push_block_viewer.py
