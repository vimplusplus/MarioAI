.PHONY: install train play checkpoints info test tensorboard clean help

# ── Default ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Mario AI — available make targets"
	@echo "  ──────────────────────────────────────────────────────────────────"
	@echo "  install        Install all Python dependencies"
	@echo "  train          Train the agent from scratch (1M timesteps)"
	@echo "  resume         Resume training from the latest checkpoint"
	@echo "  play           Watch the latest trained agent play"
	@echo "  checkpoints    List all saved model checkpoints"
	@echo "  info           Show config and environment info"
	@echo "  tensorboard    Launch TensorBoard on ./logs/"
	@echo "  test           Run the test suite"
	@echo "  clean          Remove all training artefacts and caches"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Training ───────────────────────────────────────────────────────────────────
train:
	python main.py train

resume:
	python main.py train --resume $$(python -c \
		"from mario_ai.utils import find_latest_checkpoint; \
		 p = find_latest_checkpoint('./train/'); \
		 print(p or 'NO_CHECKPOINT_FOUND')")

# ── Evaluation ────────────────────────────────────────────────────────────────
play:
	python main.py play

# ── Utilities ─────────────────────────────────────────────────────────────────
checkpoints:
	python main.py checkpoints

info:
	python main.py info

tensorboard:
	tensorboard --logdir ./logs/

# ── Testing ────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	@echo "Removing training artefacts, logs, and caches..."
	rm -rf train/ logs/ __pycache__/ mario_ai/__pycache__/ tests/__pycache__/
	rm -rf .pytest_cache/ *.egg-info/
	@echo "Done."
