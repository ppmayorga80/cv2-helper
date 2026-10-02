# default parameters
MODE ?= http
TARGET ?= ./cv2_helper

.PHONY: docs lint test all

# Documentation
docs:
	@if [ "$(MODE)" = "http" ]; then \
		echo "Starting pdoc development server..."; \
		pdoc --docformat google $(TARGET); \
	else \
		echo "Generating HTML documentation in ./html..."; \
		pdoc --docformat google $(TARGET) -o html; \
	fi

# Linter
lint:
	pylint --fail-under=10.0 cv2_helper


# all commands in a row
all:
	$(MAKE) docs
	$(MAKE) lint
	$(MAKE) docs MODE=html