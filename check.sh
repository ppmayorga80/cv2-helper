#!/usr/bin/env bash

# stop if fails
set -e

echo "🔍 Running Pylint en cv2_helper..."
pylint --fail-under=10.0 cv2_helper

# pytest

echo "✅ All Ok. Ready for commit / push."