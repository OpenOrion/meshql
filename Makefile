# Hypersim Build Pipeline Makefile

# Build Python package only
build-package:
	@echo "📦 Building Python package..."
	python -m build

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info/
	@echo "✅ Clean completed"

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m unittest discover -s tests -v

# Install package in development mode
install:
	@echo "📦 Installing package in development mode..."
	pip install -e .
	pip uninstall -y meshql

# Quick build without push (for testing)
build:
	@echo "🏗️ Building locally (no push)..."
	python -m build
	@echo "✅ Local build completed"
