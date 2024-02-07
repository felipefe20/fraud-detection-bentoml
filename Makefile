.PHONY: tests docs

install:
	@echo "Initializing Git..."

	@echo "Installing dependencies..."


activate:
	@echo "Activating virtual environment..."


tests:
	pytest

docs:
	@echo Save documentation to docs...
	pdoc src -o docs --force
	@echo View API documentation...
	pdoc src --http localhost:8080
