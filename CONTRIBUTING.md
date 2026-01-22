# Contributing to MLX Primitives

Thank you for your interest in contributing to MLX Primitives! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository and clone your fork
2. Set up the development environment (see below)
3. Create a branch, make your changes, and submit a pull request

## Development Setup

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+ (Python 3.13 recommended for MLX 0.30.x)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/zakkeown/mlx-primitives.git
cd mlx-primitives

# Create and activate a virtual environment
python3.13 -m venv .venv313
source .venv313/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import mlx; print(mlx.__version__)"  # Should show 0.30.x
python -c "import mlx_primitives; print('OK')"
```

## Code Style

We use automated tools to maintain consistent code style:

```bash
# Format code with Black
black .

# Check formatting without modifying
black --check .

# Lint with Ruff
ruff check .

# Lint with auto-fix
ruff check . --fix

# Type checking with mypy
mypy mlx_primitives --ignore-missing-imports
```

All code must pass these checks before merging. We recommend setting up pre-commit hooks:

```bash
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/attention/test_attention.py -v

# Run single test
pytest tests/attention/test_attention.py::test_flash_attention -v

# Skip slow and benchmark tests
pytest tests/ -m "not benchmark and not slow"

# Run with coverage
pytest tests/ --cov=mlx_primitives --cov-report=html
```

### Testing Guidelines

When adding or modifying functionality:

1. Add correctness tests comparing against reference implementations
2. Use standard tolerances:
   - FP16: `rtol=1e-3, atol=1e-4`
   - FP32: `rtol=1e-5, atol=1e-6`
3. For performance-critical code, add benchmark tests

## Contribution Types

### Bug Fixes

1. Check existing issues to see if the bug is already reported
2. Create an issue describing the bug if one doesn't exist
3. Reference the issue in your PR

### New Features

For significant features:

1. Open an issue to discuss the feature before implementing
2. Describe the use case and proposed API
3. Wait for feedback before starting implementation

For small features:

1. Create a PR with a clear description
2. Include tests and documentation

### Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples to docstrings
- Improve guides and tutorials

### Performance Improvements

1. Include benchmark results showing the improvement
2. Ensure no performance regressions (> 10% slowdown in other areas)
3. Document any new environment variables or configuration options

## Pull Request Process

### Branch Naming

Use descriptive branch names:
- `feature/add-sparse-attention`
- `fix/flash-attention-mask-bug`
- `docs/improve-getting-started`

### Commit Messages

Write clear, concise commit messages:

```
Add sparse attention implementation

- Implement block-sparse attention pattern
- Add tests for various sparsity patterns
- Update documentation with usage examples
```

### PR Description

Include in your PR description:
- What the change does
- Why the change is needed
- How to test the change
- Any breaking changes

### Review Checklist

Before requesting review, ensure:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black --check .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy mlx_primitives --ignore-missing-imports`)
- [ ] New code has appropriate tests
- [ ] Docstrings are added for public APIs
- [ ] CHANGELOG.md is updated (for user-facing changes)

## Architecture Notes

### Module Structure

```
mlx_primitives/
├── attention/     # Core attention mechanisms
├── kernels/       # Metal kernel wrappers and fused ops
├── primitives/    # Core parallel primitives (scan, MoE)
├── cache/         # KV cache implementations
├── generation/    # Batched generation engine
├── training/      # Training utilities
├── hardware/      # Hardware detection
├── memory/        # Memory primitives
├── dsl/           # Metal kernel DSL
├── ane/           # Apple Neural Engine dispatch
├── layers/        # NN layers
└── advanced/      # MoE, SSM, quantization
```

### Key Patterns

**MLX SDPA First**: Always prefer `mx.fast.scaled_dot_product_attention` when available. Custom Metal kernels should only be used when measurably faster.

**Tensor Layout**: Use `(batch, seq, heads, head_dim)` internally, transpose to `(batch, heads, seq, head_dim)` for MLX SDPA calls.

**Metal Kernels**: Custom shaders must:
- Stay under 32KB threadgroup memory
- Use bank conflict avoidance (`HEAD_DIM_PAD = head_dim + 4`)
- Include proper synchronization barriers

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues and documentation first
- Be specific and include reproduction steps for bugs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
