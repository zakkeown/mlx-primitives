"""Gradient computation utilities for backward pass parity tests."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx


# =============================================================================
# MLX Gradient Computation
# =============================================================================

def compute_mlx_gradients(
    fn: Callable,
    inputs: Dict[str, mx.array],
    argnums: Optional[List[int]] = None,
    wrt: Optional[List[str]] = None,
) -> Dict[str, mx.array]:
    """Compute gradients using MLX's autodiff.

    Args:
        fn: Function to differentiate. Should accept keyword arguments.
        inputs: Dictionary of MLX input arrays.
        argnums: Argument indices to differentiate with respect to.
        wrt: Input names to differentiate with respect to (alternative to argnums).

    Returns:
        Dictionary mapping input names to their gradients.
    """
    # Determine which inputs to differentiate
    if wrt is not None:
        input_names = list(inputs.keys())
        argnums = [input_names.index(name) for name in wrt if name in input_names]
    elif argnums is None:
        argnums = list(range(len(inputs)))

    # Convert dict inputs to positional for mx.grad
    input_names = list(inputs.keys())
    input_values = list(inputs.values())

    # Create a wrapper function that takes positional args
    def wrapper(*args):
        kwargs = dict(zip(input_names, args))
        result = fn(**kwargs)
        # Sum to scalar for gradient computation
        if isinstance(result, mx.array):
            return mx.sum(result)
        return result

    # Compute gradients
    grad_fn = mx.grad(wrapper, argnums=argnums)
    grads = grad_fn(*input_values)

    # Handle single gradient case
    if not isinstance(grads, tuple):
        grads = (grads,)

    # Map back to names
    grad_names = [input_names[i] for i in argnums]
    return dict(zip(grad_names, grads))


def compute_mlx_vjp(
    fn: Callable,
    inputs: Dict[str, mx.array],
    cotangent: mx.array,
    wrt: Optional[List[str]] = None,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """Compute vector-Jacobian product using MLX.

    Args:
        fn: Function to differentiate.
        inputs: Dictionary of MLX input arrays.
        cotangent: Upstream gradient (same shape as fn output).
        wrt: Input names to differentiate with respect to.

    Returns:
        Tuple of (forward output, gradients dictionary).
    """
    input_names = list(inputs.keys())
    input_values = list(inputs.values())

    if wrt is not None:
        argnums = [input_names.index(name) for name in wrt if name in input_names]
    else:
        argnums = list(range(len(inputs)))

    def wrapper(*args):
        kwargs = dict(zip(input_names, args))
        return fn(**kwargs)

    # Compute VJP
    primals, vjp_fn = mx.vjp(wrapper, input_values, argnums=argnums)
    grads = vjp_fn(cotangent)

    if not isinstance(grads, tuple):
        grads = (grads,)

    grad_names = [input_names[i] for i in argnums]
    return primals, dict(zip(grad_names, grads))


# =============================================================================
# PyTorch Gradient Computation
# =============================================================================

def compute_pytorch_gradients(
    fn: Callable,
    inputs: Dict[str, np.ndarray],
    wrt: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute gradients using PyTorch's autograd.

    Args:
        fn: Function to differentiate. Should accept numpy arrays and return numpy.
        inputs: Dictionary of numpy input arrays.
        wrt: Input names to differentiate with respect to.

    Returns:
        Dictionary mapping input names to their gradients as numpy arrays.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not available for gradient computation")

    # Convert inputs to torch tensors with requires_grad
    torch_inputs = {}
    for name, arr in inputs.items():
        t = torch.from_numpy(arr.astype(np.float32))
        if wrt is None or name in wrt:
            t = t.requires_grad_(True)
        torch_inputs[name] = t

    # Run forward pass
    def torch_fn(**kwargs):
        # Convert back to numpy for the function, then to torch
        np_inputs = {k: v.detach().numpy() for k, v in kwargs.items()}
        result = fn(**np_inputs)
        return torch.from_numpy(result.astype(np.float32))

    # Alternative: if fn is already torch-native
    output = torch_fn(**torch_inputs)

    # Compute gradients
    if wrt is None:
        wrt = list(inputs.keys())

    grad_tensors = [torch_inputs[name] for name in wrt if torch_inputs[name].requires_grad]
    grads = torch.autograd.grad(
        output.sum(),
        grad_tensors,
        create_graph=False,
    )

    # Map back to names
    grad_names = [name for name in wrt if torch_inputs[name].requires_grad]
    return {name: g.detach().numpy() for name, g in zip(grad_names, grads)}


def compute_pytorch_gradients_native(
    fn: Callable,
    inputs: Dict[str, "torch.Tensor"],
    wrt: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute gradients using PyTorch with native tensor inputs.

    Args:
        fn: PyTorch function.
        inputs: Dictionary of PyTorch tensors.
        wrt: Input names to differentiate with respect to.

    Returns:
        Dictionary of gradients as numpy arrays.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not available")

    if wrt is None:
        wrt = list(inputs.keys())

    # Ensure requires_grad
    for name in wrt:
        if name in inputs:
            inputs[name].requires_grad_(True)

    # Forward pass
    output = fn(**inputs)

    # Backward
    grad_tensors = [inputs[name] for name in wrt]
    grads = torch.autograd.grad(output.sum(), grad_tensors)

    return {name: g.detach().cpu().numpy() for name, g in zip(wrt, grads)}


# =============================================================================
# JAX Gradient Computation
# =============================================================================

def compute_jax_gradients(
    fn: Callable,
    inputs: Dict[str, np.ndarray],
    argnums: Optional[List[int]] = None,
    wrt: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute gradients using JAX's autodiff.

    Args:
        fn: Function to differentiate.
        inputs: Dictionary of numpy input arrays.
        argnums: Argument indices to differentiate with respect to.
        wrt: Input names to differentiate with respect to.

    Returns:
        Dictionary mapping input names to their gradients as numpy arrays.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX not available for gradient computation")

    input_names = list(inputs.keys())
    input_values = [jnp.array(v) for v in inputs.values()]

    if wrt is not None:
        argnums = tuple(input_names.index(name) for name in wrt if name in input_names)
    elif argnums is None:
        argnums = tuple(range(len(inputs)))

    def wrapper(*args):
        kwargs = dict(zip(input_names, args))
        result = fn(**kwargs)
        if isinstance(result, jnp.ndarray):
            return jnp.sum(result)
        return result

    grad_fn = jax.grad(wrapper, argnums=argnums)
    grads = grad_fn(*input_values)

    if not isinstance(grads, tuple):
        grads = (grads,)

    grad_names = [input_names[i] for i in argnums]
    return {name: np.array(g) for name, g in zip(grad_names, grads)}


# =============================================================================
# Numerical Gradient Computation
# =============================================================================

def numerical_gradient(
    fn: Callable,
    inputs: Dict[str, np.ndarray],
    wrt: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient using central finite differences.

    Args:
        fn: Function to differentiate.
        inputs: Input dictionary.
        wrt: Input name to compute gradient with respect to.
        eps: Epsilon for finite differences.

    Returns:
        Numerical gradient as numpy array.
    """
    x = inputs[wrt].copy().astype(np.float64)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + eps)
        x[idx] = old_value + eps
        inputs_plus = {**inputs, wrt: x.astype(inputs[wrt].dtype)}
        f_plus = fn(**inputs_plus)
        if hasattr(f_plus, "__array__"):
            f_plus = np.array(f_plus)

        # f(x - eps)
        x[idx] = old_value - eps
        inputs_minus = {**inputs, wrt: x.astype(inputs[wrt].dtype)}
        f_minus = fn(**inputs_minus)
        if hasattr(f_minus, "__array__"):
            f_minus = np.array(f_minus)

        # Central difference
        grad[idx] = np.sum(f_plus - f_minus) / (2 * eps)

        # Restore
        x[idx] = old_value
        it.iternext()

    return grad.astype(inputs[wrt].dtype)


def numerical_gradient_batch(
    fn: Callable,
    inputs: Dict[str, np.ndarray],
    wrt: List[str],
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """Compute numerical gradients for multiple inputs.

    Args:
        fn: Function to differentiate.
        inputs: Input dictionary.
        wrt: List of input names to compute gradients for.
        eps: Epsilon for finite differences.

    Returns:
        Dictionary mapping input names to numerical gradients.
    """
    return {name: numerical_gradient(fn, inputs, name, eps) for name in wrt}


# =============================================================================
# Gradient Comparison Utilities
# =============================================================================

def compare_gradients(
    mlx_grads: Dict[str, mx.array],
    ref_grads: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Compare MLX gradients to reference gradients.

    Args:
        mlx_grads: Dictionary of MLX gradient arrays.
        ref_grads: Dictionary of reference numpy gradient arrays.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if all gradients match within tolerance.
    """
    for name in ref_grads:
        if name not in mlx_grads:
            return False

        mlx_g = np.array(mlx_grads[name])
        ref_g = ref_grads[name]

        if not np.allclose(mlx_g, ref_g, rtol=rtol, atol=atol):
            return False

    return True


def assert_gradients_close(
    mlx_grads: Dict[str, mx.array],
    ref_grads: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    msg: str = "",
) -> None:
    """Assert that MLX gradients match reference gradients.

    Args:
        mlx_grads: Dictionary of MLX gradient arrays.
        ref_grads: Dictionary of reference numpy gradient arrays.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional error message prefix.
    """
    for name in ref_grads:
        assert name in mlx_grads, f"{msg} Missing gradient for {name}"

        mlx_g = np.array(mlx_grads[name])
        ref_g = ref_grads[name]

        np.testing.assert_allclose(
            mlx_g, ref_g,
            rtol=rtol, atol=atol,
            err_msg=f"{msg} Gradient mismatch for {name}"
        )


def gradient_check(
    fn: Callable,
    inputs: Dict[str, np.ndarray],
    wrt: List[str],
    eps: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Perform gradient check comparing analytical to numerical gradients.

    Args:
        fn: Function to check gradients for.
        inputs: Input dictionary.
        wrt: Input names to check gradients for.
        eps: Epsilon for numerical gradient.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if gradient check passes.
    """
    # Convert to MLX
    mlx_inputs = {k: mx.array(v) for k, v in inputs.items()}

    # Analytical gradients
    analytical = compute_mlx_gradients(fn, mlx_inputs, wrt=wrt)

    # Numerical gradients
    numerical = numerical_gradient_batch(fn, inputs, wrt, eps)

    # Compare
    return compare_gradients(analytical, numerical, rtol, atol)
