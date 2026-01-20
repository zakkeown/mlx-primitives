"""Numerical gradient checking utilities for validating backward passes.

Provides tools to verify that analytical gradients match numerical approximations,
which is critical for validating custom backward implementations.
"""

from typing import Callable, List, Optional, Tuple, Any

import mlx.core as mx


def numerical_gradient(
    fn: Callable[..., mx.array],
    inputs: List[mx.array],
    arg_idx: int,
    eps: float = 1e-5,
) -> mx.array:
    """Compute numerical gradient using central differences.

    Uses the formula: grad[i] ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
    where e_i is a unit vector in direction i.

    Args:
        fn: Function to differentiate. Should return a scalar.
        inputs: List of input tensors.
        arg_idx: Index of the argument to compute gradient for.
        eps: Finite difference step size.

    Returns:
        Numerical gradient with same shape as inputs[arg_idx].
    """
    x = inputs[arg_idx]
    flat_x = x.flatten()
    num_elements = flat_x.size
    grad_values = []

    for i in range(num_elements):
        # Create one-hot perturbation vector using MLX ops (avoids O(n) tolist)
        one_hot = mx.zeros((num_elements,), dtype=flat_x.dtype)
        one_hot = one_hot.at[i].add(1.0)

        # Create perturbed arrays
        x_plus_arr = (flat_x + eps * one_hot).reshape(x.shape)
        x_minus_arr = (flat_x - eps * one_hot).reshape(x.shape)

        # Compute function values
        inputs_plus = inputs[:arg_idx] + [x_plus_arr] + inputs[arg_idx + 1:]
        inputs_minus = inputs[:arg_idx] + [x_minus_arr] + inputs[arg_idx + 1:]

        f_plus = fn(*inputs_plus)
        f_minus = fn(*inputs_minus)

        # Central difference
        grad_i = (f_plus.item() - f_minus.item()) / (2 * eps)
        grad_values.append(grad_i)

    return mx.array(grad_values).reshape(x.shape)


def numerical_gradient_fast(
    fn: Callable[..., mx.array],
    inputs: List[mx.array],
    arg_idx: int,
    eps: float = 1e-5,
    sample_ratio: float = 0.1,
) -> Tuple[mx.array, mx.array]:
    """Compute numerical gradient for a random sample of elements.

    Much faster than full numerical gradient for large tensors.
    Returns both the sampled numerical gradient and the indices sampled.

    Args:
        fn: Function to differentiate. Should return a scalar.
        inputs: List of input tensors.
        arg_idx: Index of the argument to compute gradient for.
        eps: Finite difference step size.
        sample_ratio: Fraction of elements to sample (0 to 1).

    Returns:
        Tuple of (sampled numerical gradients, sample indices).
    """
    x = inputs[arg_idx]
    num_elements = x.size
    num_samples = max(1, int(num_elements * sample_ratio))

    # Random sample indices
    indices = mx.random.randint(0, num_elements, (num_samples,))
    flat_x = x.flatten()

    numerical_grads = []
    # Convert indices once for iteration (small array, acceptable)
    indices_list = [int(i) for i in indices.tolist()]

    for idx in indices_list:
        # Create one-hot perturbation vector using MLX ops (avoids O(n) tolist)
        one_hot = mx.zeros((num_elements,), dtype=flat_x.dtype)
        one_hot = one_hot.at[idx].add(1.0)

        # Create perturbed arrays
        x_plus_arr = (flat_x + eps * one_hot).reshape(x.shape)
        x_minus_arr = (flat_x - eps * one_hot).reshape(x.shape)

        # Compute function values
        inputs_plus = inputs[:arg_idx] + [x_plus_arr] + inputs[arg_idx + 1:]
        inputs_minus = inputs[:arg_idx] + [x_minus_arr] + inputs[arg_idx + 1:]

        f_plus = fn(*inputs_plus)
        f_minus = fn(*inputs_minus)

        # Central difference
        grad_i = (f_plus.item() - f_minus.item()) / (2 * eps)
        numerical_grads.append(grad_i)

    return mx.array(numerical_grads), indices


def check_gradient(
    fn: Callable[..., mx.array],
    inputs: List[mx.array],
    rtol: float = 1e-3,
    atol: float = 1e-4,
    eps: float = 1e-5,
    sample_ratio: float = 0.1,
) -> Tuple[bool, List[Tuple[int, float, float, float]]]:
    """Compare analytical gradients to numerical approximation.

    Uses MLX's automatic differentiation to compute analytical gradients
    and compares them against numerical approximations.

    Args:
        fn: Function to differentiate. Should return a scalar.
        inputs: List of input tensors.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        eps: Finite difference step size.
        sample_ratio: Fraction of elements to check per input.

    Returns:
        Tuple of (passed, errors) where errors is a list of
        (arg_idx, max_abs_diff, max_rel_diff, sample_size) tuples.
    """
    errors = []
    passed = True

    for arg_idx in range(len(inputs)):
        # Compute analytical gradient
        def loss_fn(*args):
            return mx.sum(fn(*args))

        # Get gradient with respect to arg_idx
        grad_fn = mx.grad(loss_fn, argnums=arg_idx)
        analytical_grad = grad_fn(*inputs)

        # Compute numerical gradient (sampled)
        numerical_grads, indices = numerical_gradient_fast(
            lambda *args: mx.sum(fn(*args)),
            inputs,
            arg_idx,
            eps=eps,
            sample_ratio=sample_ratio,
        )

        # Extract analytical gradient at sampled indices
        flat_analytical = analytical_grad.flatten()
        analytical_sampled = flat_analytical[indices]

        # Compare
        abs_diff = mx.abs(analytical_sampled - numerical_grads)
        max_abs_diff = float(mx.max(abs_diff).item())

        # Relative difference (avoid division by zero)
        denom = mx.maximum(mx.abs(analytical_sampled), mx.abs(numerical_grads)) + 1e-8
        rel_diff = abs_diff / denom
        max_rel_diff = float(mx.max(rel_diff).item())

        # Check if within tolerance
        close = mx.all(abs_diff <= atol + rtol * mx.abs(numerical_grads))
        if not close:
            passed = False

        errors.append((arg_idx, max_abs_diff, max_rel_diff, len(indices.tolist())))

    return passed, errors


def gradient_check_attention(
    attention_fn: Callable,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    sample_ratio: float = 0.05,
    **kwargs: Any,
) -> Tuple[bool, dict]:
    """Specialized gradient check for attention functions.

    Verifies that gradients with respect to Q, K, V are correct.

    Args:
        attention_fn: Attention function to check.
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        sample_ratio: Fraction of elements to sample.
        **kwargs: Additional arguments passed to attention_fn.

    Returns:
        Tuple of (passed, details) where details contains gradient info.
    """
    # Wrap attention function with kwargs
    def fn(q, k, v):
        return attention_fn(q, k, v, **kwargs)

    inputs = [q, k, v]
    passed, errors = check_gradient(
        fn, inputs, rtol=rtol, atol=atol, sample_ratio=sample_ratio
    )

    details = {
        "q_gradient": {"max_abs_diff": errors[0][1], "max_rel_diff": errors[0][2]},
        "k_gradient": {"max_abs_diff": errors[1][1], "max_rel_diff": errors[1][2]},
        "v_gradient": {"max_abs_diff": errors[2][1], "max_rel_diff": errors[2][2]},
        "sample_size": errors[0][3],
    }

    return passed, details


def check_vjp(
    fn: Callable[..., mx.array],
    inputs: List[mx.array],
    cotangent: mx.array,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Tuple[bool, List[float]]:
    """Check vector-Jacobian product against numerical approximation.

    Verifies that vjp(fn, v) matches v @ J where J is the Jacobian.
    More efficient than full gradient check for multi-output functions.

    Args:
        fn: Function to differentiate.
        inputs: List of input tensors.
        cotangent: Vector to multiply with Jacobian.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Tuple of (passed, errors) where errors is list of max diffs per input.
    """
    # Compute analytical VJP
    primals, vjp_fn = mx.vjp(fn, *inputs)
    analytical_vjps = vjp_fn(cotangent)

    errors = []
    passed = True

    for arg_idx, (input_tensor, analytical_vjp) in enumerate(zip(inputs, analytical_vjps)):
        # Numerical approximation: sum(cotangent * numerical_jacobian)
        # For each input element, compute how output changes
        def scalar_loss(x):
            modified_inputs = inputs[:arg_idx] + [x] + inputs[arg_idx + 1:]
            out = fn(*modified_inputs)
            # Dot product with cotangent
            return mx.sum(out * cotangent)

        numerical_vjp = numerical_gradient(
            scalar_loss, [input_tensor], 0, eps=1e-5
        )

        # Compare
        abs_diff = mx.abs(analytical_vjp - numerical_vjp)
        max_diff = float(mx.max(abs_diff).item())
        errors.append(max_diff)

        if max_diff > atol + rtol * float(mx.max(mx.abs(numerical_vjp)).item()):
            passed = False

    return passed, errors
