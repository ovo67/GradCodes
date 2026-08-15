"""
Gradcodes building blocks for stage-wise discrete search on quantized linear layers.

This module implements the paper's core ideas in a form that can be attached to
quantized linear layers across the model, with search enabled only on selected
targets when desired:

1. Group-wise fixed-scale quantization of a pretrained weight matrix.
2. Stage-wise introduction of an active low-rank integer block.
3. Gradient-guided inverse-distance sampling on the discrete lattice.
4. Merge-only stage commits through accumulated residual codes.

The implementation focuses on a practical training framework rather than an
exact reproduction of every experimental detail in the paper.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


# NF4 codebook values used by bitsandbytes and QLoRA. Each block shares a
# single absmax scale, while codes are the 4-bit indices into this codebook.
NF4_CODEBOOK_VALUES = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

# OCP MXFP4 uses FP4 E2M1 private elements. The values below are sorted by
# numeric value for Gradcodes's integer lattice; the OCP binary interchange order
# is sign/exponent/mantissa bit order and includes the same two zero encodings.
MXFP4_CODEBOOK_VALUES = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
MXFP4_MAX_FINITE = 6.0
MXFP4_MAX_POWER_OF_TWO = 4.0
E8M0_MIN_EXPONENT = -127.0
E8M0_MAX_EXPONENT = 127.0


def is_integer_quant_type(quant_type: str) -> bool:
    return quant_type in {"uniform", "int4"}


def qrange_from_bits(bits: int, quant_type: str = "nf4") -> Tuple[int, int]:
    """Return the discrete code range for the requested quantization type."""
    if bits < 2:
        raise ValueError("Gradcodes requires bits >= 2 for a signed lattice.")

    if quant_type == "nf4":
        if bits != 4:
            raise ValueError("NF4 uses a fixed 4-bit codebook, so --quant_bits must be 4.")
        qmin = 0
        qmax = (2**bits) - 1
    elif quant_type == "mxfp4":
        if bits != 4:
            raise ValueError("MXFP4 uses a fixed FP4 E2M1 codebook, so --quant_bits must be 4.")
        qmin = 0
        qmax = (2**bits) - 1
    elif quant_type == "int4":
        if bits != 4:
            raise ValueError("INT4 uses a fixed 4-bit signed integer lattice, so --quant_bits must be 4.")
        qmin = -8
        qmax = 7
    elif quant_type == "uniform":
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    return qmin, qmax


def module_name_matches(full_name: str, target_modules: Sequence[str]) -> bool:
    """Match modules by exact name or suffix, mirroring common LoRA targeting."""
    if "all-linear" in target_modules:
        return True

    for target in target_modules:
        if full_name == target or full_name.endswith(f".{target}"):
            return True
    return False


def get_nf4_codebook(device: torch.device) -> torch.Tensor:
    """Return the normalized NF4 codebook on the requested device."""
    return torch.tensor(NF4_CODEBOOK_VALUES, device=device, dtype=torch.float32)


def get_mxfp4_codebook(device: torch.device) -> torch.Tensor:
    """Return the sorted FP4 E2M1 codebook on the requested device."""
    return torch.tensor(MXFP4_CODEBOOK_VALUES, device=device, dtype=torch.float32)


def decode_codes_tensor(
    codes: torch.Tensor,
    *,
    quant_type: str,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Decode lattice codes into normalized reconstruction values."""
    if is_integer_quant_type(quant_type):
        return codes.detach().to(torch.float32)
    if quant_type == "nf4":
        code_indices = torch.round(codes).clamp(qmin, qmax).to(torch.long)
        return get_nf4_codebook(codes.device)[code_indices]
    if quant_type == "mxfp4":
        code_indices = torch.round(codes).clamp(qmin, qmax).to(torch.long)
        return get_mxfp4_codebook(codes.device)[code_indices]
    raise ValueError(f"Unsupported quantization type: {quant_type}")


def interval_widths_from_codebook(
    codebook: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return the quantization-cell width associated with each codebook entry.

    Each width is computed from the Voronoi interval of the current lattice
    point: the half-way midpoints to the neighboring reconstruction values,
    clipped by the representable range [lower_bound, upper_bound].
    """
    codebook = codebook.detach().to(torch.float32)
    if codebook.ndim != 1 or codebook.numel() < 2:
        raise ValueError("Expected a 1D codebook with at least two entries.")

    midpoints = 0.5 * (codebook[:-1] + codebook[1:])
    lower = torch.empty_like(codebook)
    upper = torch.empty_like(codebook)
    lower[0] = lower_bound
    lower[1:] = midpoints
    upper[:-1] = midpoints
    upper[-1] = upper_bound
    return (upper - lower).clamp_min(eps)


def build_normalized_code_interval_widths(
    bits: int,
    quant_type: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build per-code interval widths in the normalized pre-scale domain."""
    qmin, qmax = qrange_from_bits(bits, quant_type=quant_type)
    if is_integer_quant_type(quant_type):
        return torch.ones((qmax - qmin + 1,), device=device, dtype=torch.float32)
    if quant_type == "nf4":
        return interval_widths_from_codebook(
            get_nf4_codebook(device),
            lower_bound=-1.0,
            upper_bound=1.0,
        )
    if quant_type == "mxfp4":
        return interval_widths_from_codebook(
            get_mxfp4_codebook(device),
            lower_bound=-MXFP4_MAX_FINITE,
            upper_bound=MXFP4_MAX_FINITE,
        )
    raise ValueError(f"Unsupported quantization type: {quant_type}")


def quantize_to_codebook_indices(
    normalized: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Nearest-neighbor quantization onto a fixed 1D codebook."""
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    return distances.argmin(dim=-1).to(torch.float32)


def e8m0_power_of_two_scale(absmax: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Approximate an E8M0 block scale as a clamped power-of-two tensor."""
    safe_absmax = torch.clamp(absmax, min=eps)
    exponent = torch.floor(torch.log2(safe_absmax)) - math.log2(MXFP4_MAX_POWER_OF_TWO)
    exponent = torch.clamp(exponent, min=E8M0_MIN_EXPONENT, max=E8M0_MAX_EXPONENT)
    return torch.pow(torch.full_like(exponent, 2.0), exponent)


def grouped_quantize(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    quant_type: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D weight matrix with fixed group-wise scales along the input axis.

    Returns:
        base_scales: Per-group floating-point scales with shape [rows, groups].
        base_codes: Packed discrete lattice codes with shape
            [rows, groups, group_size]. For NF4/MXFP4 these are uint8 codebook
            indices in [0, 15]; for uniform/int4 quantization they are int8
            lattice values.
    """
    if weight.ndim != 2:
        raise ValueError("Gradcodes currently supports 2D linear weights only.")

    qmin, qmax = qrange_from_bits(bits, quant_type=quant_type)
    rows, cols = weight.shape
    group_size = cols if group_size <= 0 else min(group_size, cols)
    num_groups = math.ceil(cols / group_size)
    padded_cols = num_groups * group_size
    weight_f32 = weight.detach().to(torch.float32)
    if padded_cols != cols:
        weight_f32 = F.pad(weight_f32, (0, padded_cols - cols))

    grouped = weight_f32.view(rows, num_groups, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(absmax, min=eps)

    if quant_type == "nf4":
        codebook = get_nf4_codebook(weight.device)
        normalized = torch.clamp(grouped / scales, min=-1.0, max=1.0)
        codes = quantize_to_codebook_indices(normalized, codebook).to(torch.uint8)
        scales = scales.squeeze(-1).to(torch.float32)
    elif quant_type == "mxfp4":
        codebook = get_mxfp4_codebook(weight.device)
        scales = e8m0_power_of_two_scale(scales, eps=eps)
        normalized = torch.clamp(
            grouped / scales,
            min=-MXFP4_MAX_FINITE,
            max=MXFP4_MAX_FINITE,
        )
        codes = quantize_to_codebook_indices(normalized, codebook).to(torch.uint8)
        scales = scales.squeeze(-1).to(torch.float32)
    elif is_integer_quant_type(quant_type):
        scales = (scales / max(qmax, 1)).squeeze(-1).to(torch.float32)
        codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(qmin, qmax).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    return scales.contiguous(), codes.contiguous()


def expand_grouped_scales(
    scales: torch.Tensor,
    *,
    original_cols: int,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand grouped scales into a dense [rows, cols] map on demand."""
    if scales.ndim != 2:
        raise ValueError("Expected grouped scales with shape [rows, groups].")

    rows, num_groups = scales.shape
    expanded = scales.to(dtype=dtype).unsqueeze(-1).expand(rows, num_groups, group_size)
    return expanded.reshape(rows, -1)[:, :original_cols].contiguous()


def dequantize_grouped_state(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    original_cols: int,
    group_size: int,
    quant_type: str,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize grouped codes/scales into a dense [rows, cols] weight tensor."""
    if codes.ndim != 3 or scales.ndim != 2:
        raise ValueError("Expected grouped codes/scales tensors.")

    if quant_type == "nf4":
        values = get_nf4_codebook(codes.device)[codes.to(torch.long)]
    elif quant_type == "mxfp4":
        values = get_mxfp4_codebook(codes.device)[codes.to(torch.long)]
    elif is_integer_quant_type(quant_type):
        values = codes.to(torch.float32)
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    dequantized = values.to(torch.float32) * scales.unsqueeze(-1).to(torch.float32)
    dense = dequantized.reshape(codes.shape[0], -1)[:, :original_cols].contiguous()
    if output_dtype != torch.float32:
        dense = dense.to(dtype=output_dtype)
    return dense


def approximate_top_singular_vectors(
    matrix: torch.Tensor,
    power_iterations: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate the dominant singular vectors with power iteration."""
    matrix = matrix.detach().to(torch.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D guidance matrix.")

    out_features, in_features = matrix.shape
    device = matrix.device

    if out_features == 0 or in_features == 0:
        raise ValueError("Cannot initialize rank-one factors from an empty matrix.")

    v = torch.randn(in_features, device=device, dtype=torch.float32)
    v = v / (v.norm() + 1e-12)

    u = torch.zeros(out_features, device=device, dtype=torch.float32)
    for _ in range(max(power_iterations, 1)):
        u = matrix @ v
        u_norm = u.norm()
        if u_norm <= 1e-12:
            break
        u = u / u_norm

        v = matrix.t() @ u
        v_norm = v.norm()
        if v_norm <= 1e-12:
            break
        v = v / v_norm

    if u.norm() <= 1e-12:
        u = torch.randn(out_features, device=device, dtype=torch.float32)
        u = u / (u.norm() + 1e-12)
    if v.norm() <= 1e-12:
        v = torch.randn(in_features, device=device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)

    return u, v


def quantize_guidance_vector(vector: torch.Tensor, max_abs_value: int) -> torch.Tensor:
    """Map a continuous guidance vector to a bounded integer lattice."""
    if max_abs_value < 1:
        raise ValueError("The integer search range must be at least 1.")

    vector = vector.detach().to(torch.float32)
    max_abs = vector.abs().max()
    if max_abs <= 1e-12:
        return torch.zeros_like(vector)

    scaled = (vector / max_abs) * float(max_abs_value)
    quantized = torch.round(scaled).clamp(-max_abs_value, max_abs_value)

    if torch.count_nonzero(quantized) == 0:
        signs = torch.sign(vector)
        quantized = torch.where(signs == 0, torch.ones_like(signs), signs)
        quantized = quantized.clamp(-max_abs_value, max_abs_value)

    return quantized


def quantize_guidance_tensor(tensor: torch.Tensor, max_abs_value: int) -> torch.Tensor:
    """Map a continuous guidance tensor to a bounded integer lattice elementwise."""
    if max_abs_value < 1:
        raise ValueError("The integer search range must be at least 1.")

    tensor = tensor.detach().to(torch.float32)
    max_abs = tensor.abs().max()
    if max_abs <= 1e-12:
        return torch.zeros_like(tensor)

    scaled = (tensor / max_abs) * float(max_abs_value)
    quantized = torch.round(scaled).clamp(-max_abs_value, max_abs_value)

    if torch.count_nonzero(quantized) == 0:
        quantized = quantized.clone()
        flat_tensor = tensor.reshape(-1)
        max_index = int(flat_tensor.abs().argmax().item())
        sign = float(torch.sign(flat_tensor[max_index]).item())
        if sign == 0.0:
            sign = 1.0
        quantized.view(-1)[max_index] = sign * float(max_abs_value)

    return quantized


def random_lattice_vector(
    size: int,
    max_abs_value: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Fallback non-zero random vector on the integer lattice."""
    values = torch.randint(0, 2, (size,), device=device, dtype=torch.int64)
    values = (values * 2) - 1
    scale = max(1, max_abs_value)
    return values.to(torch.float32) * float(scale)


def random_integer_lattice_tensor(
    shape: Tuple[int, ...],
    max_abs_value: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Uniform random integer tensor on [-max_abs_value, max_abs_value]."""
    if max_abs_value < 1:
        raise ValueError("The integer search range must be at least 1.")

    tensor = torch.randint(
        low=-max_abs_value,
        high=max_abs_value + 1,
        size=shape,
        device=device,
        dtype=torch.int64,
    ).to(torch.float32)

    if torch.count_nonzero(tensor) == 0:
        tensor = tensor.clone()
        tensor.view(-1)[0] = float(max_abs_value)
    return tensor


def inverse_distance_sample(
    current: torch.Tensor,
    guide: torch.Tensor,
    *,
    learning_rate: float,
    max_abs_value: int,
    epsilon: float,
    tau: float = 1.0,
    min_step_norm: Optional[float] = None,
    max_step_norm: Optional[float] = None,
    norm_p: float = 2.0,
) -> torch.Tensor:
    """
    Sample a new integer vector coordinate-wise from an inverse-distance lattice
    distribution centered at current + learning_rate * guide.
    """
    lattice, probs = inverse_distance_distribution(
        current,
        guide,
        learning_rate=learning_rate,
        max_abs_value=max_abs_value,
        epsilon=epsilon,
        tau=tau,
        min_step_norm=min_step_norm,
        max_step_norm=max_step_norm,
        norm_p=norm_p,
    )
    sampled, _, _ = sample_from_inverse_distance_distribution(lattice, probs)
    return sampled


def scaled_proposal_step(
    guide: torch.Tensor,
    *,
    learning_rate: float,
    min_step_norm: Optional[float] = None,
    max_step_norm: Optional[float] = None,
    norm_p: float = 2.0,
) -> torch.Tensor:
    """Return the proposal step after optional p-norm clipping."""
    if norm_p <= 0.0:
        raise ValueError("norm_p must be positive.")
    if min_step_norm is not None and min_step_norm < 0.0:
        raise ValueError("min_step_norm must be non-negative.")
    if max_step_norm is not None and max_step_norm < 0.0:
        raise ValueError("max_step_norm must be non-negative.")
    if (
        min_step_norm is not None
        and max_step_norm is not None
        and min_step_norm > 0.0
        and max_step_norm > 0.0
        and min_step_norm > max_step_norm
    ):
        raise ValueError("min_step_norm cannot exceed max_step_norm.")

    step = learning_rate * guide.detach().to(torch.float32)
    if (
        (min_step_norm is None or min_step_norm == 0.0)
        and (max_step_norm is None or max_step_norm == 0.0)
    ):
        return step

    ord_value: Union[float, int]
    if math.isinf(norm_p):
        ord_value = float("inf")
    elif float(norm_p).is_integer():
        ord_value = int(norm_p)
    else:
        ord_value = float(norm_p)

    step_norm = torch.linalg.vector_norm(step.reshape(-1), ord=ord_value)
    if (not torch.isfinite(step_norm)) or step_norm <= 0.0:
        return step

    scale = 1.0
    if min_step_norm is not None and min_step_norm > 0.0 and step_norm <= min_step_norm:
        scale = max(scale, float(min_step_norm / step_norm))
    if max_step_norm is not None and max_step_norm > 0.0 and step_norm > max_step_norm:
        scale = min(scale, float(max_step_norm / step_norm))
    return step * scale


def build_inverse_distance_lattice(
    current: torch.Tensor,
    guide: torch.Tensor,
    *,
    learning_rate: float,
    max_abs_value: int,
    epsilon: float,
    min_step_norm: Optional[float] = None,
    max_step_norm: Optional[float] = None,
    norm_p: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-coordinate local lattice and its log-distance table."""
    if max_abs_value < 1:
        raise ValueError("The integer search range must be at least 1.")

    current = current.detach().to(torch.float32)
    step = scaled_proposal_step(
        guide,
        learning_rate=learning_rate,
        min_step_norm=min_step_norm,
        max_step_norm=max_step_norm,
        norm_p=norm_p,
    )
    reference = current + step

    offsets = torch.arange(
        -max_abs_value,
        max_abs_value + 1,
        device=current.device,
        dtype=torch.float32,
    )

    # Build a per-coordinate local integer neighborhood around the guided
    # reference instead of using one shared global lattice such as [-1, 0, 1].
    # For max_abs_value = r, coordinate i considers
    # {round(reference_i) - r, ..., round(reference_i) + r}.
    center = torch.round(reference)
    lattice = center.unsqueeze(-1) + offsets.unsqueeze(0)
    log_distances = torch.log((reference.unsqueeze(-1) - lattice).abs() + epsilon)
    return lattice, log_distances


def inverse_distance_probabilities_from_log_distances(
    log_distances: torch.Tensor,
    *,
    tau: float = 1.0,
) -> torch.Tensor:
    """Normalize inverse-distance probabilities from a precomputed log-distance table."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    log_scores = (-tau) * log_distances.to(torch.float32)
    probs = torch.softmax(log_scores, dim=-1)

    invalid_rows = (
        (~torch.isfinite(probs)).any(dim=-1)
        | (probs < 0).any(dim=-1)
        | (probs.sum(dim=-1) <= 0)
    )
    if invalid_rows.any():
        fallback = F.one_hot(
            log_distances.argmin(dim=-1),
            num_classes=log_distances.shape[-1],
        ).to(torch.float32)
        probs = torch.where(invalid_rows.unsqueeze(-1), fallback, probs)
    return probs


def inverse_distance_distribution(
    current: torch.Tensor,
    guide: torch.Tensor,
    *,
    learning_rate: float,
    max_abs_value: int,
    epsilon: float,
    tau: float = 1.0,
    min_step_norm: Optional[float] = None,
    max_step_norm: Optional[float] = None,
    norm_p: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-coordinate local lattice and normalized probabilities."""
    lattice, log_distances = build_inverse_distance_lattice(
        current,
        guide,
        learning_rate=learning_rate,
        max_abs_value=max_abs_value,
        epsilon=epsilon,
        min_step_norm=min_step_norm,
        max_step_norm=max_step_norm,
        norm_p=norm_p,
    )
    probs = inverse_distance_probabilities_from_log_distances(
        log_distances,
        tau=tau,
    )
    return lattice, probs


def summarize_coordinate_probabilities(selected_probs: torch.Tensor) -> Dict[str, float]:
    """Summarize coordinate-wise probabilities for logging."""
    selected_probs = selected_probs.detach().to(torch.float32)
    safe_probs = selected_probs.clamp_min(1e-12)
    joint_log_prob = float(torch.log(safe_probs).sum().item())
    if torch.any(selected_probs <= 0):
        joint_log_prob = float("-inf")
    return {
        "joint_log_prob": joint_log_prob,
        "mean_coordinate_prob": float(selected_probs.mean().item()),
        "min_coordinate_prob": float(selected_probs.min().item()),
        "max_coordinate_prob": float(selected_probs.max().item()),
    }


def candidate_probability_under_distribution(
    candidate: torch.Tensor,
    lattice: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate the probability summary of a concrete candidate under a lattice distribution."""
    candidate = candidate.detach().to(torch.float32)
    candidate_shape = candidate.shape
    flat_candidate = candidate.reshape(-1)
    flat_lattice = lattice.reshape(-1, lattice.shape[-1])
    flat_probs = probs.reshape(-1, probs.shape[-1])

    matches = flat_lattice.eq(flat_candidate.unsqueeze(-1))
    has_match = matches.any(dim=-1)
    match_index = matches.to(torch.int64).argmax(dim=-1)
    gathered = flat_probs.gather(dim=1, index=match_index.unsqueeze(-1)).squeeze(-1)
    selected_probs = torch.where(has_match, gathered, torch.zeros_like(gathered)).reshape(candidate_shape)

    summary = summarize_coordinate_probabilities(selected_probs)
    summary["coverage"] = float(has_match.to(torch.float32).mean().item())
    return selected_probs, summary


def sample_from_inverse_distance_distribution(
    lattice: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Sample one candidate per coordinate and return probability summaries."""
    leading_shape = probs.shape[:-1]
    flat_probs = probs.reshape(-1, probs.shape[-1]).to(torch.float32)
    flat_lattice = lattice.reshape(-1, lattice.shape[-1])

    flat_probs = torch.nan_to_num(flat_probs, nan=0.0, posinf=0.0, neginf=0.0)
    flat_probs = flat_probs.clamp_min(0.0)
    row_sums = flat_probs.sum(dim=-1, keepdim=True)
    zero_rows = row_sums.squeeze(-1) <= 0
    if zero_rows.any():
        flat_probs[zero_rows] = 1.0 / flat_probs.shape[-1]
        row_sums = flat_probs.sum(dim=-1, keepdim=True)
    flat_probs = flat_probs / row_sums

    sampled_idx = torch.multinomial(flat_probs, num_samples=1)
    flat_sampled = flat_lattice.gather(dim=1, index=sampled_idx).squeeze(-1)
    flat_selected_probs = flat_probs.gather(dim=1, index=sampled_idx).squeeze(-1)
    sampled = flat_sampled.reshape(leading_shape)
    selected_probs = flat_selected_probs.reshape(leading_shape)
    return sampled, selected_probs, summarize_coordinate_probabilities(selected_probs)


def select_nearest_from_inverse_distance_distribution(
    lattice: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Select the highest-probability lattice point for each coordinate."""
    leading_shape = probs.shape[:-1]
    flat_probs = probs.reshape(-1, probs.shape[-1]).to(torch.float32)
    flat_lattice = lattice.reshape(-1, lattice.shape[-1])

    flat_probs = torch.nan_to_num(flat_probs, nan=0.0, posinf=0.0, neginf=0.0)
    flat_probs = flat_probs.clamp_min(0.0)
    row_sums = flat_probs.sum(dim=-1, keepdim=True)
    zero_rows = row_sums.squeeze(-1) <= 0
    if zero_rows.any():
        flat_probs[zero_rows] = 1.0 / flat_probs.shape[-1]
        row_sums = flat_probs.sum(dim=-1, keepdim=True)
    flat_probs = flat_probs / row_sums

    nearest_idx = flat_probs.argmax(dim=-1, keepdim=True)
    flat_selected = flat_lattice.gather(dim=1, index=nearest_idx).squeeze(-1)
    flat_selected_probs = flat_probs.gather(dim=1, index=nearest_idx).squeeze(-1)
    selected = flat_selected.reshape(leading_shape)
    selected_probs = flat_selected_probs.reshape(leading_shape)
    return selected, selected_probs, summarize_coordinate_probabilities(selected_probs)


class GradcodesLinear(nn.Module):
    """A merge-only quantized linear layer with stage-wise discrete low-rank state."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        module_name: str,
        bits: int,
        group_size: int,
        quant_type: str,
        stage_rank: int,
        amax: int,
        bmax: int,
        search_enabled: bool = True,
        capture_weight_dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super().__init__()
        if stage_rank != -1 and stage_rank < 1:
            raise ValueError("stage_rank must be at least 1, or -1 for elementwise grid search.")
        self.module_name = module_name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.group_size = self.in_features if group_size <= 0 else min(group_size, self.in_features)
        self.quant_type = quant_type
        self.stage_rank = stage_rank
        self.full_matrix_mode = stage_rank == -1
        self.elementwise_step_radius = max(amax, bmax)
        self.amax = amax
        self.bmax = bmax
        self.search_enabled = search_enabled
        self.capture_weight_dtype = capture_weight_dtype
        self.qmin, self.qmax = qrange_from_bits(bits, quant_type=quant_type)

        base_scales, base_codes = grouped_quantize(
            linear.weight.data,
            bits=bits,
            group_size=self.group_size,
            quant_type=quant_type,
        )
        self.register_buffer("base_scales", base_scales)
        self.register_buffer("base_codes", base_codes)
        self.scale_log_factors = nn.Parameter(
            torch.zeros_like(base_scales, dtype=torch.float32),
            requires_grad=search_enabled,
        )
        # A dense FP32 residual costs four bytes per quantized weight and was the
        # largest piece of persistent memory in the original implementation.
        # Keep the legacy dense buffer empty and accumulate committed stages as
        # low-rank factors instead. This preserves the exact residual sum while
        # changing persistent storage from O(out*in) to O(rank*(out+in)).
        residual_codes = torch.empty(0, dtype=torch.int8, device=base_codes.device)
        if self.search_enabled:
            active_codes = (
                torch.zeros(self.out_features, self.in_features, dtype=torch.float32, device=base_codes.device)
                if self.full_matrix_mode
                else torch.zeros(1, dtype=torch.float32, device=base_codes.device)
            )
        else:
            active_codes = torch.zeros(1, dtype=torch.float32, device=base_codes.device)
        self.register_buffer("residual_codes", residual_codes)
        self.register_buffer(
            "residual_a",
            torch.zeros(self.out_features, 0, dtype=torch.float32, device=base_codes.device),
        )
        self.register_buffer(
            "residual_b",
            torch.zeros(self.in_features, 0, dtype=torch.float32, device=base_codes.device),
        )
        self.register_buffer("active_codes", active_codes)
        self.register_buffer("nf4_codebook", get_nf4_codebook(linear.weight.device))
        self.register_buffer("mxfp4_codebook", get_mxfp4_codebook(linear.weight.device))
        self.register_buffer(
            "normalized_code_interval_widths",
            build_normalized_code_interval_widths(
                bits,
                quant_type,
                device=linear.weight.device,
            ),
        )
        active_rank_storage = 0 if not self.search_enabled else (1 if self.full_matrix_mode else self.stage_rank)
        self.register_buffer(
            "active_a",
            torch.zeros(self.out_features, active_rank_storage, dtype=torch.float32),
        )
        self.register_buffer(
            "active_b",
            torch.zeros(self.in_features, active_rank_storage, dtype=torch.float32),
        )

        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.last_weight: Optional[torch.Tensor] = None
        self.active_stage_rank_count = 0
        self.has_residual_update = False
        self.capture_weight_gradients = False

    @property
    def weight_shape(self) -> Tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def storage_device(self) -> torch.device:
        return self.base_scales.device

    def zeros_like_weight(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.weight_shape, device=self.storage_device, dtype=dtype)

    def materialize_grouped_scales(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        grouped_scales = self.base_scales.to(torch.float32) * torch.exp(self.scale_log_factors)
        grouped_scales = torch.clamp(grouped_scales, min=1e-6)
        if dtype != torch.float32:
            grouped_scales = grouped_scales.to(dtype=dtype)
        return grouped_scales

    def materialize_scale_map(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return expand_grouped_scales(
            self.materialize_grouped_scales(dtype=torch.float32),
            original_cols=self.in_features,
            group_size=self.group_size,
            dtype=dtype,
        )

    def materialize_base_code_map(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        dense = self.base_codes.reshape(self.out_features, -1)[:, : self.in_features].contiguous()
        return dense.to(dtype=dtype)

    def materialize_base_weight(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_grouped_state(
            self.base_codes,
            self.base_scales,
            original_cols=self.in_features,
            group_size=self.group_size,
            quant_type=self.quant_type,
            output_dtype=dtype,
        )

    def ensure_search_enabled(self) -> None:
        if not self.search_enabled:
            raise RuntimeError(f"{self.module_name}: this layer is quantized-only and does not participate in search.")

    def set_capture_weight_gradients(self, enabled: bool) -> None:
        self.capture_weight_gradients = bool(enabled and self.search_enabled)

    def reset_active_stage(self) -> None:
        if self.active_codes.numel() > 0:
            self.active_codes.zero_()
        if self.active_a.numel() > 0:
            self.active_a.zero_()
        if self.active_b.numel() > 0:
            self.active_b.zero_()
        self.active_stage_rank_count = 0

    def snapshot_active_stage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ensure_search_enabled()
        if self.full_matrix_mode:
            return (
                self.active_codes.detach().clone(),
                torch.empty(0, device=self.active_codes.device, dtype=torch.float32),
            )
        active_rank_count = max(1, self.active_stage_rank_count)
        return (
            self.active_a[:, :active_rank_count].detach().clone(),
            self.active_b[:, :active_rank_count].detach().clone(),
        )

    def set_active_stage(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self.ensure_search_enabled()
        if self.full_matrix_mode:
            if a.shape != self.weight_shape:
                raise ValueError(
                    f"{self.module_name}: invalid elementwise active code shape {a.shape}, expected {self.weight_shape}."
                )
            if b.numel() != 0:
                raise ValueError(f"{self.module_name}: elementwise mode expects an empty secondary state tensor.")

            self.reset_active_stage()
            self.active_codes.copy_(a.detach().to(self.active_codes.device, dtype=self.active_codes.dtype))
            self.active_stage_rank_count = 1
            return

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{self.module_name}: active stage tensors must be 2D.")
        if a.shape[0] != self.active_a.shape[0]:
            raise ValueError(f"{self.module_name}: invalid a shape {a.shape}.")
        if b.shape[0] != self.active_b.shape[0]:
            raise ValueError(f"{self.module_name}: invalid b shape {b.shape}.")
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"{self.module_name}: stage rank mismatch between a {a.shape} and b {b.shape}.")
        if not 1 <= a.shape[1] <= self.stage_rank:
            raise ValueError(
                f"{self.module_name}: active stage rank must be in [1, {self.stage_rank}], got {a.shape[1]}."
            )

        self.reset_active_stage()
        active_rank_count = a.shape[1]
        self.active_a[:, :active_rank_count].copy_(a.detach().to(self.active_a.device, dtype=self.active_a.dtype))
        self.active_b[:, :active_rank_count].copy_(b.detach().to(self.active_b.device, dtype=self.active_b.dtype))
        self.active_stage_rank_count = active_rank_count

    def active_update(self) -> torch.Tensor:
        if self.active_stage_rank_count <= 0:
            return self.zeros_like_weight(dtype=torch.float32)
        if self.full_matrix_mode:
            return self.active_codes
        active_a = self.active_a[:, : self.active_stage_rank_count]
        active_b = self.active_b[:, : self.active_stage_rank_count]
        return active_a @ active_b.t()

    def residual_code_map(self, *, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        residual: Optional[torch.Tensor] = None
        if self.residual_codes.numel() > 0:
            if self.residual_codes.shape != self.weight_shape:
                raise RuntimeError(
                    f"{self.module_name}: residual code shape {self.residual_codes.shape} "
                    f"does not match weight shape {self.weight_shape}."
                )
            residual = self.residual_codes.to(dtype=dtype)

        if self.residual_a.shape[1] != self.residual_b.shape[1]:
            raise RuntimeError(f"{self.module_name}: committed residual factor ranks do not match.")
        if self.residual_a.shape[1] > 0:
            low_rank_residual = self.residual_a.to(dtype=dtype) @ self.residual_b.to(dtype=dtype).t()
            residual = low_rank_residual if residual is None else residual + low_rank_residual
        return residual

    def current_code(self) -> torch.Tensor:
        merged_code = self.materialize_base_code_map(dtype=torch.float32)
        residual_codes = self.residual_code_map(dtype=torch.float32)
        if residual_codes is not None:
            merged_code = merged_code + residual_codes
        if self.active_stage_rank_count > 0:
            merged_code = merged_code + self.active_update()
        return torch.clamp(merged_code, self.qmin, self.qmax)

    def normalized_interval_width_map(self, codes: torch.Tensor) -> torch.Tensor:
        code_indices = torch.round(codes).clamp(self.qmin, self.qmax).to(torch.long) - self.qmin
        return self.normalized_code_interval_widths[code_indices]

    def effective_step_map(self, codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the local deployed-weight step implied by the current quantization bin.

        For uniform/int4 quantization this reduces to the fixed block scale. For
        NF4/MXFP4 it additionally multiplies by the width of the current code's
        Voronoi cell in the normalized domain, so each position reflects both its
        fixed block scale and the interval of the active lattice point.
        """
        if codes is None:
            codes = self.current_code()
        interval_width_map = self.normalized_interval_width_map(codes)
        return torch.clamp(self.materialize_scale_map(dtype=torch.float32) * interval_width_map, min=1e-6)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes into normalized lattice values.

        For uniform/int4 quantization the code is already the normalized integer
        lattice value after scaling. For NF4/MXFP4 the code is a 4-bit index into
        the non-uniform codebook.
        """
        if is_integer_quant_type(self.quant_type):
            return codes
        if self.quant_type == "nf4":
            code_indices = torch.round(codes).clamp(self.qmin, self.qmax).to(torch.long)
            return self.nf4_codebook[code_indices]
        if self.quant_type == "mxfp4":
            code_indices = torch.round(codes).clamp(self.qmin, self.qmax).to(torch.long)
            return self.mxfp4_codebook[code_indices]
        raise ValueError(f"Unsupported quantization type: {self.quant_type}")

    def materialize_weight(
        self,
        *,
        capture_grad: bool,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        materialize_dtype = (
            self.capture_weight_dtype
            if capture_grad and self.capture_weight_dtype is not None
            else dtype
        )
        if not self.search_enabled or (self.active_stage_rank_count <= 0 and not self.has_residual_update):
            weight = self.materialize_base_weight(dtype=materialize_dtype)
        else:
            weight = self.materialize_scale_map(dtype=torch.float32) * self.decode_codes(self.current_code())
            if materialize_dtype != torch.float32:
                weight = weight.to(dtype=materialize_dtype)
        if capture_grad:
            weight = weight.detach().requires_grad_(True)
            self.last_weight = weight
        else:
            self.last_weight = None
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capture_grad = torch.is_grad_enabled() and self.capture_weight_gradients
        compute_dtype = x.dtype if x.is_floating_point() else torch.float32
        weight = self.materialize_weight(
            capture_grad=capture_grad,
            dtype=compute_dtype,
        )
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=compute_dtype)
        return F.linear(x, weight.to(dtype=compute_dtype), bias)

    def weight_gradient(self) -> torch.Tensor:
        if self.last_weight is None or self.last_weight.grad is None:
            return self.zeros_like_weight(dtype=torch.float32)
        return self.last_weight.grad.detach().to(torch.float32)

    def guidance_matrix(self) -> torch.Tensor:
        return self.weight_gradient() * self.effective_step_map()

    def initialize_active_stage_from_gradient(
        self,
        *,
        active_rank_count: Optional[int] = None,
        power_iterations: int = 8,
    ) -> None:
        self.ensure_search_enabled()
        if self.full_matrix_mode:
            self.reset_active_stage()
            self.active_stage_rank_count = 1
            active_codes = quantize_guidance_tensor(
                -self.guidance_matrix(),
                self.elementwise_step_radius,
            )
            self.active_codes.copy_(active_codes.to(self.active_codes.device))
            return

        active_rank_count = self.stage_rank if active_rank_count is None else active_rank_count
        if not 1 <= active_rank_count <= self.stage_rank:
            raise ValueError(
                f"{self.module_name}: active_rank_count must be in [1, {self.stage_rank}], got {active_rank_count}."
            )

        self.reset_active_stage()
        self.active_stage_rank_count = active_rank_count
        residual_guide = -self.guidance_matrix()

        for rank_idx in range(active_rank_count):
            if residual_guide.abs().max() <= 1e-12:
                a = random_lattice_vector(self.out_features, self.amax, device=self.active_a.device)
                b = random_lattice_vector(self.in_features, self.bmax, device=self.active_b.device)
            else:
                u, v = approximate_top_singular_vectors(residual_guide, power_iterations=power_iterations)
                a = quantize_guidance_vector(u, self.amax)
                b = quantize_guidance_vector(v, self.bmax)

                if torch.count_nonzero(a) == 0:
                    a = random_lattice_vector(self.out_features, self.amax, device=self.active_a.device)
                if torch.count_nonzero(b) == 0:
                    b = random_lattice_vector(self.in_features, self.bmax, device=self.active_b.device)

                sigma = torch.dot(u, residual_guide @ v)
                residual_guide = residual_guide - (sigma * torch.outer(u, v))

            self.active_a[:, rank_idx].copy_(a.to(self.active_a.device))
            self.active_b[:, rank_idx].copy_(b.to(self.active_b.device))

    def initialize_active_stage_like_lora(
        self,
        *,
        active_rank_count: Optional[int] = None,
    ) -> None:
        """
        Initialize the active stage like LoRA:
        A is random on the integer lattice and B starts at zero.
        """
        self.ensure_search_enabled()
        if self.full_matrix_mode:
            self.reset_active_stage()
            self.active_stage_rank_count = 1
            self.active_codes.zero_()
            return

        active_rank_count = self.stage_rank if active_rank_count is None else active_rank_count
        if not 1 <= active_rank_count <= self.stage_rank:
            raise ValueError(
                f"{self.module_name}: active_rank_count must be in [1, {self.stage_rank}], got {active_rank_count}."
            )

        self.reset_active_stage()
        self.active_stage_rank_count = active_rank_count
        random_a = random_integer_lattice_tensor(
            (self.out_features, active_rank_count),
            self.amax,
            device=self.active_a.device,
        )
        self.active_a[:, :active_rank_count].copy_(random_a.to(self.active_a.device))
        self.active_b[:, :active_rank_count].zero_()

    def proposal_guidance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ensure_search_enabled()
        if self.active_stage_rank_count <= 0:
            raise RuntimeError(f"{self.module_name}: active stage is empty. Initialize the stage before proposing.")
        guide = self.guidance_matrix()
        if self.full_matrix_mode:
            return -guide, torch.empty(0, device=guide.device, dtype=torch.float32)
        a = self.active_a[:, : self.active_stage_rank_count].to(torch.float32)
        b = self.active_b[:, : self.active_stage_rank_count].to(torch.float32)
        ga = -(guide @ b)
        gb = -(guide.t() @ a)
        return ga, gb

    def sample_candidate(
        self,
        *,
        proposal_lr_a: float,
        proposal_lr_b: float,
        epsilon: float,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ensure_search_enabled()
        ga, gb = self.proposal_guidance()
        if self.full_matrix_mode:
            candidate_codes = inverse_distance_sample(
                self.active_codes,
                ga,
                learning_rate=proposal_lr_a,
                max_abs_value=self.elementwise_step_radius,
                epsilon=epsilon,
                tau=tau,
            )
            return candidate_codes, torch.empty(0, device=candidate_codes.device, dtype=torch.float32)

        candidate_a = inverse_distance_sample(
            self.active_a[:, : self.active_stage_rank_count],
            ga,
            learning_rate=proposal_lr_a,
            max_abs_value=self.amax,
            epsilon=epsilon,
            tau=tau,
        )
        candidate_b = inverse_distance_sample(
            self.active_b[:, : self.active_stage_rank_count],
            gb,
            learning_rate=proposal_lr_b,
            max_abs_value=self.bmax,
            epsilon=epsilon,
            tau=tau,
        )
        return candidate_a, candidate_b

    def commit_active_stage(self) -> None:
        if not self.search_enabled:
            return
        if self.active_stage_rank_count > 0:
            if self.full_matrix_mode:
                # Elementwise grid mode is intrinsically dense; keep a compact
                # signed integer residual for that explicit opt-in mode.
                committed_residual = self.active_update()
                residual_codes = self.residual_code_map(dtype=torch.float32)
                if residual_codes is not None:
                    committed_residual = committed_residual + residual_codes
                min_value, max_value = torch.aminmax(committed_residual)
                min_value = int(min_value.item())
                max_value = int(max_value.item())
                if -128 <= min_value and max_value <= 127:
                    storage_dtype = torch.int8
                elif -32768 <= min_value and max_value <= 32767:
                    storage_dtype = torch.int16
                else:
                    storage_dtype = torch.int32
                self.residual_codes = committed_residual.to(storage_dtype).contiguous()
            else:
                active_rank_count = self.active_stage_rank_count
                committed_a = self.active_a[:, :active_rank_count].detach().clone()
                committed_b = self.active_b[:, :active_rank_count].detach().clone()
                self.residual_a = torch.cat((self.residual_a, committed_a), dim=1)
                self.residual_b = torch.cat((self.residual_b, committed_b), dim=1)
            self.has_residual_update = True
        self.reset_active_stage()

    def export_search_state(self) -> Dict[str, Union[torch.Tensor, int]]:
        active_rank_count = max(0, self.active_stage_rank_count)
        state: Dict[str, Union[torch.Tensor, int]] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "quant_type": self.quant_type,
            "stage_rank": self.stage_rank,
            "search_enabled": int(self.search_enabled),
            "full_matrix_mode": int(self.full_matrix_mode),
            "active_stage_rank_count": active_rank_count,
            "amax": self.amax,
            "bmax": self.bmax,
            "grouped_scales": self.materialize_grouped_scales(dtype=torch.float32).detach().cpu(),
            "residual_codes": self.residual_codes.detach().cpu(),
            "residual_a": self.residual_a.detach().cpu(),
            "residual_b": self.residual_b.detach().cpu(),
            "active_a": self.active_a[:, :active_rank_count].detach().cpu(),
            "active_b": self.active_b[:, :active_rank_count].detach().cpu(),
        }
        if self.full_matrix_mode:
            state["active_codes"] = self.active_codes.detach().cpu()
        return state


def replace_linear_with_gradcodes(
    model: nn.Module,
    *,
    quantized_modules: Sequence[str],
    target_modules: Sequence[str],
    bits: int,
    group_size: int,
    quant_type: str,
    stage_rank: int,
    amax: int,
    bmax: int,
    capture_weight_dtype: Optional[torch.dtype] = torch.float32,
) -> Tuple[List[str], List[str]]:
    """Wrap quantized linear layers and mark which ones participate in search."""
    quantized: List[str] = []
    searchable: List[str] = []

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and module_name_matches(full_name, quantized_modules):
                search_enabled = module_name_matches(full_name, target_modules)
                wrapped = GradcodesLinear(
                    child,
                    module_name=full_name,
                    bits=bits,
                    group_size=group_size,
                    quant_type=quant_type,
                    stage_rank=stage_rank,
                    amax=amax,
                    bmax=bmax,
                    search_enabled=search_enabled,
                    capture_weight_dtype=capture_weight_dtype,
                )
                setattr(module, child_name, wrapped)
                quantized.append(full_name)
                if search_enabled:
                    searchable.append(full_name)
                continue

            _replace(child, prefix=full_name)

    _replace(model)
    return quantized, searchable


def iter_gradcodes_modules(
    model: nn.Module,
    *,
    search_only: bool = False,
) -> Iterable[Tuple[str, GradcodesLinear]]:
    """Yield wrapped GradcodesLinear modules, optionally restricted to searchable ones."""
    for name, module in model.named_modules():
        if isinstance(module, GradcodesLinear) and (not search_only or module.search_enabled):
            yield name, module


def collect_gradcodes_state(model: nn.Module) -> Dict[str, Dict[str, Union[torch.Tensor, int]]]:
    """Collect per-module residual search state for checkpointing."""
    return {
        name: module.export_search_state()
        for name, module in iter_gradcodes_modules(model)
    }


@torch.no_grad()
def load_gradcodes_state(
    model: nn.Module,
    search_state: Dict[str, Dict[str, Union[torch.Tensor, int, str]]],
    *,
    strict: bool = True,
) -> List[str]:
    """Restore a saved search state into already wrapped Gradcodes modules."""
    wrapped_modules = {name: module for name, module in iter_gradcodes_modules(model)}
    if strict:
        missing_in_model = sorted(set(search_state) - set(wrapped_modules))
        missing_in_state = sorted(set(wrapped_modules) - set(search_state))
        if missing_in_model:
            raise ValueError(f"Checkpoint modules are missing from the model: {missing_in_model[:8]}")
        if missing_in_state:
            raise ValueError(f"Wrapped modules are missing from the checkpoint: {missing_in_state[:8]}")

    loaded: List[str] = []
    for module_name, module_state in search_state.items():
        module = wrapped_modules.get(module_name)
        if module is None:
            continue

        expected = {
            "bits": module.bits,
            "group_size": module.group_size,
            "quant_type": module.quant_type,
            "stage_rank": module.stage_rank,
            "search_enabled": int(module.search_enabled),
        }
        for key, expected_value in expected.items():
            saved_value = module_state[key]
            if key in {"bits", "group_size", "stage_rank", "search_enabled"}:
                saved_value = int(saved_value)
            else:
                saved_value = str(saved_value)
            if saved_value != expected_value:
                raise ValueError(
                    f"{module_name}: checkpoint {key}={saved_value!r}, expected {expected_value!r}."
                )

        saved_grouped_scales = module_state.get("grouped_scales")
        if saved_grouped_scales is not None:
            grouped_scales = saved_grouped_scales.to(module.storage_device, dtype=torch.float32)
            if grouped_scales.shape != module.base_scales.shape:
                raise ValueError(
                    f"{module_name}: grouped scale shape {grouped_scales.shape} does not match "
                    f"{module.base_scales.shape}."
                )
            scale_ratio = torch.clamp(
                grouped_scales / module.base_scales.to(torch.float32).clamp_min(1e-6),
                min=1e-6,
            )
            module.scale_log_factors.copy_(torch.log(scale_ratio))

        saved_residual_codes = module_state.get("residual_codes")
        if saved_residual_codes is None:
            saved_residual_codes = torch.empty(0, dtype=torch.int8)
        residual_codes = saved_residual_codes.to(module.storage_device)
        if residual_codes.numel() > 0 and residual_codes.shape != module.weight_shape:
            raise ValueError(
                f"{module_name}: residual code shape {residual_codes.shape} does not match {module.weight_shape}."
            )
        module.residual_codes = residual_codes

        residual_a = module_state.get("residual_a")
        residual_b = module_state.get("residual_b")
        if residual_a is None and residual_b is None:
            residual_a = torch.zeros(module.out_features, 0, dtype=torch.float32)
            residual_b = torch.zeros(module.in_features, 0, dtype=torch.float32)
        elif residual_a is None or residual_b is None:
            raise ValueError(f"{module_name}: checkpoint must contain both residual_a and residual_b.")
        residual_a = residual_a.to(module.storage_device, dtype=torch.float32)
        residual_b = residual_b.to(module.storage_device, dtype=torch.float32)
        if residual_a.ndim != 2 or residual_b.ndim != 2:
            raise ValueError(f"{module_name}: residual factors must be rank-2 tensors.")
        if residual_a.shape[0] != module.out_features or residual_b.shape[0] != module.in_features:
            raise ValueError(f"{module_name}: residual factor shapes do not match the wrapped weight.")
        if residual_a.shape[1] != residual_b.shape[1]:
            raise ValueError(f"{module_name}: residual factor ranks do not match.")
        module.residual_a = residual_a
        module.residual_b = residual_b

        module.reset_active_stage()
        active_rank_count = int(module_state.get("active_stage_rank_count", 0))
        if active_rank_count > 0:
            if module.full_matrix_mode:
                active_codes = module_state.get("active_codes")
                if active_codes is None:
                    raise ValueError(f"{module_name}: checkpoint is missing active_codes.")
                module.set_active_stage(
                    active_codes.to(module.storage_device, dtype=torch.float32),
                    torch.empty(0, device=module.storage_device, dtype=torch.float32),
                )
            else:
                active_a = module_state.get("active_a")
                active_b = module_state.get("active_b")
                if active_a is None or active_b is None:
                    raise ValueError(f"{module_name}: checkpoint is missing active_a/active_b.")
                if active_a.shape[1] < active_rank_count or active_b.shape[1] < active_rank_count:
                    raise ValueError(f"{module_name}: saved active factors do not cover the active rank.")
                module.set_active_stage(
                    active_a[:, :active_rank_count].to(module.storage_device, dtype=torch.float32),
                    active_b[:, :active_rank_count].to(module.storage_device, dtype=torch.float32),
                )

        scale_changed = bool(module.scale_log_factors.detach().abs().max().item() > 0.0)
        module.has_residual_update = bool(
            module.residual_codes.numel() > 0
            or module.residual_a.shape[1] > 0
            or active_rank_count > 0
            or scale_changed
        )
        module.last_weight = None
        loaded.append(module_name)

    return loaded


def get_named_module(model: nn.Module, module_name: str) -> nn.Module:
    """Resolve a dotted module path from a model."""
    current = model
    for segment in module_name.split("."):
        current = getattr(current, segment)
    return current


def get_named_module_parent(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """Resolve the parent module and leaf attribute for a dotted module path."""
    segments = module_name.split(".")
    if not segments:
        raise ValueError("module_name must not be empty.")

    current = model
    for segment in segments[:-1]:
        current = getattr(current, segment)
    return current, segments[-1]


def materialize_search_state_weight(
    base_weight: torch.Tensor,
    module_state: Dict[str, Union[torch.Tensor, int, str]],
) -> torch.Tensor:
    """Rebuild the deployed weight tensor implied by one saved search state."""
    bits = int(module_state["bits"])
    group_size = int(module_state["group_size"])
    quant_type = str(module_state["quant_type"])
    qmin, qmax = qrange_from_bits(bits, quant_type=quant_type)

    base_scales, base_codes = grouped_quantize(
        base_weight,
        bits=bits,
        group_size=group_size,
        quant_type=quant_type,
    )
    saved_grouped_scales = module_state.get("grouped_scales")
    if saved_grouped_scales is not None:
        grouped_scales = saved_grouped_scales.to(base_weight.device, dtype=torch.float32)
        if grouped_scales.shape != base_scales.shape:
            raise ValueError(
                f"Saved grouped_scales shape {grouped_scales.shape} does not match expected shape {base_scales.shape}."
            )
    else:
        grouped_scales = base_scales
    dense_base_codes = base_codes.reshape(base_weight.shape[0], -1)[:, : base_weight.shape[1]].to(torch.float32)
    saved_residual_codes = module_state["residual_codes"]
    if saved_residual_codes.numel() == 0:
        merged_code = dense_base_codes
    else:
        residual_codes = saved_residual_codes.to(base_weight.device, dtype=torch.float32)
        if residual_codes.shape != base_weight.shape:
            raise ValueError(
                f"Saved residual_codes shape {residual_codes.shape} does not match "
                f"base weight shape {base_weight.shape}."
            )
        merged_code = dense_base_codes + residual_codes

    residual_a = module_state.get("residual_a")
    residual_b = module_state.get("residual_b")
    if residual_a is not None or residual_b is not None:
        if residual_a is None or residual_b is None:
            raise ValueError("Saved residual low-rank state must include both residual_a and residual_b.")
        residual_a = residual_a.to(base_weight.device, dtype=torch.float32)
        residual_b = residual_b.to(base_weight.device, dtype=torch.float32)
        if residual_a.ndim != 2 or residual_b.ndim != 2:
            raise ValueError("Saved residual_a/residual_b must be rank-2 tensors.")
        if residual_a.shape[0] != base_weight.shape[0] or residual_b.shape[0] != base_weight.shape[1]:
            raise ValueError("Saved residual factor shapes do not match the base weight shape.")
        if residual_a.shape[1] != residual_b.shape[1]:
            raise ValueError("Saved residual_a/residual_b ranks do not match.")
        if residual_a.shape[1] > 0:
            merged_code = merged_code + (residual_a @ residual_b.t())

    full_matrix_mode = bool(int(module_state.get("full_matrix_mode", 0)))
    active_rank_count = int(module_state.get("active_stage_rank_count", 0))
    if active_rank_count > 0:
        if full_matrix_mode:
            active_codes = module_state.get("active_codes")
            if active_codes is None:
                raise ValueError(
                    "The saved search state reports elementwise active codes but does not include active_codes."
                )
            active_codes = active_codes.to(base_weight.device, dtype=torch.float32)
            if active_codes.shape != base_weight.shape:
                raise ValueError("Saved active_codes shape does not match the quantized base code shape.")
            merged_code = merged_code + active_codes
        else:
            active_a = module_state.get("active_a")
            active_b = module_state.get("active_b")
            if active_a is None or active_b is None:
                raise ValueError(
                    "The saved search state reports an active stage but does not include active_a/active_b. "
                    "This checkpoint was likely written by an older format and cannot be reconstructed exactly."
                )
            active_a = active_a.to(base_weight.device, dtype=torch.float32)
            active_b = active_b.to(base_weight.device, dtype=torch.float32)
            if active_a.ndim != 2 or active_b.ndim != 2:
                raise ValueError("Saved active_a/active_b must be rank-2 tensors.")
            if active_a.shape[1] < active_rank_count or active_b.shape[1] < active_rank_count:
                raise ValueError("Saved active_a/active_b do not cover active_stage_rank_count.")
            merged_code = merged_code + (active_a[:, :active_rank_count] @ active_b[:, :active_rank_count].t())

    merged_code = torch.clamp(merged_code, qmin, qmax)
    decoded = decode_codes_tensor(
        merged_code,
        quant_type=quant_type,
        qmin=qmin,
        qmax=qmax,
    )
    scale_map = expand_grouped_scales(
        grouped_scales,
        original_cols=base_weight.shape[1],
        group_size=group_size,
        dtype=torch.float32,
    )
    return scale_map * decoded


def apply_search_state_to_model(
    model: nn.Module,
    *,
    search_state: Dict[str, Dict[str, Union[torch.Tensor, int, str]]],
    compute_dtype: Optional[torch.dtype] = None,
    reference_model: Optional[nn.Module] = None,
) -> List[str]:
    """
    Materialize saved Gradcodes search state onto a plain HF model in-place.

    The target modules are replaced with dense nn.Linear layers whose weights are
    reconstructed from the saved discrete codes and the current base weights.
    """
    replaced: List[str] = []

    for module_name, module_state in search_state.items():
        original_module = get_named_module(model, module_name)
        reference_module = original_module if reference_model is None else get_named_module(reference_model, module_name)
        if not hasattr(original_module, "weight"):
            raise TypeError(
                f"{module_name}: expected a linear-like module with a weight attribute, "
                f"got {type(original_module)}."
            )
        if not hasattr(reference_module, "weight"):
            raise TypeError(
                f"{module_name}: expected the reference module to expose a weight attribute, "
                f"got {type(reference_module)}."
            )

        original_device = original_module.weight.device
        base_weight = reference_module.weight.detach().to(device=original_device, dtype=torch.float32)
        materialized_weight = materialize_search_state_weight(base_weight, module_state)
        target_dtype = compute_dtype if compute_dtype is not None else base_weight.dtype
        materialized_weight = materialized_weight.to(device=original_device, dtype=target_dtype)

        bias = getattr(original_module, "bias", None)
        new_linear = nn.Linear(
            in_features=materialized_weight.shape[1],
            out_features=materialized_weight.shape[0],
            bias=bias is not None,
            device=materialized_weight.device,
            dtype=target_dtype,
        )
        with torch.no_grad():
            new_linear.weight.copy_(materialized_weight)
            if bias is not None:
                new_linear.bias.copy_(bias.detach().to(device=materialized_weight.device, dtype=target_dtype))
        new_linear.requires_grad_(False)

        parent_module, child_name = get_named_module_parent(model, module_name)
        setattr(parent_module, child_name, new_linear)
        replaced.append(module_name)

    return replaced


def build_merged_hf_state_dict(
    model: nn.Module,
    *,
    save_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a standard HF-compatible state dict with wrapped GradcodesLinear modules
    materialized back into plain weight tensors.
    """
    wrapped_modules = {name: module for name, module in iter_gradcodes_modules(model)}
    raw_state_dict = model.state_dict()
    merged_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in raw_state_dict.items():
        if any(key.startswith(f"{module_name}.") for module_name in wrapped_modules):
            continue
        merged_state_dict[key] = value.detach().cpu()

    for module_name, module in wrapped_modules.items():
        weight = module.materialize_weight(capture_grad=False).detach()
        if save_dtype is not None and weight.is_floating_point():
            weight = weight.to(save_dtype)
        merged_state_dict[f"{module_name}.weight"] = weight.cpu()

        if module.bias is not None:
            bias = module.bias.detach()
            if save_dtype is not None and bias.is_floating_point():
                bias = bias.to(save_dtype)
            merged_state_dict[f"{module_name}.bias"] = bias.cpu()

    return merged_state_dict
