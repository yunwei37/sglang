"""SM120 must not autotune the NVFP4 FP4 GEMM during graph capture.

The decode and speculative-draft CUDA-graph runners autotune inside capture
(``run_lm_head=True``). On sm120 the FP4 runner backend resolves to
``flashinfer_cutlass``, and profiling that GEMM faults the process -- SIGSEGV in
the driver at ``cuStreamSynchronize`` -- so a serve whose lm_head is NVFP4 dies
while capturing its graphs, before any request is served.

sm120 therefore joins FlashInfer's skipped-op set, which leaves the heuristic
tactic in place. These pin the two things that must hold for that to be a fix
rather than a silent correctness change: the skip is resolved on sm120 (and only
there), and a skipped op still returns the same GEMM.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip("SM120 FP4 autotune skip requires CUDA SM120", allow_module_level=True)

from contextlib import contextmanager
from unittest.mock import patch

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import (
    Fp4GemmRunnerBackend,
    fp4_quantize,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    fp4_gemm,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
)
from sglang.srt.model_executor.runner import flashinfer_autotune

# Draft lm_head NVFP4 GEMM at TP=2: vocab 248320 / 2, K = hidden 5120.
LM_HEAD_TP2_MKN = (9, 5120, 124160)
GROUP_SIZE = 16


@contextmanager
def _exec_skip_ops(user_skip_ops):
    """Stand in for the published exec config bag the skip set is read from."""
    with patch.object(
        flashinfer_autotune,
        "get_exec",
        lambda: type(
            "Cfg",
            (),
            {
                "kernel": type(
                    "K", (), {"flashinfer_autotune_skip_ops": user_skip_ops}
                )()
            },
        )(),
    ):
        yield


class _Runner:
    device = "cuda"


def _resolved_skip_ops(user_skip_ops=None):
    with _exec_skip_ops(user_skip_ops):
        return flashinfer_autotune.get_flashinfer_autotune_skip_ops(_Runner())


def test_sm120_skips_fp4_gemm_autotune():
    assert "fp4_gemm" in _resolved_skip_ops()


def test_user_skip_ops_are_preserved():
    resolved = _resolved_skip_ops({"mxfp8_gemm"})
    assert resolved == {"mxfp8_gemm", "fp4_gemm"}


def test_non_sm120_is_not_skipped():
    """The workaround is scoped to the architecture that faults."""
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        assert "fp4_gemm" not in _resolved_skip_ops()


def _nvfp4_layer(k, n):
    """Weight/scale/alpha laid out exactly as the runner passes them."""
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.rand(n, k // GROUP_SIZE, device="cuda").to(torch.float8_e4m3fn)
    weight, padding_cols = pad_nvfp4_weight(weight.data)
    scales = scale.unsqueeze(0)
    _, rows, cols = scales.shape
    rows_pad = ((rows + 127) // 128) * 128
    cols_pad = ((cols + 3) // 4) * 4
    padded = torch.zeros((1, rows_pad, cols_pad), dtype=scales.dtype, device="cuda")
    padded[0, :rows, :cols] = scales
    padded = padded.reshape(1, rows_pad // 128, 4, 32, cols_pad // 4, 4)
    padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous().reshape(rows_pad, cols_pad)
    alpha = torch.tensor(1.0 / (448 * 6), device="cuda", dtype=torch.float32)
    return weight, padded, alpha, padding_cols


def test_skipped_fp4_gemm_profiles_nothing_and_matches_cutlass():
    from flashinfer.autotuner import AutoTuner, autotune

    fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.FLASHINFER_CUTLASS
    assert fp4_utils.get_fp4_gemm_runner_backend().get_flashinfer_backend() == "cutlass"

    m, k, n = LM_HEAD_TP2_MKN
    torch.manual_seed(0)
    weight, weight_sf, alpha, padding_cols = _nvfp4_layer(k, n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 8
    x_fp4, x_sf = fp4_quantize(x, torch.tensor(1.0, device="cuda", dtype=torch.float32))
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, padding_cols)

    def run():
        return fp4_gemm(x_fp4, weight.T, x_sf, weight_sf.T, alpha, torch.bfloat16, n)

    tuner = AutoTuner.get()
    tuner.clear_cache()
    expected = run()
    torch.cuda.synchronize()

    tuner.clear_cache()
    with autotune(True, skip_ops=_resolved_skip_ops()):
        actual = run()
        torch.cuda.synchronize()
    assert len(tuner.profiling_cache) == 0, (
        "a skipped fp4_gemm must not be profiled; profiling is what faults on sm120"
    )

    assert actual.shape == (m, n)
    assert torch.equal(actual, expected), "heuristic fallback diverged from cutlass"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
