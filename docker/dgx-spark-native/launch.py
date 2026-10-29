"""Launch one native SGLang rank using the existing model service's placement."""

import json
import os
from pathlib import Path
import sys


def command(env, extra=()):
    model = env.get("MODEL_PATH", "/models/DeepSeek-V4.1-Flash")
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--trust-remote-code",
        "--load-format",
        "safetensors",
        "--served-model-name",
        env.get("SERVED_MODEL_NAME", "deepseek-v4.1-flash"),
        "--tp",
        "4",
        "--ep-size",
        env.get("EP_SIZE", "2"),
        "--nnodes",
        "4",
        "--node-rank",
        env["NODE_RANK"],
        "--dist-init-addr",
        env["DIST_INIT_ADDR"],
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--fp8-gemm-backend",
        "flashinfer_cutlass",
        "--enable-decoder-swa-bounded-replay",
        "--tool-call-parser",
        "deepseekv41",
        "--reasoning-parser",
        "deepseek-v41",
        "--host",
        "0.0.0.0",
        "--port",
        env.get("SERVER_PORT", "8000"),
        "--mem-fraction-static",
        env.get("MEM_FRACTION_STATIC", "0.80"),
        "--cuda-graph-max-bs-decode",
        "16",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-dspark-block-size",
        env.get("DSPARK_BLOCK_SIZE", "5"),
        *extra,
        "--context-length",
        "1048576",
    ]
    secret = env.get("SGLANG_API_KEY") or env.get("VLLM_API_KEY")
    if secret:
        args += ["--api-key", secret]
    return args


def main():
    env = dict(os.environ)
    # The image never downloads or re-hashes the existing local checkpoint.
    env["DSV41_SOURCE"] = env.get("MODEL_PATH", "/models/DeepSeek-V4.1-Flash")
    # The existing Mesh carrier supplies both the NCCL library and the net plugin.
    # No wheel directory or Python-version path needs to be pinned.
    if Path("/opt/nccl-mesh/lib").is_dir():
        env["SGLANG_NCCL_SO_PATH"] = "/opt/nccl-mesh/lib/libnccl.so.2"
        env["LD_LIBRARY_PATH"] = "/opt/nccl-mesh/lib:" + env.get("LD_LIBRARY_PATH", "")
    argv = command(env, sys.argv[1:])
    print(
        json.dumps(
            {
                "engine": "sglang-native",
                "node_rank": env["NODE_RANK"],
                "context_length": 1048576,
                "model_path": env["DSV41_SOURCE"],
                "thinking_default": env.get("SGLANG_DEFAULT_THINKING"),
                "provenance": str(Path(__file__).with_name("provenance.json")),
            }
        ),
        flush=True,
    )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
