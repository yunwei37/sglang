#!/usr/bin/env python3
"""Stage the pinned native recipe as a self-contained ARM64 build context."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

RECIPE_REPO = "https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4.git"
RECIPE_COMMIT = "cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa"
SGLANG_COMMIT = "f80c91a4b97e52e17fbcff088aa6b47b6270f6b9"
BASE_IMAGE = "lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9"


def fetch_script(recipe):
    # Retain upstream's exact backport order and revision-pinned helper downloads.
    # Checksums are provenance, never download/build/acceptance gates here.
    source = (recipe / "scripts/fetch-sglang-canary.sh").read_text()
    lines = []
    for line in source.splitlines():
        if "| shasum -a 256 -c -" in line:
            continue
        if line.startswith('python3 "$ROOT/scripts/verify_'):
            continue
        if line.startswith('python3 "$ROOT/scripts/test_'):
            continue
        if line.startswith("ROOT="):
            line = "ROOT=" + shlex.quote(str(recipe))
        # No task-specific download deadline; curl's error status remains effective.
        line = line.replace(" -m 600 ", " ").replace(" -m 120 ", " ")
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sglang-native/v262/context")
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("artifacts/sglang-native/recipe-cfe3cfb")
    )
    args = parser.parse_args()
    recipe = args.recipe.resolve()
    if not recipe.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", RECIPE_REPO, str(recipe)], check=True
        )
        subprocess.run(
            ["git", "-C", str(recipe), "checkout", "--detach", RECIPE_COMMIT],
            check=True,
        )
    ref = subprocess.check_output(
        ["git", "-C", str(recipe), "rev-parse", "HEAD"], text=True
    ).strip()
    if ref != RECIPE_COMMIT:
        raise ValueError(
            f"Use recipe {RECIPE_COMMIT}; found {ref}. Existing tree left intact."
        )
    engine = recipe / "runtime/sglang-canary/python"
    receipt = recipe / "runtime/sglang-canary/REF"
    if not receipt.exists() or receipt.read_text().strip() != SGLANG_COMMIT:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as script:
            script.write(fetch_script(recipe))
            script.flush()
            subprocess.run(["bash", script.name], check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (engine, "python"),
        (recipe / "adapter", "adapter"),
        (recipe / "scripts", "scripts"),
        (recipe / "runtime/b12x_next/b12x_next", "b12x_next"),
    ]:
        shutil.copytree(
            src,
            output / dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.so", "*.orig", "*.rej"
            ),
        )
    here = Path(__file__).resolve().parent
    subprocess.run(
        ["patch", "-p1", "-i", str(here / "native-runtime.patch")],
        cwd=output,
        check=True,
    )
    shutil.copy2(
        recipe / "runtime/flash_mla_sm120.canary.py",
        output / "python/sglang/kernels/ops/attention/flash_mla_sm120.py",
    )
    pce = output / "runtime/flashinfer_pce"
    pce.mkdir(parents=True, exist_ok=True)
    for name in ["gen_pce_sources.py", "APPROVED_CLOSURE.json", "PROVENANCE.json"]:
        shutil.copy2(recipe / "runtime/flashinfer_pce" / name, pce / name)
    shutil.copy2(
        recipe / "runtime/flashinfer_pce/dsv41_pce.py",
        output / "python/sglang/kernels/ops/attention/dsv41_pce.py",
    )
    subprocess.run(
        ["patch", "-p1", "-i", str(here / "pce-provenance.patch")],
        cwd=output,
        check=True,
    )
    shutil.copy2(here / "build_pce.py", pce / "build_pce.py")
    shutil.copy2(recipe / "NOTICE", output / "NOTICE.recipe")
    # Apply the recipe's native text-placeholder encoding fix to the staged tree.
    patch = (recipe / "runtime/patch_encoding_dsv41.py").read_text()
    patch = patch.replace("/sgl-workspace/sglang/python", str(output / "python"))
    subprocess.run(["python3", "-c", patch], check=True)
    # Source is already fixed by Git revision. Remove the optional Anthropic
    # source-hash check; retain its functional validator and idempotency checks.
    compat = output / "adapter/encoding_compat.py"
    text = compat.read_text()
    text = text.replace(
        "        if hashlib.sha256(source.encode()).hexdigest() != ANTHROPIC_THINKING_SHAPE_SHA256:\n"
        "            raise ValueError('AnthropicThinkingParam._validate_thinking_shape source drift')\n",
        "",
    )
    compat.write_text(text)
    for name in ["Dockerfile", "launch.py"]:
        shutil.copy2(here / name, output / name)
    for src, dst in [
        (recipe / "LICENSE", "LICENSE.recipe"),
        (recipe / "runtime/b12x_next/LICENSE", "LICENSE.b12x_next"),
    ]:
        shutil.copy2(src, output / dst)
    provenance = {
        "recipe": RECIPE_REPO,
        "recipe_commit": ref,
        "sglang_commit": SGLANG_COMMIT,
        "base_image": BASE_IMAGE,
        "kernel_packages": {"sglang-kernel": "0.4.7", "sgl-deep-gemm": "0.2.0"},
        "backports": (recipe / "runtime/sglang-canary/PATCHES")
        .read_text()
        .splitlines(),
        "b12x_next_commit": (recipe / "runtime/b12x_next/b12x_next/SOURCE_COMMIT")
        .read_text()
        .strip(),
        "b12x_next_patches": (recipe / "runtime/b12x_next/b12x_next/SOURCE_PATCH")
        .read_text()
        .strip(),
        "native_extra_pages": "prebuilt sm_121a; opt-in; digests are provenance only",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {output}")


if __name__ == "__main__":
    main()
