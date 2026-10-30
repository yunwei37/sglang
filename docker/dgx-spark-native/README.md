# Native DeepSeek V4.1 on the four-Spark ring

This directory stages the native FP8/FP4 SGLang recipe from
[knapcio's project](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/cfe3cfb81e27675eaef69e1c65bfc6d63b7315aa).
It does not deploy or change the existing model service. Infrastructure and GPU
handover remain with the `spark-manage` Flux owner.

```sh
python docker/dgx-spark-native/prepare.py
python docker/dgx-spark-native/validate.py
docker buildx build --platform linux/arm64 \
  --file artifacts/sglang-native/v262/context/Dockerfile \
  --tag docker.io/library/sglang-dsv41-native:cfe3cfb-swa-20261030b \
  --output type=oci,dest=sglang-native-arm64.oci.tar \
  artifacts/sglang-native/v262/context
```

Import the fixed OCI tag through the existing containerd lifecycle, or use a
published image digest. The generated
`provenance.json` records the recipe commit, SGLang commit, ordered backports,
base ARM64 manifest and kernel package versions. No checkpoint download or
SHA256 download/install/startup/acceptance check is added. The base is the
ARM64 manifest of `lmsysorg/sglang:dev-dsv41`, resolved on 2026-10-29. Its bundled
FlashInfer is 0.6.18. The two updated packages are `sglang-kernel==0.4.7` and
`sgl-deep-gemm==0.2.0`. Recipe adapters carry their upstream AGPL-3.0 license;
the image includes it and the b12x_next license. Local adapter changes are
reproducible from `prepare.py`. The previous f744d83 image and its sources remain
available; the latest source adds the upstream b12x producer-claim barrier fix
before the EP1 trial and packages the optional native-page prefill module.
`build_pce.py` compiles its unchanged FlashInfer kernel templates for sm_121a
using the installed image headers, without importing Torch or requiring a GPU.
`pce-provenance.patch` removes the new hash acceptance gates while retaining
source anchors, functional dispatch validation and digest provenance.
`native-runtime.patch` redacts authentication fields and launch arguments in
startup logs and info readbacks. The underlying authentication values and
middleware remain unchanged. `dspark-swa-budget.patch` uses the actual bundled
DeepSeek draft stages when budgeting paged SWA, rather than the target's inherited
40-layer depth. Its CPU resolver regression is
`python docker/dgx-spark-native/test_dspark_swa_budget.py -v`;
four-rank startup and workload acceptance remain required. Native
extra pages and the solo-short-pass scheduler remain off pending field A/B.
See [PERFORMANCE.md](PERFORMANCE.md) for the retrieved baseline, exact measurement
definitions, EP1 route and outer-owner acceptance commands.

## Differences from the upstream switched profile

| Difference | Reason |
|---|---|
| Existing NCCL 2.29.7 / Mesh 19924dcc, RoCEnante off | The live switchless ring already has this transport. No switched-fabric one-shot collective or new hairpin/route configuration is introduced. |
| Fixed local checkpoint mount, direct SGLang exec | All four nodes already have the full pinned weights. No upstream NFS, downloader, re-hashing, startup smoke traffic or key-file copy is needed. |
| Context 1,048,576; default thinking enabled | Explicit user requirements. Includes #45762/#45804 Chat/Responses parser parity. The backend name stays `deepseek-v4.1-flash`, preserving the existing router's `local-large` mapping. `local-small` is unaffected. |
| EP2 with FlashInfer MXFP4 initially | At TP4/EP1 the expert width is 2304/4=576, which is not divisible by the backend's required 128. The packaged b12x_next EP1 route is an A/B candidate. |
| NVMe Engram, bounded indexer, SM12x split metadata, SWA safety backports and folded-result fence | Required memory/correctness adaptations for the native path; full Engram tables cannot occupy Spark unified memory. |
| Upstream output cap and repetition abort disabled | Preserve native completion semantics and the client's requested budget. |
| No extra checksum gates in preparation/build/default serving | User requirement. Anthropic and native-page source/module digests do not gate execution; functional and idempotency checks remain. Other optional adapters retain upstream behavior while disabled. |
| Initial memory fraction 0.80, row cache 4 GiB, graph batch sizes through 16 | Diagnostic starting values from the recipe, pending fleet measurements. The adapter's 64 GiB default cache plus model/KV is unsuitable for unified memory; large speculative graph pools also consume it. Running/queue capacities remain engine defaults; graph size is not a concurrency cap. |

Other optional performance adapters are packaged but remain disabled. No
switched-fabric performance result is a result for this ring. The initial
memory/cache/graph choices are test candidates, not qualified permanent tuning.

## Cutover contract for the existing Flux owner

Wait for active vLLM evaluations and requests to finish. Preserve their results
and the original declaration. Replace the image/entrypoint within the existing
head Deployment and worker DaemonSet; do not start a second four-rank owner.

Supply `NODE_RANK` and `DIST_INIT_ADDR` from the current authoritative rank map
and distributed-init endpoint. Keep the existing model mount at
`/models/DeepSeek-V4.1-Flash` (official revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` initially), host networking/IPC, GPU
allocation, RDMA devices, existing NCCL Mesh carrier and existing transport
environment. Keep the existing backend API-key Secret reference; the launcher
accepts `VLLM_API_KEY` as well as `SGLANG_API_KEY` without persisting it.

Remove the vLLM entrypoint and Python overlay mounts, the vLLM tokenizer patch,
the wheel-specific vLLM NCCL mount, `VLLM_*` engine settings other than the reused
key reference, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and vLLM's
NVMe KV extension. The image has its own native runtime. Keep the carrier
mounted at `/opt/nccl-mesh`; the launcher selects its library for SGLang pynccl
and puts its `lib` first in the child's dynamic-library search path. This does
not prove which NCCL PyTorch loaded: inspect the final ARM64 image's actual wheel
directory and preserve the existing carrier-over-wheel mount at that discovered
path if its RPATH takes precedence. Verify the serving processes' `/proc/PID/maps`.
Reuse node-local cache storage for JIT caches and an
optional packed-Engram directory. No NFS or new service/Secret is needed.

Engram works directly from the existing checkpoint. Packing is an optional
disk-layout optimization, not a startup condition. Once node-local space is
confirmed, the image supports:

```sh
python /opt/dsv41/pack_engram.py --model /models/DeepSeek-V4.1-Flash \
  --rank "$NODE_RANK" --tp 4 --out /existing-node-local-path
```

Set `DSV41_PACKED_DIR` to that directory only for the packed/unpacked A/B. There
are two files per rank; the recipe estimates approximately 48 GiB per host.
Reuse across the three weight variants only after the prepared tensor diff
confirms that both Engram tables are identical.

## Evidence after handover

```sh
python docker/dgx-spark-native/probe.py \
  --base-url "$OPENAI_BASE_URL" --model local-large \
  --output artifacts/sglang-native/business-official.json
```

Authentication comes from the existing `OPENAI_API_KEY`, `API_KEY`, `SGLANG_API_KEY` or
`VLLM_API_KEY` environment. The evidence contains only the synthetic requests,
responses and elapsed times. The probe verifies default separate reasoning,
explicit thinking off, the vLLM thinking alias, a tool round trip, JSON schema,
streaming finish/usage and native vision. It does
not measure model quality or prove the entire context window.

For the unchanged lm_eval 0.4.13 five-shot MMLU client, see [MMLU.md](MMLU.md).
It uses the existing native token-specific output-logprob API without prompt
logprobs, retokenizing requests, another server or an image rebuild.

Also record all four ranks' image IDs, Ready/restart counts, source provenance,
NCCL library/plugin paths, native `/server_info` effective context, KV capacity,
and graph-capture completion. Then run the outer owner's existing near-1M
retrieval, vision/structured/streaming checks, MMLU and pinned speedbench.
Preserve the same prompts, request budgets, concurrency, sampling, thinking
mode, cache state and harness revision for vLLM/SGLang on the official weights.
Keep thinking-on results separate from speedbench's thinking-off protocol.

On SGLang, compare official, pqhaz Heretic `12d58ad` and mrfakename `517efab`
with exactly the same image and runtime choices. Record full checkpoint
revisions from their actual local provenance, not these abbreviated refs alone.
Evaluate performance changes one at a time: EP2 versus b12x_next EP1, Engram
packing/cache, prefill chunking, then speculative adapters. Measure repeated
runs and actual accepted tokens/step as well as output tokens/s. Do not retain
optional performance switches without a reproducible fleet A/B and the user's
selection. Restore the original Flux declaration if a candidate fails.

## Current candidate: upstream v2.6.2 Production profile

The previous stripped launch.py is retired. prepare.py derives production.env
from the pinned recipe's .env.tp4.example Production block and README release
gate. The existing fleet wrapper sources that file and runs upstream boot.py
with the existing injected API credential. All production compute/prefill
switches are included; the documented switchless-ring NCCL fallback disables
RoCEnante-specific transport/gather/two-shot switches. No hardware forwarding,
clock cap, extra response limit or adaptive SWA allocator is introduced.

The profile uses TP4/EP1, 1M context, 16 running requests, static memory 0.78,
4,194,304 KV tokens, packed local Engram, 4 GiB/16-way cache and 96 I/O threads.
The encoder-only replay and diagnostic autotune-disable overrides are removed.
Local production-profile.patch updates checkpoint provenance, avoids persisting
another API-key copy and removes the four newly enabled source-SHA gates while
retaining functional validation. Digests remain provenance. Qualification is
pending; the prior EP1 text baseline is not the upstream performance result.
