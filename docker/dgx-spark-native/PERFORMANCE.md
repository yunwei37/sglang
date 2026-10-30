# Four-Spark performance and quality acceptance

Use the existing four-rank owner and existing evaluation client. Source/build
work stays in this Workspace. Keep context 1,048,576 and default thinking on;
the speedbench requests explicitly turn thinking off, as in the saved vLLM run.
No host networking, clocks, compaction or node restart is part of this change.

## Actual baseline, retrieved from Filer on 2026-10-30

`artifacts/sglang-native/baseline/baseline-review.json` records the raw inputs
and exclusions. Archives are under Filer
`/evaluations/2026-10-28-deepseek1017/official-1bae0e1-*.tar.gz`.

| Measurement | Official vLLM, same fixed 1017 weights | Comparison needed |
| --- | --- | --- |
| MMLU 5-shot, lm_eval 0.4.13 | 12,077 / 14,042 = 86.0063%; 14,015 unique contexts | Same saved prompts and single-token A-D protocol; preserve all 14,042 sample results |
| Pilot | one-forward 103/114; teacher-forced 105/114 | Keep separate references and inspect raw margins |
| speedbench full C1 | ALL 44.53, code 59.92 tok/s; 73 requests, 0 failed, 71 valid | Same 73 prompts; report each arm and paired valid intersection |
| load block tok/s | C1 36.75, C4 62.62, C8 83.62; 60 s per level | Repeat these settings for engine comparison; C16/90 s is a separate target run |
| cold prefill | 8K 2249, 16K 2202.5, 32K 2189.8, 64K 2128.4, 128K 2046.1 tok/s | Preserve salt, cold/cache distinction, actual token count and TTFT |
| near 1M | 1,039,095 input tokens, 3 needles correct, 966.974 s | Replay the original payload/generator, not just its saved answer |

`agentic-01` and `agentic-06` finished successfully with tool calls. Their timed
decode lengths were 24 and 23 tokens, below speedbench's existing 32-token
full-mode statistic threshold. Preserve these business successes; do not
silently count them as failures, pad their outputs or drop the prompts.
The saved near-1M JSON contains seed 20261028, needle depths 0.1/0.5/0.9 and
answers, but does **not** contain the payload/generator; the outer client must
retain that input to establish the same retrieval task.

Both pilot references have identical reconstructed token contexts for all 114
questions with the official tokenizer and `add_bos_token=false`. The two winner
changes are college_computer_science doc 1 and public_relations doc 1. The
one-forward margins are 0.125 and 0.625 logprob; teacher-forced margins are
0.14030 and 0.27630. These are measured method differences, not equivalent
floating-point paths. The outer handoff confirms PreTrainedTokenizerFast with
add_special_tokens=False/add_bos_token=false. The frozen IDs reproduce that
encoding; the original archive did not separately save wire IDs.

## Reuse speedbench, do not add a second benchmark implementation

The available fixed public 0.3 source is
`knapcio/speedbench@e95a638b5f92d26cc0724e7941abb8053cde3e20`, retained locally
at `artifacts/sglang-native/speedbench-e95a638`. Its `report` command reproduced
the saved baseline. Its `compare` pairs by prompt key, excludes sentinels and
keeps per-arm exclusion accounting. Use the same evaluator environment/location
for both engines; client placement and network hop affect TTFT.

Run on the outer client's already-authorized backend route, with its existing
OPENAI_API_KEY environment. Commands below are the published measurement
definitions, not service limits. `PYTHONPATH` points at the retained speedbench
source; model name remains the existing backend ID (or local-large when testing
the gateway separately).

```sh
python -m speedbench run --url "$EXISTING_BACKEND_URL" \
  --model deepseek-v4.1-flash --label sglang-official-ep2 \
  --mode full --conc 1 --max-tokens 256 --temperature 0 \
  --effort none --effort-field both --metrics --out sglang-ep2-full.json

python -m speedbench load --url "$EXISTING_BACKEND_URL" \
  --model deepseek-v4.1-flash --conc 1,4,8 --duration 60 \
  --effort none --effort-field both --metrics --out sglang-ep2-load.json

python -m speedbench longctx --url "$EXISTING_BACKEND_URL" \
  --model deepseek-v4.1-flash --contexts 8k,16k,32k,64k,128k \
  --decode-contexts 32k,64k,128k --reps 2 \
  --effort none --effort-field both --metrics --out sglang-ep2-longctx.json
```

Keep continuous per-chunk usage enabled. This source natively requests SGLang
`return_spec_tokens_details` and reads `sglext.spec_tokens_details`, supported
by the shipped core. Check those real response fields and speculative metrics
before interpreting tokens per step. SSE event counts alone are insufficient
when an engine coalesces tokens. Compare verifies effective request settings;
an engine-specific telemetry-field difference should be reported explicitly
if its `--force` option is needed. Do not silently relax workload differences.

The target run is full C1 (two complete runs, report both and their median),
load C1/C4/C16 with `--duration 90`, and cold 4K/64K with `--reps 3`. A 388K
or near-1M varied-text cold prompt supplements the curve. Keep the initial
warm-up outside the measurement; record the tool's cached-token counts. A
fresh salt creates a cold prefix without flushing the serving cache or
restarting a healthy owner. Cold prefill uses actual prompt tokens / TTFT;
decode geomean and load block throughput are separate metrics.

Published 89.3/112.2/211 results used mrfakename@517efab, switched RoCE and a
different host/client setup. Latest v2.6.2 release-gate official results are
C1 88.38, C16 207.62 and cold 1M 5310 tok/s. These remain external references;
the user's 89.3/112.2/211/5300–6000 targets are unchanged. First compare the
same official weights across engines, then compare all three SGLang weights
on one frozen runtime/protocol. A faster speculative trajectory is part of
the model result, not proof that the underlying step became faster.

## EP1 route and the minimum meaningful A/B

FlashInfer MXFP4 cannot execute EP1/TP4's 576-wide expert shard (it requires
128 alignment). Do not change EP_SIZE alone or pad/requantize checkpoint
weights to bypass the check. Latest recipe's packaged b12x_next supports
64 alignment and the existing adapter replaces weight preparation and the
MoE runner together. Its partial weights still cover every expert across TP.

The v2.6.2 image includes Martin Vit's b12x d7937d7f producer-claim barrier
fix. **Use that image before the EP1 trial.** EP2 FlashInfer in the old image
does not execute the affected b12x_next kernel and can finish its startup and
baseline evidence while the replacement is prepared.

After the EP2 baseline, the outer owner can trial these existing upstream
switches, leaving Mesh/Ring and all other settings unchanged:

```text
EP_SIZE=1
DSV41_MOE_B12X_NEXT=1
DSV41_MOE_B12X_NEXT_DETERMINISTIC=1
```

This is a comparison of complete MoE routes (backend, sharding and deterministic
reduction), not a claim to isolate one flag. Require actual log lines showing
the adapter armed, EP1 and prepared N=576 plans, then record the frozen pilot,
business probes, peak/steady memory, full C1 and load throughput. MMLU scoring
must still return all four full-vocabulary probabilities; no logit screen may
substitute for their values. The b12x and FlashInfer floating paths are not
bit-identical. If attribution is needed, b12x EP2 is the intermediate route;
its N=1152 deterministic kernel is unsupported, so leave determinism off there
and describe that difference rather than pretending the modes are identical.

Compilation/tuning caches default to /state in the adapter. Before repeated
trials, inspect their actual location and bytes. If persistence is needed,
reuse the already-mounted per-node /cache through
`B12X_NEXT_COMPILE_CACHE_DIR=/cache/b12x-next-compile`; no new volume or owner
is needed. Do not use cache-only mode before a complete first compilation.

Next test `DSV41_PREFILL_NATIVE_EXTRA=1` alone: this uses prebuilt sm_121a
FlashInfer native 128/256 extra pages, removes the persistent extra-page split
copy, and must show DSV41_PCE_HIT for actual page sizes plus successful wide
verify at C16. Keep it off in the control boot. Run the upstream GPU parity
test and the same near-1M request; measure pool size, MemAvailable and post-pool
allocations as well as throughput. More free memory is useful only if 1M and
quality actually pass.

Then investigate bounded-indexer 8192-token prefill chunks and memory fraction
together using measured transient headroom. Engram packing of the existing
checkpoint, prefetch, prepared MoE launches and sequence-parallel prefill are
later candidates. Do not copy the entire switched production environment:
RoCE two-shot, compact gather and L2-prefetch hooks have transport dependencies.
The shipped image deliberately retains the proven NCCL/Mesh ring; a custom
ring collective needs its own real transport evidence. Permanent performance
settings follow reproducible A/B and the user's selection.

## Observed first-boot correction (2026-10-30)

The four ranks initialized with Mesh NCCL 2.29.7 and loaded the official shards.
At memory fraction 0.80, pool sizing failed: the SWA cap was 707,840 tokens,
15.78 GiB, against an available pool budget of 8.13 GiB. Native max-running
requests resolved to 256; the default prefix-tail count is four times that,
1,024. This is actual startup failure evidence, not a throughput result.
The outer owner is retrying MEM_FRACTION_STATIC=0.90 with context and concurrency
unchanged. Retain that explicit trial while inspecting full/SWA token capacities,
post-pool allocations, graph capture and near-1M/vision transient headroom.
Native-page extra-cache savings occur after pool sizing; that feature alone
does not fix the initial fixed-SWA budget failure. Do not reduce context or
silently add a concurrency cap to make readiness pass.

The first startup also exposed api_key in INFO server_args dumps. The successor
applies native-runtime.patch: api_key/admin_api_key and their CLI/Engine launch
arguments are redacted in the shared diagnostic renderers used by startup logs
and info readbacks. Runtime authentication values remain unchanged. This also
keeps speedbench metadata archives from copying a credential via /server_info.
After all ranks run the successor, remove the temporary warning-only log level
so normal pool/graph diagnostics are available. Preserve outer log filtering;
do not retrieve, print, copy or rotate the existing credential.
