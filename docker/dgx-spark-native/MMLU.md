# MMLU compatibility for the shipped native image

No image rebuild or serving change is required. The existing lm_eval 0.4.13
client can use `mmlu_native.attach(lm, native_url, evidence_path)` before its
unchanged `simple_evaluate(model=lm, ...)` call. Keep the same tasks, five-shot
configuration, tokenizer revision, add-BOS setting, task/few-shot seeds, sample
IDs, concurrency and sample logging. The public serving names remain unchanged.
The native URL is the existing backend's `/generate` endpoint, authenticated
through the existing client's headers. This is a client method replacement,
without an API proxy or another model server.

```python
from mmlu_native import attach

# lm is the same configured LocalCompletionsAPI object as the baseline.
attach(lm, native_url=existing_backend_generate_url,
       evidence_path=output_directory / "native-score-trace.jsonl")
# Continue the existing lm_eval.simple_evaluate(model=lm, ...) invocation.
```

The replacement receives the harness's already-tokenized `(cache_key,
context_ids, continuation_ids)` triples. It groups identical contexts, sends
one scoring request per unique context and restores every original result
position and cache-hook result. It does not retokenize or append the candidate
to the model input. Multi-token continuations are unsupported by this method;
there is no fallback to bounded replay's unavailable prompt-logprob path.
Do not share a response cache between checkpoints, engines or scoring methods.

## Exact request and score semantics

```json
{
  "input_ids": [0, 10, 11],
  "sampling_params": {"temperature": 0, "max_new_tokens": 1},
  "return_logprob": true,
  "logprob_start_len": 3,
  "token_ids_logprob": [334, 406, 345, 420],
  "top_logprobs_num": 1,
  "return_text_in_logprobs": false,
  "stream": false
}
```

The three input IDs above are a synthetic shape example. Actual calls send the
entire frozen MMLU context, with `logprob_start_len=len(input_ids)`. The candidate
IDs encode **`" A"`, `" B"`, `" C"`, `" D"`**, including the leading space;
bare letters have IDs 35, 36, 37, 38. This was checked with the official 1017
tokenizer at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

Read `meta_info.output_token_ids_logprobs[0]`: each entry is
`[logprob, token_id, null]`. Match by token ID, not array position. The first
output position scores the next token conditioned on the original context.
The greedy sampler's `log_softmax` normalizes over the full vocabulary; there
is no A/B/C/D grammar, logit bias or four-choice renormalization. The global
top-1 is requested only to provide lm_eval's actual `is_greedy` boolean, which
can be false for every answer when another token wins globally. MMLU accuracy
uses the ranking of the four candidate log-likelihoods.

This is raw token scoring, as in the baseline `/v1/completions` protocol.
It applies no chat template or thinking opener. The serving default thinking
setting stays enabled and is separately checked with the business API probe.

The fixed source's decoder bounded-replay guard permits start lengths at the
prompt boundary and rejects reading earlier prompt positions. The target
worker samples the prefill's last-position logits, the output logprob processor
gathers the requested IDs, and the scheduler/tokenizer return them as output
logprobs. These paths are exercised against the shipped source by the CPU
contract tests, including a real vocabulary-width synthetic tensor.

With DSpark enabled, `Req.is_prefill_only` is false even at
`max_new_tokens=0`. Consequently, zero-token scoring must not be described as
an already-qualified sampling/draft bypass. The primary probe requests one
token. A separate zero-token replay is available for field comparison. One
request per context means one target next-token scoring operation for the
four answers; it does not claim one CUDA launch, one prefill chunk, or no
DSpark draft preparation.

## Freeze the saved baseline, then replay after GPU handover

Use the baseline's saved standard `log_samples` files, keeping its tokenizer
directory and add-BOS setting. Freezing runs entirely on CPU. It calls the
actual lm_eval 0.4.13 `_encode_pair`, including its trailing-space handling,
and saves the original prompt hash, full input IDs, candidates, reference
scores, reference method and sample identity. Do not regenerate five-shot
prompts or use a different tokenizer configuration.
The replay's correct/scored counters are pilot diagnostics; the full MMLU
aggregate must still come from the same standard lm_eval task aggregation.

```sh
python docker/dgx-spark-native/mmlu_native.py freeze \
  --samples /existing/reference-oneforward-samples.jsonl \
  --tokenizer /existing/exact-baseline-tokenizer \
  --reference-method vllm-oneforward \
  --output artifacts/sglang-native/mmlu-oneforward-frozen.jsonl
```

Only after the outer owner releases the GPU window and starts the existing
service on SGLang:

```sh
python docker/dgx-spark-native/mmlu_native.py replay \
  --prompts artifacts/sglang-native/mmlu-oneforward-frozen.jsonl \
  --native-url "$EXISTING_BACKEND_GENERATE_URL" \
  --output artifacts/sglang-native/mmlu-sglang-pilot.jsonl
```

Authentication uses the existing OPENAI_API_KEY/API_KEY/SGLANG_API_KEY/
VLLM_API_KEY environment. There is no new credential copy, server or retry
deadline. Use separate output files for `--new-tokens 0`, teacher-forced
reference samples and each checkpoint. Every replay saves the wire request,
raw response, all four scores, margins, deltas and winner changes. Input-ID
digests are provenance only; no SHA download/install/startup/acceptance gate
is added. The original `prompt_hash` is retained in its own field.

The earlier teacher-forced pilot scored 105/114; the vLLM one-forward pilot
103/114, with matching prompt hashes. They are distinct numerical methods.
Do not pool their reference scores or claim equality. First run this fixed
pilot on SGLang, inspect small-margin disagreements, then use one consistent
native protocol for all three SGLang checkpoints. Cross-engine differences
can also reflect kernels, reductions, cache state, chunking and batch shapes;
same inputs do not establish bitwise logits equivalence or model quality.

## Retrieved baseline and ready-to-replay pilot

On 2026-10-30 the existing Filer archives were read in this Workspace. The
saved samples reproduce 12,077/14,042 full-set answers (14,015 unique contexts),
103/114 one-forward pilot answers and 105/114 teacher-forced pilot answers.
The latter two have identical reconstructed contexts for all 114 prompt hashes;
the changed winners are college_computer_science doc 1 and public_relations
doc 1. `artifacts/sglang-native/baseline/baseline-review.json` retains their
actual four scores and margins. This is analysis of real saved vLLM results,
not a new SGLang score.

`native-eval-client-20261030.tar.gz` is available in the same existing Filer
evaluation directory. It includes this client, probes, performance instructions
and both frozen 114-context references. Token IDs were reconstructed with the
fixed official tokenizer and add_bos_token=false. The outer handoff now confirms
PreTrainedTokenizerFast, add_special_tokens=False/add_bos_token=false. Run:

```sh
python mmlu_native.py replay --prompts vllm-oneforward-pilot-frozen.jsonl \
  --native-url "$EXISTING_BACKEND_GENERATE_URL" --output sglang-official-pilot.jsonl
```

This consumes the saved context IDs directly and never regenerates few-shot
prompts. Keep the reference-teacher-forced file for separate comparisons.

## Local validation and field acceptance

Run the source/adapter tests with lm_eval 0.4.13, the official tokenizer files
at `artifacts/sglang-native/official-tokenizer` and CPU Torch:

```sh
CUDA_VISIBLE_DEVICES= python docker/dgx-spark-native/test_mmlu_native.py
python docker/dgx-spark-native/test_probe.py
```

They use synthetic logits and mocked transport; no model accuracy or GPU
runtime result is inferred. Field acceptance must additionally record:

1. Four imported image identities, native process imports, checkpoint revision,
   graph capture, Ready/restarts and actual Mesh NCCL libraries in process maps.
2. Business `probe.py` on the existing backend and gateway: default separate
   reasoning, explicit off/alias, tools, JSON schema, SSE finish/usage and native
   red-circle/blue-square vision. The image fixture follows the recipe's smoke.
3. The same saved three-needle near-1M request used for vLLM, with actual token
   count, all three answers, elapsed time, post-request health and no restarts.
4. Frozen MMLU pilot with raw probabilities and deltas, then full standard
   MMLU 5-shot and the same pinned speedbench on official weights. Keep
   thinking-on business checks separate from thinking-off speedbench.
5. Identical image/evaluation settings on the two derivative weight directories,
   preserving the official checkpoint for restoration. Tune performance only
   with separately recorded reproducible A/B results and the user's selection.
