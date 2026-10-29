#!/usr/bin/env python3
"""Record business API evidence after the owner hands over the GPU window."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="OpenAI base URL, including /v1"
    )
    parser.add_argument("--model", default="local-large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    secret = (
        os.environ.get("API_KEY")
        or os.environ.get("SGLANG_API_KEY")
        or os.environ.get("VLLM_API_KEY")
    )
    if secret:
        headers["Authorization"] = "Bearer " + secret
    evidence = {
        "base_url": args.base_url,
        "model": args.model,
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
        "tests": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def chat(label, **fields):
        payload = dict(model=args.model, temperature=0, max_tokens=2048, **fields)
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        started = time.monotonic()
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
        evidence["tests"].append(
            {
                "name": label,
                "request": payload,
                "response": result,
                "seconds": time.monotonic() - started,
            }
        )
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
        )
        return result["choices"][0]["message"]

    arithmetic = [dict(role="user", content="What is 19 + 23? Reply with the number.")]
    default = chat("default thinking", messages=arithmetic)
    assert default.get("reasoning_content"), (
        "Omitted thinking must return a separate reasoning field"
    )
    assert "42" in (default.get("content") or ""), default
    off = chat(
        "thinking explicitly off",
        messages=arithmetic,
        chat_template_kwargs={"thinking": False},
    )
    assert not off.get("reasoning_content") and "42" in (off.get("content") or ""), off
    alias = chat(
        "vLLM thinking alias off",
        messages=arithmetic,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert not alias.get("reasoning_content") and "42" in (
        alias.get("content") or ""
    ), alias
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_fixture",
            "description": "Look up a fixture value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }
    messages = [
        dict(
            role="user",
            content="Use lookup_fixture to get the value for key alpha. Do not guess.",
        )
    ]
    called = chat(
        "tool call",
        messages=messages,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    calls = called.get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_fixture", called
    assert json.loads(calls[0]["function"]["arguments"]) == {"key": "alpha"}, called
    messages += [
        called,
        dict(role="tool", tool_call_id=calls[0]["id"], content='{"value":42}'),
    ]
    answer = chat(
        "tool continuation",
        messages=messages,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    assert "42" in (answer.get("content") or ""), answer
    evidence["passed"] = True
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(
        f"PASS: default thinking, explicit off, alias and tool round trip; {args.output}"
    )


if __name__ == "__main__":
    main()
