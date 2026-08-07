# CPU inference

CPU is the default target. An operator with no GPU budget still has to run agents, and
llama.cpp with GGUF quantization is the mature path for that. It serves OpenAI's wire
format through `llama-server`, so it arrives through the same adapter as vLLM: an agent
does not become a different agent because the weights moved onto a CPU.

What the kit owns is the part around the server — the flags it is started with, the prompt
cache each request asks for, and whether the model fits in the machine at all.

```python
from tesserix_adk.models.gguf import GgufModel
from tesserix_adk.models.providers import LlamaCppProvider, LlamaCppTuning

provider = LlamaCppProvider(
    "llama-3.1-8b-instruct",
    base_url="http://127.0.0.1:8080",
    capabilities=serves,
    tuning=LlamaCppTuning(threads=8, batch_size=512, context_tokens=8_192),
    weights=GgufModel(name="llama-3.1-8b-instruct", parameters_b=8.03),
    available_bytes=16 * 1024**3,
)
```

`weights` and `available_bytes` together are what turn an OOM kill into a typed refusal.
Given neither, nothing is checked — the kit does not guess a machine's memory.

## Fitting a model in a machine

Three things occupy memory, and they move for different reasons.

| Part | Size | Moves with |
|---|---|---|
| Weights | parameters × bits per weight ÷ 8 | The quantization |
| KV cache | `kv_bytes_per_token` × `context_tokens` | The context the *server was started with*, used or not |
| Buffers | ~512 MiB | Roughly constant across model sizes on the CPU backend |

`GgufModel.footprint(context_tokens=…)` returns those three as a `MemoryEstimate`.
`refuse_if_it_will_not_fit` compares the total against the memory there is and raises
`ModelTooLargeError` — a `ConfigurationError`, so anything already handling assembly
failures catches it — naming the shortfall and, where there is one, a lighter quantization
that would have fitted:

```
llama-3.1-8b-instruct at q5_k_m with a 32768-token context needs 9.80 GiB (5.30 GiB of
weights, 4.00 GiB of KV cache, 0.50 GiB of buffers) and this machine has 8.00 GiB:
q2_k would fit
```

The KV cost per token is a field rather than a constant because grouped-query attention
changes it by an order of magnitude between two models of the same size. It is
`2 × layers × kv_heads × head_dim × 2` bytes at f16 — 131,072 for an 8B Llama, 36,864 for
Qwen2.5 3B. Leaving the default on a model without grouped-query attention under-counts
badly, which is exactly the case where the machine then OOMs.

## Quantization

Fewer bits per weight means a smaller file and less memory bandwidth per token, which on
CPU is the entire performance budget — and worse output. Q4_K_M is where the published
trade-off sits, so it is `DEFAULT_QUANTIZATION`, and `quantization_for(...)` never returns
anything heavier: past that point more bits buys little quality and costs the one resource
there is none of.

| Format | Bits per weight | Where it is the right answer |
|---|---|---|
| `q2_k` | 3.35 | Nothing else fits, and the quality cost is visible |
| `q3_k_m` | 3.91 | A tight machine, one step down from the default |
| `q4_0` | 4.55 | Legacy; prefer `q4_k_m` at nearly the same size |
| `q4_k_m` | 4.83 | **The default.** Where quality holds |
| `q5_k_m` | 5.67 | Quality-sensitive work with memory to spare |
| `q6_k` | 6.56 | Near-lossless, at a bandwidth cost |
| `q8_0` | 8.5 | Reference comparisons |
| `f16` | 16.0 | Measuring what quantization cost |

```python
quantization_for(8.03, context_tokens=4_096, available_bytes=5 * 1024**3)
# Quantization.Q3_K_M — the heaviest that fits, never heavier than the default
```

## A CPU-viable model set

Computed with `GgufModel.footprint` at `q4_k_m`, using each architecture's own KV cost.
These are the numbers the kit will check against, not a benchmark — see below for
measuring throughput on the hardware you actually have.

| Model | Params (B) | KV bytes/token | Weights | Total @ 4k | Total @ 8k |
|---|---|---|---|---|---|
| `qwen2.5-3b-instruct` | 3.09 | 36,864 | 1.74 GiB | 2.38 GiB | 2.52 GiB |
| `llama-3.2-3b-instruct` | 3.21 | 114,688 | 1.80 GiB | 2.74 GiB | 3.18 GiB |
| `mistral-7b-instruct-v0.3` | 7.25 | 131,072 | 4.08 GiB | 5.08 GiB | 5.58 GiB |
| `llama-3.1-8b-instruct` | 8.03 | 131,072 | 4.52 GiB | 5.52 GiB | 6.02 GiB |
| `qwen2.5-14b-instruct` | 14.77 | 196,608 | 8.30 GiB | 9.55 GiB | 10.30 GiB |

An 8B at `q4_k_m` is the size that fits a 16 GiB machine with room for the rest of the
container, which is why it is the reference model everywhere in these docs.

## Measuring throughput on your own hardware

Tokens per second on CPU is a property of the machine, not of the model: it tracks memory
bandwidth almost linearly, so a number measured on one box says nothing useful about
another. The kit therefore ships no throughput table. Measure yours:

```bash
llama-bench -m ./llama-3.1-8b-instruct-q4_k_m.gguf -p 512 -n 128 -t 8 -r 5
```

`-p` is prompt ingestion (the first seconds of a run), `-n` is generation (everything
after), `-t` is threads, `-r` is repetitions. Report both: they scale differently, and a
box that ingests quickly can still generate too slowly to be worth using.

Then sweep `-t` across physical core counts and take the best, rather than assuming more
is better — see below.

## Tuning

`LlamaCppTuning` is how the server was started, in the form both the operator and the kit
read. It exists so the fit check and the launch command cannot disagree about the context
length. `server_arguments()` renders the flags:

```python
LlamaCppTuning(threads=8, batch_size=512, context_tokens=8_192).server_arguments()
# ('--threads', '8', '--batch-size', '512', '--ctx-size', '8192')
```

| Field | Flag | What it is for |
|---|---|---|
| `threads` | `--threads` | Generation threads. **Physical** cores, not hyperthreads |
| `batch_size` | `--batch-size` | Prompt tokens per pass; fills memory bandwidth on ingestion |
| `micro_batch_size` | `--ubatch-size` | The physical batch inside a logical one |
| `context_tokens` | `--ctx-size` | What the KV cache is sized from, and what the fit is checked against |
| `prompt_cache` | — | Per-request, not a flag: see below |

A field left `None` renders no flag, leaving llama.cpp's own default — rendering a guess at
a thread count is worse than rendering nothing. A `batch_size` below its `micro_batch_size`
is refused at construction, because llama.cpp silently reads it as a batch of the larger.

Oversubscribing threads makes a CPU backend slower, reliably and counter-intuitively: the
threads contend for the same memory bandwidth they are all waiting on. Start at physical
cores, then measure downwards.

## The prompt cache

llama.cpp does not keep the evaluated prefix between turns unless each request asks it to,
and without it every turn re-evaluates the whole prompt — on CPU that is the run. The
provider sends `cache_prompt: true` by default; `LlamaCppTuning(prompt_cache=False)` turns
it off for a server sharing one slot between unrelated callers, where the cache thrashes.

The cache hits on an exact byte prefix. Anything that reorders a system message, restamps a
timestamp into it, or re-serialises a tool schema differently between turns is a cache that
never hits, which is why prompt assembly is byte-stable and has a fingerprint test guarding
it. If you add to the prompt, add at the end.

## Running the same agent elsewhere

The CPU path is worth having only if it is not a fork. An agent written against vLLM runs
here unchanged — same `Agent`, same `AgentRunner`, same structured output, same usage
accounting — with only the provider swapped. `examples/cpu_inference.py` runs one agent
against both and asserts the answers match, over a stub transport, with no network.

## Known limitations

- **No throughput numbers.** As above: they are a property of the box. `llama-bench` is the
  answer, not a table in a repository.
- **The footprint is an estimate.** Buffers are approximated as a constant, and a build with
  different BLAS backends or a flash-attention kernel will differ. It is deliberately not
  optimistic, but leave headroom.
- **No GPU offload tuning.** `--n-gpu-layers` and partial offload are a separate story.
- **Weights are not shipped or verified here.** The kit describes a model; obtaining and
  checksumming the GGUF file is the operator's, and the supply chain's, business.
- **Capabilities are declared, not probed.** As for any self-hosted endpoint, what the build
  and the model can do is stated by whoever deployed it.
