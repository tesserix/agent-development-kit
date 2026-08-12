# Sandboxed code execution

Model-generated code is untrusted by construction. Whatever wrote it read tool output,
retrieved documents and user text, and any of those may have been supplied by someone who
wanted the agent to run something. Executing it in the agent's process hands an injected
prompt that process's credentials, its network position and its filesystem.

So it runs somewhere else:

```python
sandbox = SubprocessSandbox()
registry = ToolRegistry((sandbox_tool(sandbox),))
```

The agent calls `run_python(code=…)`; the code runs in a fresh interpreter, in a temporary
workspace, with an environment built from nothing.

## What the code does not get

| | |
|---|---|
| The network | `socket` is replaced before the code is compiled; a connection raises `OSError`. |
| The host's environment | The child's environment is constructed, not inherited, so no credential of the host's is in it. |
| The host's working directory | It starts in a workspace that is deleted when the call returns. |
| The kit itself | `-I -S` means no site packages and no inherited `sys.path`, so `import tesserix_adk` fails. |
| Unbounded time or memory | Ceilings are set on the child before any generated code runs. |

## Ceilings

```python
SandboxLimits(wall_seconds=10.0, cpu_seconds=5, memory_bytes=256 * 1024 * 1024)
```

Wall time and processor time are different diagnoses. Elapsed time catches code waiting for
something that will never arrive; processor time catches code spinning, and four threads
burn four processor seconds in one, so the two ceilings are not redundant. Hitting either
raises `SandboxTimeoutError`, whose `limit` says which.

`memory_bytes` becomes `RLIMIT_AS` on the child. Linux enforces it and an allocation past
it raises `SandboxMemoryError`. macOS refuses address-space ceilings outright, so there it
is an intent rather than a guarantee — the time ceilings still bound the run.

Output and artifacts are bounded too, because both are channels back into the conversation.
A stream longer than `max_output_chars` is cut and says so; a file larger than
`max_artifact_bytes` comes back capped with `truncated` set; more files than
`max_artifacts` keeps the first by name.

## What comes back

```python
result = await sandbox.run("open('out.csv', 'w').write('a,b')", files={"in.csv": "1,2"})
result.artifacts   # (SandboxArtifact(name='out.csv', content=b'a,b'),)
```

A non-zero exit is a *result*, not an error: the caller wanted to know what the code did,
and a traceback in `stderr` is what it did. Errors are reserved for the other case — the
sandbox took the process away, so there is nothing to report. That is why
`SandboxTimeoutError` and `SandboxMemoryError` are raised rather than encoded in a field:
a result object for a run that produced no result invites reading half an answer as a whole
one.

Input files are readable and are not returned as artifacts, since returning them would
charge the conversation twice for content it already had.

## How strong the boundary is

`SubprocessSandbox` is defence in depth inside one process tree, not a virtual machine.
Generated code that reaches for `ctypes` is one kernel boundary from the host, and that
boundary is the container the sandbox itself runs in — so run the agent in one, with a
seccomp profile and no egress, exactly as `docs/security.md` describes for every other
service.

`Sandbox` is the seam for the stronger case. A deployment that needs gVisor, Kata, a
microVM or a remote executor binds its own:

```python
class KataSandbox:
    async def run(self, code, *, limits=None, files=None) -> SandboxResult: ...
```

Everything above it — the tool, the registry, the run loop — is unchanged.

## In an agent

`sandbox_tool` is not parallel-safe by declaration: two runs in one workspace would see
each other's files. It carries its own `limits` so a tool exposed to a model can be more
careful than the backend it calls, and a ceiling that fires becomes a
`sandbox_limit_exceeded` refusal rather than a failure, because running the same code again
hits the same ceiling.
