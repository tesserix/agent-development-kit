# Secrets

A credential in a config field is a credential in every rendering of that config: the
log line that dumped it for debugging, the span attribute somebody added for context, the
error payload returned to a caller, the `repr` in a traceback frame. None of those were
written by anyone intending to leak a key. That is exactly why it happens.

So configuration holds a `SecretRef` — a name and a version — and the value is fetched at
the point of use through a `SecretResolver`. A reference is safe to log, commit and diff.
The value arrives as a pydantic `SecretStr`, which redacts itself everywhere, and is
revealed on the one line that needs it.

```python
from tesserix_adk.core import EnvironmentSecrets, SecretRef

ref = SecretRef(name="openai-key", version="7")
print(ref.describe())  # openai-key@7 — no value anywhere in it

secrets = EnvironmentSecrets()
key = await secrets.resolve(ref)
client = OpenAI(api_key=key.get_secret_value())
```

## What the kit ships

| Resolver | Reads from | For |
|---|---|---|
| `EnvironmentSecrets` | `os.environ`, name folded to `OPENAI_KEY` | local development |
| `ProvidedSecrets` | the kit's synchronous `SecretProvider` | an existing lookup |
| `ChainedSecrets` | the first resolver that holds it | mixed deployments |
| `CachingSecrets` | an inner resolver, with a ttl | production |

`SecretResolver` is a protocol, so a deployment's own backend is a class with one method —
see [a resolver of your own](#a-resolver-of-your-own).

## Resolution fails loudly

A resolver raises `SecretResolutionError` rather than returning `None`. A caller that
cannot tell "there is no such secret" from "the backend is unreachable" carries on
unauthenticated, and an unauthenticated call is a failure somebody finds in a dashboard a
week later rather than in a stack trace immediately.

The refusal names the reference and never the value:

```
SecretResolutionError: openai-key@7 is not set in the environment (looked for OPENAI_KEY)
```

That message is safe to log, which is the point — a redaction rule nobody can safely
violate beats one everybody has to remember.

## One reference, every tenant

A `{tenant}` placeholder is filled by `for_tenant`, so one config line serves every tenant:

```python
ref = SecretRef(name="{tenant}-openai-key")
acme = ref.for_tenant("acme")   # acme-openai-key, bound to acme
```

A bound reference cannot be rebound. `acme.for_tenant("globex")` is refused, because a
reference that can be re-templated is one tenant's configuration reading another tenant's
secret — the same failure [`tenancy.md`](tenancy.md) refuses everywhere else, arriving
through the credential path instead of the data path.

Cache entries are keyed by tenant as well as by reference, so a hit for one tenant is
never served to another.

## Caching and rotation

```python
secrets = CachingSecrets(EnvironmentSecrets(), clock=clock, ttl_seconds=300)
```

- Nothing is served past its ttl. A backend that is down produces a refusal, not the last
  value that worked — serving a revoked credential to stay up is how a revocation fails to
  revoke anything.
- Concurrent callers asking for the same reference share one fetch, so a cold start does
  not turn every provider construction into its own round trip. Different references still
  resolve in parallel.
- `invalidate(ref)` and `invalidate_all()` pick up a rotation without waiting out the ttl
  and without a restart.
- A call already holding a revealed value completes with it. That is what the backend's own
  overlap window between old and new versions is for.

## The config linter

`literal_credentials(config)` returns the dotted paths of fields whose names look like
credentials and whose values are plain strings:

```python
>>> literal_credentials(settings)
('providers[1].api_key', 'postgres.password')
```

`SecretStr` passes, because it redacts itself. `SecretRef` passes, because it is not a
value. Run it at startup and refuse to boot on a finding — a config file gets committed,
pasted into a ticket and copied into a chat window, and a literal in it is a credential in
all three.

Field names matched, as substrings: `password`, `secret`, `token`, `api_key`, `apikey`,
`dsn`, `credential`.

## A resolver of your own

The kit does not ship a client for any cloud secret manager. Each one is a dependency with
a transitive tree of its own, and taking it would put that tree in every install of the
kit for the benefit of the deployments using that one vendor. The protocol is one method,
so the binding lives where the vendor choice already lives:

```python
from google.cloud import secretmanager
from pydantic import SecretStr
from tesserix_adk.core import SecretRef, SecretResolutionError

class GoogleSecrets:
    def __init__(self, project: str) -> None:
        self._project = project
        self._client = secretmanager.SecretManagerServiceAsyncClient()

    async def resolve(self, ref: SecretRef) -> SecretStr:
        path = (
            f"projects/{self._project}/secrets/{ref.name}"
            f"/versions/{ref.version or 'latest'}"
        )
        try:
            response = await self._client.access_secret_version(name=path)
        except Exception as failure:
            raise SecretResolutionError(
                f"{ref.describe()} could not be read", ref=ref.describe(), backend="gcp"
            ) from failure
        return SecretStr(response.payload.data.decode())
```

Wrap it in `CachingSecrets` and the access-count on the secret stops tracking the request
rate. The same shape works for Vault, AWS Secrets Manager and Kubernetes secrets: name,
version, one call, a typed refusal that names the reference.

## Known limitations

- A `SecretStr` holds its value in process memory in the clear once resolved. The kit does
  not lock pages or zero buffers; a core dump of the process contains the key.
- Rotation is picked up at the next resolution, not pushed. A ttl shorter than the
  backend's overlap window is what makes that safe.
- The linter matches field names, so a credential in a field called `value` is not caught.
  It narrows the common mistake rather than proving the absence of one.
- `EnvironmentSecrets` ignores versions. The environment has one value at a time, which is
  what makes it a development resolver.

## See also

- [`tenancy.md`](tenancy.md) — the tenant a reference is bound to
- [`agent-identity.md`](agent-identity.md) — who a run acts for, as opposed to what it can
  authenticate as
- [`tenant-config.md`](tenant-config.md) — where a per-tenant `SecretRef` is declared
