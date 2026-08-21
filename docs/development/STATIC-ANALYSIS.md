# Static analysis scope

Three things run bandit against this repository, and they do **not** agree on
what to scan. That is a deliberate, recorded decision rather than an oversight,
because it has already produced one false-positive issue and will produce more.

## Who scans what

| Runner | Scope | Where the scope is set |
| --- | --- | --- |
| `make security` | `src` only | `.rhiza/make.d/python.mk` — scans `${SOURCE_FOLDER}` |
| pre-commit hook | everything **except** `tests`, `.rhiza/tests`, `.venv` | `.pre-commit-config.yaml:71`, in the hook's `--exclude` argument |
| CodeFactor | **everything, `tests` included** | nothing — see below |

The first two are what CI enforces. The third is what comments on pull requests.

## Why CodeFactor sees more

CodeFactor prefers a repository's own tool configuration over its defaults, and
for bandit the file it looks for is `.bandit`. This repository has one:

```ini
[bandit]
skips = B101
```

Two consequences follow, and both are easy to miss.

**The scope is not in that file.** `.bandit` carries `skips` and nothing else.
The exclusion of `tests` lives in the *pre-commit hook's arguments*, not in the
config, so every bandit runner that is not pre-commit — CodeFactor, an IDE
plugin, a contributor typing `bandit -r .` — reads `.bandit`, finds no
`exclude`, and scans the test suite.

**Adopting `.bandit` also discards CodeFactor's own skip list.** CodeFactor's
[default `.bandit.yml`](https://github.com/codefactor-io/default-configs/blob/master/.bandit.yml)
skips 26 checks, `B311` among them. Ours skips exactly one. So pointing
CodeFactor at our config does not narrow its rule-set, it *widens* it —
`B311` is off by default at CodeFactor and on here.

Together those explain [#29](https://github.com/Jebel-Quant/quadprog/issues/29):
ten `B311` reports in `tests/test_against_c.py`, invisible to every local gate.

## `B311` in the test suite is a known false positive

`B311` is "standard pseudo-random generators are not suitable for
security/cryptographic purposes". It matches on the **call name**
`random.randint`. The tests seed `np.random.RandomState` for reproducibility,
and while the local holding it was named `random`, bandit resolved
`random.randint` to the stdlib function.

The tell is that `randn` and `rand` sit on the same object and were never
flagged — they are not on the blacklist. It is name matching, not analysis.

That particular instance is fixed: the locals are now named `rng`
(see [#29](https://github.com/Jebel-Quant/quadprog/issues/29)). Nothing in this
repository has a cryptographic context, so any future `B311` report against
`tests/` is the same false positive and should be treated as one.

## What to do about it

**Nothing, by default.** The wider scope is accepted rather than suppressed. It
costs an occasional false positive; in exchange, test code gets static analysis
that the local gates deliberately skip, and test code is still code. The
alternative — silencing it — also silences findings nobody is currently looking
for.

Two things are worth knowing if that trade stops paying:

- **To silence it per-repository**, add an ignore pattern in the CodeFactor web
  UI under *Settings → Ignore Files* (`tests/*`). There is no committed
  `.codefactor.yml`; CodeFactor has no such file, and exclusions are a UI
  setting only. Anything done there is invisible to this repository, which is
  precisely why this page exists.
- **The durable fix is upstream.** `.bandit` is template-owned
  (`.rhiza/template.lock:12`), so editing it here is reverted by the next
  `/rhiza:update`. The scope belongs *in* `.bandit` as an `exclude` key rather
  than in the pre-commit hook's arguments, so that every runner agrees. Reported
  as [jebel-quant/rhiza#1493](https://github.com/jebel-quant/rhiza/issues/1493).

## Why this is written down at all

A check that reports things nobody can act on teaches people to ignore the
check. This repository already makes that argument about itself — see
[Mutation testing](MUTATION.md#why-this-is-not-in-ci-at-all), where the
template's mutation gate is left disabled because fourteen provably-equivalent
mutants mean it could never go green, and a permanently red check is worse than
no check.

The same reasoning applies to a scanner whose scope silently differs from CI's.
Recording the difference is what keeps it a known quantity instead of a
recurring surprise.
