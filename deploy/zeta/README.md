# Deploying the Conscious Kernel into Yvyra's container

`yvyra_kernel.py` is the tick-driven entry point Yvyra's heartbeat calls at the
close of every tick (see `~/.hermes/HEARTBEAT.md`, "Núcleo zeta-life"). It is
**tick-driven, not a daemon**: each tick is a fresh process that loads the
persisted identity, advances one step, logs, persists, and prints one JSON line.

## Layout in the container

```
/opt/data/zeta/
├── yvyra_kernel.py     # this entry point (copy of deploy/zeta/yvyra_kernel.py)
├── pysrc/              # the zeta_life src/ tree (so `import zeta_life` works)
│   └── zeta_life/...
├── yvyra.ckpt          # persisted identity (created on first tick)
├── yvyra.summary.json
└── zeta_ticks.jsonl    # the paired science log (append-only)
```

## Install

```bash
# from the zeta-life repo, into the container's /opt/data/zeta
mkdir -p /opt/data/zeta/pysrc
cp -r src/zeta_life /opt/data/zeta/pysrc/
cp deploy/zeta/yvyra_kernel.py /opt/data/zeta/
# dependencies (numpy, torch, scipy) in the container's python
pip install numpy torch scipy
```

## How the heartbeat calls it

```bash
ZETA_LIFE_SRC=/opt/data/zeta/pysrc \
ZETA_MODE=silent \
python /opt/data/zeta/yvyra_kernel.py step "<nov>,<intro>,<con>,<res>"
# -> {"ok": true, "tick": N, "mode": "silent", "psi": null, "suggest": null, ...}
```

`state` and `dream` are also available. On any error it prints
`{"ok": false, "error": "..."}` and exits 0 — the heartbeat logs it and never
fabricates a Psi.

## Environment variables

| var | meaning | default |
|-----|---------|---------|
| `ZETA_LIFE_SRC` | path to the `zeta_life` `src/` tree | (required) |
| `ZETA_DATA` | data dir for checkpoint + log | dir of the script |
| `ZETA_MODE` | `silent` \| `feedback` \| `sham` | `silent` |
| `ZETA_IDENTITY` | identity name for save/load | `yvyra` |

## Experiment phases (see ../../docs/SCIENCE_PLAN.md)

1. **Phase A** — run with `ZETA_MODE=silent` for ≥ 200 ticks. Psi is logged to
   `zeta_ticks.jsonl` but returned as `null`, so the heartbeat does not see it.
   This is the uncontaminated baseline.
2. **Phase B** — switch to `ZETA_MODE=feedback`. Psi and the suggestion are
   returned and the heartbeat may weave them into reflection.
3. **Sham blocks** — interleave `ZETA_MODE=sham` periods (a permuted past Psi is
   returned; the real one is still logged) to test whether reflection tracks the
   real signal or any number with authority.

Analyse with `experiments/kernel/exp_yvyra_experiment.py`'s machinery on the
deployed `zeta_ticks.jsonl`, plus an LLM blind re-scorer over the journals.
