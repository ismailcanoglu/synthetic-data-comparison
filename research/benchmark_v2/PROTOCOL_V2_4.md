# Benchmark v2.4 — exploratory runner review

**Not a frozen confirmatory runner.** The included smoke configuration uses seed 9000, outside the proposed untouched confirmatory seed range 10000–10029.

## Run

```powershell
git fetch origin
git merge origin/benchmark-v2-4-runner-review
python -m pytest research/benchmark_v2/ -q
python research/benchmark_v2/benchmark_v2_4.py --config research/benchmark_v2/config_v2_4_smoke.json --output research/benchmark_v2/outputs_v2_4_smoke
```

The smoke test evaluates S1, baseline and CTGAN, LR with 100 CTGAN epochs, RF with 10 CTGAN epochs, seed 9000. Outputs are saved after each model/method unit; the run rejects an output directory with a different config fingerprint. Rerunning skips completed result pairs. Inspect `failures.csv` and `results_per_seed.csv`.

## Important limitations

- **Do not use proposed evaluation seeds 10000–10029 for smoke tests or tuning.**
- CTGAN epoch values in smoke config are **illustrative** and derived from prior exploratory work, not confirmatory choices.
- The runner does not yet produce quality-per-seed, split audit, package lockfile, or full failure policy; previously logged failures remain in `failures.csv` on retry, even if the pair later succeeds.
- Results are persisted by rewriting CSV after each unit, not transactionally; a hard interruption during write could corrupt output. For production use, add atomic writes and corruption recovery.
- This runner deliberately rejects `FROZEN_BEFORE_CONFIRMATORY_RUN` status.
- The existing v2.3 inference tool requires full five-method × five-scenario × thirty-seed coverage and cannot run on this smoke output.
- Before confirmatory run: independent development-only epoch selection, prospective freeze, full audit/quality outputs, predeclared missingness/failure policy, and additional integration tests.
