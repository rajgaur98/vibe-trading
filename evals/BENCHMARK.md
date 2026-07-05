# Model Benchmark

_Not yet generated._ This table is produced by running the golden-set eval across
several models (requires the relevant provider API keys):

    EVAL_JUDGE_MODEL=gemini-3.1-flash-lite \
    python -m vibe_trading.eval.benchmark \
      --models "gemini/gemini-3.1-flash-lite,gemini/gemma-4-31b-it" \
      --throttle-seconds 4.5

Running it regenerates this file with a cost-vs-quality table sorted by
score-per-dollar, and writes a timestamped JSON report under `data/reports/`.
The committed regression baseline (`evals/baseline.json`) is never touched by a
benchmark run.
