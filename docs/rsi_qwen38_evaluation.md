# Qwen3.8-Flash-Next evaluation on RSI

RSI generates predictions with vLLM in the evaluation process. It reads the 131
weight shards from `/fs2/tianfeng/models/Qwen3.8-Flash-Next` through a read-only
mount and does not start an HTTP service. The launcher uses the existing ROLL
image and a container without external networking. GPT and Opus credentials
remain in the Windows copy of `api_config.toml`.

After GPU allocation and run approval, launch from the deployed project on RSI:

```bash
bash scripts/run_rsi_predict.sh RUN_ID smoke
```

The smoke run requires all eight A100s on `rsi-N-1` and selects 12 SoMBench
questions, including a Q4. The `full` option runs all five datasets in the same
engine process and requires its own approved allocation. Use a new run ID.

The output is under `/fs2/chenxiaoyang23/tomeval-qwen38/runs/RUN_ID/`. Use SFTP
to download its `results` directory contents into `results/qwen38/rsi/` in the
Windows project. Preserve the `DATASET/qwen3.8-flash-next/exp_TIMESTAMP/`
hierarchy and note the common `exp_TIMESTAMP` shown by the remote run.

From the Windows project, grade the downloaded SoMBench predictions:

```powershell
.\.venv\Scripts\python.exe -X utf8 run_eval.py --experiment-config experiment_config_qwen38_rsi_metric.yaml --exp-dir exp_TIMESTAMP
```

For all five datasets, use `experiment_config_qwen38_rsi_metric_full.yaml`.
Only the SoMBench Q4 judges require API settings in this phase. Both judges must
return a valid rubric score before the Q4 metrics are written.
