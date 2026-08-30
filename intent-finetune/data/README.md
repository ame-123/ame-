# 数据（不入库）

`train.jsonl` / `val.jsonl` / `test.jsonl` / `dpo_*.jsonl` 只留在本地，不推 GitHub。

本地生成：

```bash
python -m src.generate_data
```

规模与划分见同目录 `summary.json`。字段约定见 `dataset_info.json`。权重与 `outputs/` 也不入库。
