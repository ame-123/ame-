# 旅行客服路由模型微调

Agent 技术要点（主链路在 `travel_agent`，这里只训路由小模型）：

规则层 + Qwen 意图 → Context（行程号优先级、历史压缩）→ **规则白名单 RoutePlan（不是大模型选工具）** → 工具/RAG/HITL 拿事实 → DeepSeek 写话术 → 公开 Trace / Cost。

本目录：造数、**本地部署 Qwen 跑测试集**、LoRA SFT，再 DPO。不要用硅基流动刷测试集。不要全参微调。

Windows 解释器：

- 造数 / 读 travel_agent 规则：`C:\Users\admin\anaconda3\envs\langchain12\python.exe`
- 本地 GPU 推理与微调：`C:\Users\admin\anaconda3\envs\RWKV7\python.exe`（已有 CUDA Torch）

## 1. 数据

```bash
C:\Users\admin\anaconda3\envs\langchain12\python.exe -m src.generate_data
```

当前规模（`data/summary.json`）：train **497** / val **61** / test **258**（holdout 230）。DPO train **347** / val **38**。

SFT 的 `input` 是「当前句 + `[session_state]` + 最近 8 条 `[recent_dialogue]`」，与线上 `route_session_hint` 一致。其中 `ctx_hist_*` 是「当前指令很短、答案完全依赖历史」的对比样本（同一句「好的呢」：确认清单后是预订，打招呼/报完价后是闲聊）。

DPO 只保留 ctx 族或规则错分：chosen = 结合历史的正确 JSON，rejected = 只看当前句的规则意图，或粘住 `recent_intent` 的错意图。

测试 holdout 用大阪/杭州/厦门和不同说法，训练用东京/京都，`input` 不重叠。

## 2. 本地部署并跑测试集

```bash
C:\Users\admin\anaconda3\envs\RWKV7\python.exe -m pip install -r requirements-local.txt
set LOCAL_ROUTER_MODEL=Qwen/Qwen2.5-3B-Instruct
C:\Users\admin\anaconda3\envs\RWKV7\python.exe -u -m src.eval_channel --split test --backend local
```

首次从 ModelScope 拉约 6GB 权重。12GB 不能同时跑评测和 `llamafactory api`。

## 3. LoRA SFT，再 DPO

先 SFT（输出 `outputs/sft-qwen25-3b-ctx-lora`）：

```bash
C:\Users\admin\anaconda3\envs\RWKV7\python.exe -m llamafactory.cli train train/sft_qwen25_3b.yaml
```

云端 4090 用 `train/sft_qwen25_3b_4090.yaml`。不要改成全参。

SFT 复测：

```bash
set LOCAL_ROUTER_MODEL=Qwen/Qwen2.5-3B-Instruct
set LOCAL_ROUTER_ADAPTER=outputs/sft-qwen25-3b-ctx-lora
C:\Users\admin\anaconda3\envs\RWKV7\python.exe -u -m src.eval_channel --split test --backend local
```

再 DPO（接上 ctx SFT adapter，输出 `outputs/dpo-qwen25-3b-ctx-lora`）：

```bash
C:\Users\admin\anaconda3\envs\RWKV7\python.exe -m llamafactory.cli train train/dpo_qwen25_3b.yaml
```

4090 用 `train/dpo_qwen25_3b_4090.yaml`。DPO 复测把 `LOCAL_ROUTER_ADAPTER` 改成 `outputs/dpo-qwen25-3b-ctx-lora`。

接到旅行客服主链路：不要在 `langchain12` 里加载权重。用 `train/api_qwen25_3b.yaml` 起 OpenAI 兼容服务；DPO 训完后把 yaml 里的 `adapter_name_or_path` 改成 `outputs/dpo-qwen25-3b-ctx-lora`。步骤见 `travel_agent/docs/路由小模型微调与接入.md`。
