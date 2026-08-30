# 旅行客服 Agent

配置：复制 `course.env.example` 为 `course.env`。只填自己的 Key，不要提交 `course.env`。

本机三个端口，互不占用：

| 端口 | 作用 |
|---|---|
| 8081 | 业务 mock（库存、行程） |
| 8000 | 本地 Qwen 分类 |
| 8002 | Agent HTTP（`/chat`） |

命令行对话用 `backend/chat.py`，不占端口。日常开 8081 + 8000 即可；网页调试再开 8002。不要运行默认的 `python backend/main.py`（它会抢 8000）。

## 1. 业务 mock — 8081

```powershell
cd mock_backend
python main.py
```

## 2. 本地分类 — 8000

```powershell
cd ../intent-finetune
python -m llamafactory.cli api train/api_qwen25_3b.yaml
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即可。浏览器打开 `/docs` 或 `/v1/models`，根路径 `/` 会 404。

## 3. Agent HTTP — 8002（可选）

```powershell
cd travel_agent/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

## 4. 命令行对话

```powershell
cd travel_agent/backend
python chat.py
```

空行或 `q` 结束。预订齐了回 `确认`；待审批时 `/approve` 或 `/reject`。本演示不执行真实出票或退票。

演进记录：`docs/发现问题到解决-完整路径.md`
