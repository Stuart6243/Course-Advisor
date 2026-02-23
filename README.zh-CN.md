# Course Advisor AI（中文说明）

[English (Default)](./README.md)

本项目是一个本地优先（localhost）的 Columbia University 课程顾问系统。
用户用自然语言提问，系统完成「意图提取 -> 结构化检索 -> 基于课程数据生成回答」，并通过 SSE 流式返回。

## 项目概括

核心能力包括：
- 课程搜索、课程比较、课程推荐
- 多语言问答（`en` / `zh` / `es` / `fr`）
- PDF/HTML 课程文件导入（失败时进入手动录入）
- 对话导出（Markdown / JSON）
- 基于 `conversation_id` 的多轮记忆
- Groq 云模型优先 + Ollama 本地离线 fallback

项目规范基线来自 [`CLAUDE.md`](./CLAUDE.md)。

## 技术框架

- 前端：Vite + React 19 + TypeScript + Tailwind CSS v4
- 后端：Python FastAPI + Uvicorn
- 云端推理：Groq API
- 本地推理：Ollama

推理模式（`INFERENCE_MODE`）：
- `hybrid`（默认）：Groq 优先，不可用时自动回退 Ollama
- `groq`：只走 Groq
- `local`：只走 Ollama

## 系统逻辑（主链路）

```text
用户提问
  -> 多语言归一化
  -> 规则引擎意图提取（优先）
  -> LLM 意图 fallback（仅规则失败时）
  -> 结构化检索（enriched index + 课程详情）
  -> 按 conversation_id 拼接历史
  -> 防幻觉分支判断
  -> LLM 生成回答（SSE 流式）
  -> 返回 chunk + 更新对话历史
```

### 1) 意图提取策略

优先规则引擎（低延迟、稳定）：
- 课程代码、系别、教授、学分、时间段、星期、学期、比较目标等
- 模糊问题才走 LLM fallback

### 2) 对话历史管理

后端维护内存会话字典（进程重启后清空）：
- 以 `conversation_id` 作为 key
- 每个会话最多保留 `CONVERSATION_MAX_TURNS` 轮
- 总会话数超限按 LRU 淘汰（`CONVERSATION_MAX_SESSIONS`）

### 3) 防幻觉机制

响应生成强约束：
- 禁止百科式/通识性扩写
- 禁止编造课程、教师、时间、地点
- 当检索为空且无可用历史时，直接返回固定引导文案（不调用 LLM）
- follow-up 场景允许基于历史回答（即使本轮无新课程命中）

### 4) 导入校验与质量门禁

- `course_code` 严格正则校验（例如 `CIEN E3125`, `COMS W4111`）
- 非法 code（如 `GNIRPS 6202`）必须拒绝
- 低质量导入（`quality_score` 低于阈值）不直接入库，转手动修正流程

## 数据结构

数据目录：[`data/`](./data)

- `courses_flat/*.json`：单课程完整结构
- `courses_flat_index.json`：原始索引
- `courses_enriched_index.json`：扩充索引（检索主入口）

扩充索引包含检索增强字段：
- `department_prefix`
- `prerequisites_codes`
- `sections_summary`
- `all_instructors` / `all_terms`
- `searchable_text`

## API 说明

核心接口实现见 [`backend/server.py`](./backend/server.py)：

- `POST /api/chat`：SSE 流式聊天
  - 请求：`{ message, conversation_id, language }`
  - 事件：`chunk` / `sources` / `done` / `error`

- `POST /api/import`：上传 PDF/HTML 导入课程
  - 可能返回 `needs_manual_input=true`

- `POST /api/import/manual`：手动录入课程
  - 走同样的严格 code 校验

- `POST /api/export`：导出对话（markdown/json）
- `GET /api/health`：运行状态与模型可用性
- `GET /api/courses/stats`：课程总量/系别/学期统计

## 目录结构

```text
course-advisor/
├── README.md
├── README.zh-CN.md
├── CLAUDE.md
├── frontend/
│   ├── src/components/
│   ├── src/hooks/
│   ├── src/services/
│   ├── src/i18n/
│   └── src/types/
├── backend/
│   ├── server.py
│   ├── config.py
│   ├── query_parser.py
│   ├── course_retriever.py
│   ├── response_generator.py
│   ├── file_importer.py
│   └── tests/
└── data/
    ├── courses_flat/
    ├── courses_flat_index.json
    └── courses_enriched_index.json
```

## 本地运行

### 后端

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."
export INFERENCE_MODE="hybrid"   # hybrid | groq | local

uvicorn server:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

打开：`http://localhost:3000`

## 运维与排障要点

端口占用清理：

```bash
kill -9 $(lsof -t -i :8000) 2>/dev/null
kill -9 $(lsof -t -i :3000) 2>/dev/null
```

Ollama 状态检查：

```bash
ollama list
ollama ps
ollama stop qwen3-nothink
```

## 测试

```bash
cd backend
pytest -q
python tests/test_e2e.py
```

## 补充说明

- 课程总数取决于当前数据快照。
- 多轮历史为内存态，后端重启后会清空。
- `hybrid` 模式通常质量与速度最好；`local` 模式用于离线可用性。
