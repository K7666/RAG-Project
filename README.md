# 🤖 生产级多智能体协作系统 (Production-Grade Multi-Agent API)

本项目是一个从零构建的、具备完整工程化能力的 AI 智能体（Agent）服务。项目经历了从本地脚本 -> 模块化重构 -> MCP 工具解耦 -> 多智能体协作 -> FastAPI 生产级服务化的完整演进。

## 🏗️ 系统架构
- **前端交互层**：支持 HTTP POST 请求，原生支持 SSE (Server-Sent Events) 流式输出。
- **服务端 (FastAPI)**：异步高并发处理，挂载 LangChain Callback 黑匣子进行可观测性监控。
- **多智能体系统 (LangGraph Supervisor)**：
  - 主管 Agent（Supervisor）：负责意图识别与任务分派。
  - 天气专家 Agent：负责天气查询。
  - 知识库专家 Agent：基于 RAG 检索本地文档。
  - 音乐专家 Agent：负责歌曲推荐。
- **底层工具链 (MCP)**：采用 Model Context Protocol 标准协议，彻底解耦业务工具（天气/音乐/RAG）与 Agent 核心代码。
- **记忆持久化**：SQLite 持久化多轮会话 + ChromaDB 向量记忆库。

## ✨ 核心特性
1. **标准化 MCP 协议**：工具独立运行于 Server，Agent 通过 stdio 动态发现并调用，实现即插即用。
2. **流式输出 (SSE)**：基于 LangChain `astream_events` 实现 Token 级流式响应，提供打字机般的用户体验。
3. **全链路可观测性**：自定义 `BaseCallbackHandler`，精准追踪大模型思考、工具调用耗时与 Token 消耗。
4. **提示词注入防御**：实现安全隔离机制，成功阻断恶意工具返回数据中的 Prompt Injection 攻击。
5. **三层记忆系统**：短期 Session 记忆 + 长期向量事实记忆 + 本地 RAG 知识库检索。

## 🚀 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量：在 `core/config.py` 填入你的 DeepSeek API Key。
3. 启动 MCP 工具服务：`python mcp_server.py`
4. 启动 API 服务：`python api_server.py`
5. 发送请求：`POST http://localhost:8000/chat`
