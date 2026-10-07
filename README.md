# RAG Knowledge Base

一个基于 LangChain + DeepSeek + ChromaDB 的工程化 RAG 知识问答系统。

## 🚀 技术栈
- Python 3.10+
- LangChain (文档加载、切分、检索)
- DeepSeek API (大模型问答)
- ChromaDB (向量数据库)
- HuggingFace Embeddings (BAAI/bge-large-zh-v1.5)

## 📁 项目结构
- `core/`：核心逻辑模块（配置、加载器、向量库管理、大模型调用）
- `docs/`：本地知识库文档（Markdown）
- `main.py`：项目入口

## ⚡ 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 API Key：在 `core/config.py` 填入你的 DeepSeek API Key
3. 运行：`python main.py`
## MCP 工具服务

本项目实现了 MCP (Model Context Protocol) 标准化工具服务：
- `mcp_server.py`：独立的 MCP 工具服务器（天气、音乐推荐）
- `agent_mcp.py`：通过 MCP 协议动态加载工具的 LangGraph Agent