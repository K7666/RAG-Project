# 1. 准备工作：导入工具和模型
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from core.config import settings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver #SqliteSaver 是同步的，它不支持异步方法 aget_tuple()。LangGraph 在异步模式下尝试调用它，就会报错。
import asyncio
from langchain_community.vectorstores import Chroma    #从 LangChain 社区包里，拿出 Chroma 这个向量数据库类
from langchain_huggingface import HuggingFaceEmbeddings  #拿出 Embedding 模型类，用来把文字变成向量。和第一周用的是同一个模型（BAAI/bge-large-zh-v1.5）。
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"      #设置国内镜像，防止下载模型时连接超时。

# 初始化记忆向量库（跟你的知识库 chroma_db 分开存，避免混淆）
memory_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")   #实例化一个 Embedding 模型对象，赋给 memory_embeddings。
memory_store = Chroma(
    persist_directory="./memory_db",           #persist_directory="./memory_db"：指定数据库存到 ./memory_db 文件夹。
    embedding_function=memory_embeddings       #告诉数据库用哪个模型向量化。
)

async def main():              #这部分需要放在 async 函数里，因为 MCP 连接是异步的
    # 在 main() 开头初始化异步记忆管理器
    async with AsyncSqliteSaver.from_conn_string("memory.db") as memory:  #AsyncSqliteSaver 不能像 SqliteSaver 那样直接 sqlite3.connect 创建连接，它需要在异步上下文里初始化。所以我们要把它放在 main() 函数的开头

        # 1. 配置 MCP 客户端，告诉它去哪里找你的服务器
        client = MultiServerMCPClient(     #MultiServerMCPClient：这是 LangChain 专门为 MCP 协议提供的一个“大管家”类
            {
                "my_life_tools": {        #（最外层的键）这是你给这个服务器起的代号。你可以叫它 "life_server"、"weather_bot" 等等。它的作用：如果以后你连了多个服务器，可以用这个代号来区分谁是谁。
                    "command": r"D:\pycharm\RAG_Project\venv\Scripts\python.exe",  #告诉电脑：“用 python 这个程序去启动它。”意思是“运行 Python 解释器”。
                    "args": ["mcp_server.py"],  #告诉 Python 解释器：“去执行 mcp_server.py 这个文件。”
                    "transport": "stdio",   #（沟通方式）意思是：“用标准输入输出流（stdio）来和它交流。”
                }
            }
        )

        # 2. 自动从 MCP 服务器加载工具！
        tools = await client.get_tools()
        print(f"✅ 成功加载了 {len(tools)} 个工具：{[t.name for t in tools]}")
        # 3. 绑定工具给大模型
        llm = ChatOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )

        # ==========================================
        # 创建三个专家 Agent
        # ==========================================
        from langgraph.prebuilt import create_react_agent

        # 1. 天气专家：只负责天气
        weather_agent = create_react_agent(     #create_react_agent(llm, tools, prompt)：LangGraph 提供的一键创建 Agent 方法。
            llm,
            tools=[t for t in tools if t.name == "get_weather"],
            prompt="你是一个天气专家。用户问天气时，调用 get_weather 工具。",
            name = "weather_agent",       #weather_agent 这只是Python 变量名,对图来说是不可见的，图只知道 name 参数里注册的名字。
        )

        # 2. 知识库专家：只负责 RAG 检索
        knowledge_agent = create_react_agent(
            llm,
            tools=[t for t in tools if t.name == "search_knowledge_base"],
            prompt="你是一个技术知识专家。用户问 RAG、大模型等概念时，调用 search_knowledge_base 工具。",
            name = "knowledge_agent",
        )

        # 3. 音乐专家：只负责推荐歌曲
        music_agent = create_react_agent(
            llm,
            tools=[t for t in tools if t.name == "get_music"],
            prompt="你是一个音乐推荐专家。用户让你推荐歌曲时，必须调用 get_music 工具，禁止凭记忆推荐歌曲。",
            name = "music_agent",
        )
        # ==========================================
        # 创建主管 Agent（Supervisor）
        # ==========================================
        from langgraph_supervisor import create_supervisor    #create_supervisor：LangGraph 官方提供的 Supervisor 模式封装。

        supervisor = create_supervisor(
            agents=[weather_agent, knowledge_agent, music_agent],   #agents=[...]：把三个专家注册进去。
            model=llm,
            prompt=(                  # prompt：主管的“工作手册”，告诉它什么任务该派给谁。
                 "你是一个主管。你的任务是根据用户的问题，决定交给哪个专家处理。\n"
                 "【强制规则】\n"
                 "1. 天气问题 → 必须交给 weather_agent。\n"
                 "2. RAG 概念 → 必须交给 knowledge_agent。\n"
                 "3. 推荐歌曲 → 必须交给 music_agent。\n"
                 "4. 【最重要】如果用户的问题涉及多个方面（比如'天气+音乐'），"
                 "你必须依次派出所有相关专家，收集完所有结果后才能生成最终回答。"
                 "禁止在只派了一个专家后就自己回答。\n"
                 "5. 你只能汇总专家返回的结果，不能自己编造内容。"
            )
        )
        app = supervisor.compile(checkpointer=memory)
        # 5. 运行测试
        # 运行测试
        config = {"configurable": {"thread_id": "multi_agent_test"}}

        # 测试 1：单专家任务
        print("\n--- 测试 1：天气 ---")
        result = await app.ainvoke({"messages": [("user", "上海今天天气怎么样？")]}, config)
        print(f"AI：{result['messages'][-1].content}")

        # 测试 2：跨专家任务（天气 + 音乐）
        print("\n--- 测试 2：天气 + 音乐 ---")
        result = await app.ainvoke({"messages": [("user", "上海天气怎么样？根据天气推荐首歌。")]}, config)
        print(f"AI：{result['messages'][-1].content}")

        # 测试 3：知识库任务
        print("\n--- 测试 3：RAG 概念 ---")
        result = await app.ainvoke({"messages": [("user", "什么是大模型？")]}, config)
        print(f"AI：{result['messages'][-1].content}")
        # async def extract_and_store_memory(user_input: str, ai_response: str):      #async def：定义一个异步函数（因为它在 agent_mcp.py 的异步环境里调用）。
        #     prompt = (
        #     "请从以下对话中提取用户的关键信息（如偏好、身份、地点等）。\n"
        #     "要求：\n"
        #     "1. 只提取跨会话仍然成立的事实，不要提取临时问题或意图。\n"
        #     "2. 每条信息单独一行，不要编号、不要加粗、不要任何格式符号。\n"
        #     "3. 用第三人称陈述句，例如'用户是研一学生'，而不是'我想...'。\n"
        #     "4. 如果没有新信息，只回复一个'无'字。\n"
        #     f"\n用户：{user_input}\nAI：{ai_response}"
        #     )
        #     extracted = (await llm.ainvoke(prompt)).content     #此处调用大模型不会将prompt和extracted作为对话传进SqliteSaver，因为 SqliteSaver 只保存 LangGraph 图内部 State 的变化，而 llm.ainvoke(prompt) 是完全绕过图，直接调用大模型。SqliteSaver 只认 LangGraph 图里的 State 变化。图外面发生的事，它完全看不见。
        #     print(f"提取到的关键信息：{extracted}")
        #     if extracted.strip() != "无":   #extracted.strip()：去掉前后空格。如果结果不是“无”，说明提取到了新信息。
        #         facts = [line.strip() for line in extracted.split("\n") if line.strip() and not line.strip().startswith("根据")]   #extracted.split("\n")：按换行符切分成字符串列表。line.strip():去空格，得到干净的一行。if line.strip():过滤空行。and not line.strip().startswith("根据"):过滤掉以"根据"开头的解释句。结果：["用户是研一学生", "用户在北京", "用户喜欢晴天"]
        #         # 去掉编号和 Markdown 符号
        #         facts = [
        #             line.lstrip("0123456789.-*# ").replace("**", "").strip()   #lstrip 是"从左开始剥掉所有出现在给定字符集合里的字符"。.replace("**", ""):删除所有 **（Markdown 加粗符号）。.strip():再次去首尾空格，因为 lstrip 之后可能留下尾部空格。
        #             for line in facts
        #         ]
        #         if facts:   #如果列表非空才存。如果全是空行或全被过滤掉了，facts 是 []，跳过存储。
        #             memory_store.add_texts(facts)   #memory_store.add_texts(facts)：把这 3 条事实逐条向量化，存入 memory_db。下次检索时，这些记忆就能被找到。
        #         print(f"✅ 已存入 {len(facts)} 条新记忆：{facts}")
        #
        # await extract_and_store_memory(user_input, ai_response)   #在这里调用记忆提取函数！把用户说的话和AI的回答传进去
if __name__ == "__main__":
    asyncio.run(main())
# 异步不是“同时用多个 CPU 核心计算”，它只是在一个线程里，通过切换任务来避免无意义的等待。
# 异步的本质是在 I/O 等待期间释放 CPU，让事件循环去处理其他任务。
# 它不增加计算能力，而是提高 CPU 利用率。对于 I/O 密集型场景，异步能把总等待时间从串行的 N×T 降低到接近 max(T)，所以并发请求能几乎同时返回。


#多智能体协作：
#（1）Supervisor（主管模式）：一个主管 Agent 负责任务分派，多个下属 Agent 负责执行。适合：任务可拆分、有明确分工
#（2）Network（网状模式）：所有 Agent 平等，可以互相调用。适合：任务复杂、需要自由协作

# Supervisor 模式的局限:
#“我用 Supervisor 模式实现了多智能体协作，但在测试中发现主管 Agent 在多意图任务中会‘越权回答’，绕过专家直接生成结果。
# 我分析了原因，发现这是 Supervisor 模式的固有倾向——主管拿到部分结果后容易自己总结。
# 后来我通过加强 prompt 约束和调整分派逻辑来缓解这个问题。如果要彻底解决，需要改用手写 StateGraph 的 Network 模式，让专家之间可以互相调用。”