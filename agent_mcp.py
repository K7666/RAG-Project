# ==========================================
# 1. 准备工作：导入工具和模型
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from core.config import settings
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.memory import MemorySaver   #MemorySaver 是 LangGraph 内置的内存检查点工具。它就像一个“内存账本管理器”，专门帮你把每个对话的账本快照保存下来。
memory = MemorySaver()
import asyncio


async def main():              #这部分需要放在 async 函数里，因为 MCP 连接是异步的
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
    llm_with_tools = llm.bind_tools(tools)

    # 4. 构建图和记忆（这一部分和你之前手写的一样）
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver

    def agent_node(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("agent", tools_condition)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # 5. 运行测试
    config = {"configurable": {"thread_id": "mcp_test"}}
    print("\n--- 第一轮 ---")
    result = await app.ainvoke({"messages": [("user", "北京今天天气怎么样？")]}, config)
    print(f"AI：{result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())