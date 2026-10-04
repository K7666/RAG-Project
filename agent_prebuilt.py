import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from core.config import settings
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建一个 SQLite 连接对象（连接到一个叫 memory.db 的文件）
# 如果文件不存在，会自动创建
conn = sqlite3.connect("memory.db", check_same_thread=False)  #sqlite3.connect("memory.db", ...)：打开/创建 memory.db 这个数据库文件。check_same_thread=False：允许数据库连接跨线程使用。因为 Streamlit 或某些框架可能会用多线程，加了这个参数更保险。

# 使用 SqliteSaver 作为检查点，它会自动把账本写入数据库
memory = SqliteSaver(conn)                         #SqliteSaver(conn)：把数据库连接包装成一个“检查点保存器”。



# 定义工具（和之前一样）
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。"""
    if city == "北京":
        return "北京今天晴朗，气温 25 度。"
    elif city == "上海":
        return "上海今天有小雨，气温 22 度。"
    return f"抱歉，没有找到{city}的天气数据。"

@tool
def get_exchange_rate(currency: str) -> str:
    """查询指定货币对人民币的汇率。参数 currency 请传入中文（例如：美元、欧元）"""
    if currency == "美元":
        return "1美元 = 7.2人民币"
    elif currency == "欧元":
        return "1欧元 = 7.8人民币"
    return f"抱歉，没有找到{currency}的汇率数据。"
@tool
def get_music(mood: str) -> str:
    """根据天气推荐一首歌。"""
    if "雨天" in mood or "雨" in mood:
        return "推荐歌曲：周杰伦《晴天》—— 适合在下雨天怀念阳光。"
    elif "晴朗" in mood or "户外" in mood:
        return "推荐歌曲：五月天《倔强》—— 适合阳光明媚的日子去郊游。"
    else:
        return "推荐歌曲：陈奕迅《稳稳的幸福》—— 适合任何时候听。"
@tool
def mock_web_search(query: str) -> str:
    """模拟网页搜索工具，输入搜索关键词，返回相关的网页摘要。"""
    if "RAG" in query or "检索增强" in query:
        return (
            "网页1：RAG是一种结合了检索和生成的技术，通过外部知识库增强大模型回答的准确性。\n"
            "网页2：RAG的核心流程包括：文档切分、向量化、检索、生成。\n"
            "网页3：相比微调，RAG更容易更新知识，成本更低，但依赖高质量的检索系统。"
        )
    elif "Agent" in query or "智能体" in query:
        return (
            "网页1：AI Agent是一种能自主感知环境、做出决策并执行行动的智能系统。\n"
            "网页2：Agent的核心能力包括：规划、记忆、工具使用。\n"
            "网页3：LangGraph 是构建 Agent 的主流框架，支持多智能体协作。"
        )
    return f"没有找到关于 '{query}' 的搜索结果。"

# 初始化大模型并绑定工具
llm = ChatOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)
tools = [get_weather, get_exchange_rate,get_music,mock_web_search]
llm_with_tools = llm.bind_tools(tools)
# ==========================================
# 2. 全自动定义图
from langchain.agents import create_agent
app = create_agent(llm_with_tools, tools, checkpointer=memory)
# 3. 运行测试
if __name__ == "__main__":
    print("=== 全自动 LangGraph Agent ===")
    config = {"configurable": {"thread_id": "user_1"}}
    # 第一轮对话
    print("\n--- 第一轮 ---")
    user_input_1 = "那北京呢"
    print(f"用户：{user_input_1}")
    result = app.invoke({"messages": [("user", user_input_1)]}, config)  #当你第一轮调用 app.invoke({"messages": [("user", user_input_1)]}, config) 时：LangGraph 去 MemorySaver/SqliteSaver 里找 thread_id="user_1" 的账本发现没找到（因为是第一轮）。于是，它就把你传入的这个字典 {"messages": [...]} 当作初始的 State（初始化账本）。
    print(f"AI：{result['messages'][-1].content}")
    config2 = {"configurable": {"thread_id": "user_2"}}
    # 第二轮对话（注意：这次只说了“那上海呢”，没有提“天气”二字）
    print("\n--- 第二轮 ---")
    user_input_2 = "我刚刚问你了什么问题？"
    print(f"用户：{user_input_2}")
    result = app.invoke({"messages": [("user", user_input_2)]}, config2)  #当你第二轮调用 app.invoke({"messages": [("user", "那上海呢？")]}, config) 时：发现有历史账本（里面记着第一轮的“北京天气”）。LangGraph 会把你的新字典，合并（更新）到旧账本里。
    print(f"AI：{result['messages'][-1].content}")