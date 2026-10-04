# ==========================================
# 1. 准备工作：导入工具和模型
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from core.config import settings


#from langgraph.checkpoint.memory import MemorySaver   #MemorySaver 是 LangGraph 内置的内存检查点工具。它就像一个“内存账本管理器”，专门帮你把每个对话的账本快照保存下来。
#memory = MemorySaver()


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
# 2. 定义状态（State）—— 共享账本
# ==========================================
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# 这就是 LangGraph 自带的“带消息记录功能的账本”
# 为什么用它？它会自动帮我们把新消息追加到 messages 列表里，不用我们自己写 .append()
# MessagesState 是 LangGraph 帮我们写好的一种状态。它里面自动包含了一个 messages 键。你上一轮手动拼 [HumanMessage, response, ToolMessage] 的痛苦，从今天起彻底消失了。
# ==========================================
# 3. 定义节点（Node）—— 干活的人
# ==========================================
from langchain_core.messages import SystemMessage
# 节点 A：大模型节点
def agent_node(state: MessagesState):    #“定义一个叫 agent_node 的函数，它接收一个名叫 state 的参数（这是一个 MessagesState 类型的账本）。”MessagesState 是什么？本质是一个字典，大概是这样的结构：{"messages": [消息1, 消息2, 消息3...]}。 它是 LangGraph 官方提供的一个专门的账本类。这个账本内部自带一个 messages 字段。
    # 从账本里取出历史消息
    messages = state["messages"]   #从账本里，把名为 messages 的那一页抽出来。把抽出来的消息列表赋值给变量 messages。此时，messages 里可能包含用户的问题、上一轮的历史等等。
    sys_msg = SystemMessage(
        content="你是一个生活助手，你可以查询天气、汇率和推荐音乐,当然也可以搜索资料。请根据用户的需求，自主决定调用哪些工具。在回答时，请给出有温度的建议。当用户让你整理某个主题的资料时，请严格按以下步骤执行：\n"
                "1. 首先，调用 mock_web_search 工具，搜索该主题。\n"
                "2. 然后，阅读搜索返回的内容，从中筛选出最核心的 3 个观点。\n"
                "3. 最后，不要直接复制搜索结果，而是用你自己的话，把这三个观点整理成一份结构清晰的简报。\n"
                "简报格式要求：\n"
                "### 主题：[主题名称]\n"
                "**核心观点 1**：...\n"
                "**核心观点 2**：...\n"
                "**核心观点 3**：...\n"
                "**总结**：..."
    )
    # 把消息交给绑定了工具的大模型
    response = llm_with_tools.invoke([sys_msg] + messages)
    # 把大模型的回复包成字典，追加回账本
    return {"messages": [response]}      #为什么不用 state["messages"].append 然后 return state？因为 state 是一个“只读快照”，修改它不会生效，框架只接受你返回的“增量字典”。

# 节点 B：工具节点
# ToolNode 是 LangGraph 内置的“万能工具执行器”
# 你只要把 tools 列表传进去，它就能自动提取大模型的要求，执行函数，并且自动带上 tool_call_id 返回结果！
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)   #实例化这个工具节点，tools 是你之前定义的列表（查天气、查汇率）。你只要把工具列表给它，它就会自动去执行。返回的结果也是符合 LangGraph 规范的字典 {"messages": [ToolMessage...]}。
# ==========================================
# 4. 定义图（Graph）—— 拼装流水线
# ==========================================

# 用“账本”作为图纸，建一个流程图
workflow = StateGraph(MessagesState)   #StateGraph：这是 LangGraph 的核心类，意思是“状态图”。一个图包含了节点和边。MessagesState：告诉图，我们用的“账本”类型是 MessagesState（会自动携带 messages 列表）。

# 把节点加入图里，起个名字
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 画普通的边（单向流动）
workflow.add_edge(START, "agent")       # 起点 -> 大模型节点（说明图一跑起来，先把用户的问题发给大模型节点）
workflow.add_edge("tools", "agent")     # 工具节点 -> 大模型节点（这说明工具执行完后，一定会回到大模型节点）

# 画条件边（最核心的“大脑”）
# 引入内置的判断函数 tools_condition
from langgraph.prebuilt import tools_condition   #tools_condition：LangGraph 内置的“判断函数”。它会自动检查 agent 节点返回的 AIMessage 里有没有 tool_calls
#from langchain_core.messages import HumanMessage
# 意思：大模型节点执行完后，判断一下。如果它要调工具，就去 "tools" 节点；
# 如果它说完了（不需要调工具），就走向终点 END。
workflow.add_conditional_edges(    #add_conditional_edges：添加“条件边”（虚线）。意思是：agent 节点执行完后，根据 tools_condition 的判断结果决定走哪条路
    "agent",
    tools_condition,               # 自动判断：有大模型发来的 tool_calls 吗？
)                                  #如果大模型要调工具，就去 "tools" 节点；如果大模型回答完了（无 tool_calls），就自动走向 END（终点）。因为这里传了 tools_condition，LangGraph 会自动帮我们连好 "tools" 和 END 这两条线。
# 编译这张图
app = workflow.compile(checkpointer=memory)           #compile()：把图纸变成可以运行的应用；checkpointer 是编译时传的参数。你告诉图：“以后每次流转，都把这个账本的快照存到 memory 里，别用完就扔。”

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    print("=== 全自动 LangGraph Agent ===")
    config = {"configurable": {"thread_id": "user_1"}}
    # 第一轮对话
    print("\n--- 第一轮 ---")
    user_input_1 = "那上海呢"
    print(f"用户：{user_input_1}")
    result = app.invoke({"messages": [("user", user_input_1)]}, config)  #当你第一轮调用 app.invoke({"messages": [("user", user_input_1)]}, config) 时：LangGraph 去 MemorySaver/SqliteSaver 里找 thread_id="user_1" 的账本发现没找到（因为是第一轮）。于是，它就把你传入的这个字典 {"messages": [...]} 当作初始的 State（初始化账本）。
    print(f"AI：{result['messages'][-1].content}")
    config2 = {"configurable": {"thread_id": "user_2"}}
    # 第二轮对话（注意：这次只说了“那上海呢”，没有提“天气”二字）
    print("\n--- 第二轮 ---")
    user_input_2 = "那北京呢？"
    print(f"用户：{user_input_2}")
    result = app.invoke({"messages": [("user", user_input_2)]}, config2)  #当你第二轮调用 app.invoke({"messages": [("user", "那上海呢？")]}, config) 时：发现有历史账本（里面记着第一轮的“北京天气”）。LangGraph 会把你的新字典，合并（更新）到旧账本里。
    print(f"AI：{result['messages'][-1].content}")



#如果面试官问：“你的 Agent 怎么实现记忆持久化？”
#你可以回答：
#“我使用 LangGraph 的 SqliteSaver 作为检查点保存器，把对话状态持久化到 SQLite 数据库中。通过 thread_id 区分不同用户的会话。这样即使程序重启，也能恢复完整的对话历史。
# 在生产环境，我会把 SQLite 替换为 Postgres 或 Redis，来支持高并发。”