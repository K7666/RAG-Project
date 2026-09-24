# ==========================================
# 1. 准备工作：导入工具和模型
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from core.config import settings

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

# 初始化大模型并绑定工具
llm = ChatOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)
tools = [get_weather, get_exchange_rate]
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

# 节点 A：大模型节点
def agent_node(state: MessagesState):    #“定义一个叫 agent_node 的函数，它接收一个名叫 state 的参数（这是一个 MessagesState 类型的账本）。”MessagesState 是什么？本质是一个字典，大概是这样的结构：{"messages": [消息1, 消息2, 消息3...]}。 它是 LangGraph 官方提供的一个专门的账本类。这个账本内部自带一个 messages 字段。
    # 从账本里取出历史消息
    messages = state["messages"]         #从账本里，把名为 messages 的那一页抽出来。把抽出来的消息列表赋值给变量 messages。此时，messages 里可能包含用户的问题、上一轮的历史等等。
    # 把消息交给绑定了工具的大模型
    response = llm_with_tools.invoke(messages)
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
from langchain_core.messages import HumanMessage
# 意思：大模型节点执行完后，判断一下。如果它要调工具，就去 "tools" 节点；
# 如果它说完了（不需要调工具），就走向终点 END。
workflow.add_conditional_edges(    #add_conditional_edges：添加“条件边”（虚线）。意思是：agent 节点执行完后，根据 tools_condition 的判断结果决定走哪条路
    "agent",
    tools_condition,               # 自动判断：有大模型发来的 tool_calls 吗？
)                                  #如果大模型要调工具，就去 "tools" 节点；如果大模型回答完了（无 tool_calls），就自动走向 END（终点）。因为这里传了 tools_condition，LangGraph 会自动帮我们连好 "tools" 和 END 这两条线。
# 编译这张图
app = workflow.compile()           #compile()：把图纸变成可以运行的应用

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    print("=== 全自动 LangGraph Agent ===")

    # 模拟用户提问
    user_input = "北京今天天气怎么样？顺便告诉我1美元相当于多少人民币？"
    print(f"用户提问：{user_input}\n")

    # 向图里输入初始状态
    # 注意：这里格式变了，要传入 {"messages": [("user", user_input)]} LangGraph 需要传入符合初始状态字典格式的数据。
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})   #给 state 这个字典的 messages 键传入了一个初始列表。

    # result 是最终的状态（账本），里面包含了所有的对话历史
    # 我们只需要账本里最后一条消息（也就是大模型的最终总结）
    final_message = result["messages"][-1]
    print(f"最终的AI回答：{final_message.content}")