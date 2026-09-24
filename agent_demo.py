import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from core.config import settings
# 用 @tool 装饰器，告诉 LangChain 这是一个工具
@tool
def get_weather(city: str) -> str:           #定义两个简单的函数，假装它们是真实的 API
    """查询指定城市的天气信息。"""               #多行字符串，文档字符串，用来给函数或类写说明。大模型就是靠这段文字来决定“什么情况用哪个工具”的。这必须是一句清晰的自然语言描述。
    print(f"\n[🔧 工具被调用了: get_weather, 参数: {city}]")
    # 假装查询数据库
    if city == "北京":
        return "北京今天晴朗，气温 25 度。"
    elif city == "上海":
        return "上海今天有小雨，气温 22 度。"
    else:
        return f"抱歉，没有找到{city}的天气数据。"

@tool
def get_exchange_rate(currency: str) -> str:
    """查询指定货币对人民币的汇率。"""
    print(f"\n[🔧 工具被调用了: get_exchange_rate, 参数: {currency}]")
    if currency == "美元":
        return "1美元 = 7.2人民币"
    elif currency == "欧元":
        return "1欧元 = 7.8人民币"
    else:
        return f"抱歉，没有找到{currency}的汇率数据。"
#把工具绑定给大模型
# 1. 初始化大模型
llm = ChatOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# 2. 关键一步：把工具告诉大模型
tools = [get_weather, get_exchange_rate]   #把工具函数放进一个列表
llm_with_tools = llm.bind_tools(tools)     #这是 LangChain 提供的方法。它会把这两个工具的名字、描述、参数格式，翻译成大模型 API 能看懂的格式，然后绑定上去。以后大模型回消息时，就知道自己有哪些“手脚”可用。
if __name__ == "__main__":
    print("=== 第一轮测试：直接问大模型，看它怎么回复 ===")
    question = "北京今天天气怎么样？"

    # 直接调用绑定了工具的模型
    response = llm_with_tools.invoke(question)  #bind_tools 是动作（方法），llm_with_tools 是动作产生的结果（新对象）。

    print(f"用户提问：{question}")
    print(f"AI的原始回复：{response}")
    print(f"AI回复的类型：{type(response)}")
    print(f"AI的纯文本内容：{response.content}")

    # 打印工具调用请求（如果有的话）
    print(f"AI请求调用的工具：{response.tool_calls}")
    # 如果大模型决定调用工具
    if response.tool_calls:     #（真值测试）        #response：大模型返回的 AIMessage 对象;tool_calls：AIMessage 对象的一个属性
        # 1. 提取工具信息
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]  # 拿到 "get_weather"
        tool_args = tool_call["args"]  # 拿到 {"city": "北京"}

        # 2. 根据名字，去我们的工具列表里找到对应的函数并执行
        # 这就是程序在干具体的脏活累活
        if tool_name == "get_weather":
            tool_result = get_weather.invoke(tool_args)
        elif tool_name == "get_exchange_rate":
            tool_result = get_exchange_rate.invoke(tool_args)

        print(f"工具执行结果：{tool_result}")

        # 3. 把大模型的请求和工具的结果，一起打包发回给大模型
        # 构造消息历史
        from langchain_core.messages import HumanMessage, ToolMessage

        messages = [
            HumanMessage(content=question),  # 用户问的问题
            response,  # 大模型的工具调用请求
            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])  # 工具执行结果
        ]

        # 4. 再次调用大模型，让它根据工具结果生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终的AI回答：{final_response.content}")