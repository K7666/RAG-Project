# api_server.py


#编写一个日志监控类
import logging     #导入 Python 自带的标准日志库。它比 print 高级得多，能自动加上时间戳、日志级别（INFO、ERROR），并且能把日志统一写进文件里。
from langchain_core.callbacks import BaseCallbackHandler    #从 LangChain 核心库里，拿出一个叫“基础回调处理器”的父类图纸。
                                                            #什么是 Callback（回调）？你之前学过 EventSourceResponse 的流式输出，LangChain 内部在发生各种事件（开始调工具、大模型开始说话、大模型说完）时，会“大喊”一声。这个类的作用，就是站在旁边听喊声。
# 配置 Python 标准日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")   #logging.basicConfig(...)：配置日志长什么样。level=logging.INFO：设置门槛。只记录 INFO 及以上的日志（DEBUG 这种啰嗦的就不管了）。format="%(asctime)s - %(levelname)s - %(message)s"：定义日志的排版，即：时间戳 - 日志级别 - 你写的信息。打印出来大概是 2026-10-18 21:00:01 - INFO - 工具开始...。
logger = logging.getLogger("AgentMonitor")   #logger = logging.getLogger("AgentMonitor")：创建一个叫 AgentMonitor 的日志记录员（对象）。以后我们让它去哪打印，它就去哪打印。

# 自定义回调类，专门用来记录 Agent 的轨迹
class AgentMonitorCallback(BaseCallbackHandler):   #定义一个名为 AgentMonitorCallback 的新类，继承前面拿出来的 BaseCallbackHandler 父类。因为继承了父类，只要 LangChain 内部触发了事件（比如调用工具），自动就会跑到我们这个类里找对应的方法（比如 on_tool_start）并执行。
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):   #def on_tool_start(...)：当 Agent 决定调用某个工具，还没真正执行时，LangChain 会自动调用这个方法。serialized：包含工具信息的字典（比如工具的 name）。input_str：传给工具的入参（比如 北京）。**kwargs：极重要。LangChain 会额外塞给你很多参数，但我们可能不需要全用。**kwargs 代表“接收剩下的所有任意参数”。
        tool_name = serialized.get("name", "未知工具")  #tool_name = serialized.get("name", "未知工具")：从字典里安全地取名字。如果有 name 就取，没有就返回默认值“未知工具”。
        logger.info(f"🔧 [工具开始] {tool_name} | 入参: {input_str}")  #logger.info(f"...")：打印日志。记录哪个工具开始了、参数是什么。

    def on_tool_end(self, output, **kwargs):       #def on_tool_end(...)：工具执行完毕后，自动调用。output：工具返回的具体数据。len(str(output))：因为返回的可能是字符串、列表或对象，str() 强制转换后取长度，看看返回了多少字。这在排查“工具返回空数据”的 Bug 时非常有用。
        logger.info(f"✅ [工具结束] 返回结果长度: {len(str(output))} 字符")

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):  #def on_llm_start(...)：大模型开始生成（不管是普通对话还是流式输出）。prompts：发给大模型的提示词。这里我们就不打印了（太长），只记录一句“正在生成”。
        logger.info(f"🧠 [大模型思考] 正在生成...")

    def on_llm_end(self, response, **kwargs):  #def on_llm_end(self, response, **kwargs)：大模型生成完毕。
        # 尝试提取 Token 消耗（如果是 OpenAI 兼容格式，通常在这里面）
        try:
            token_usage = response.llm_output.get("token_usage", {})  #response.llm_output：这是一个包含本次调用元数据的字典。如果是 OpenAI/DeepSeek 兼容格式，里面会有 token_usage（Token 消耗账单）。response.llm_output.get("token_usage", {})：安全提取。如果没有这个键，就返回空字典。
            if token_usage:
                logger.info(f"🏁 [大模型结束] 消耗 Token: {token_usage.get('total_tokens', '未知')}")
            else:
                logger.info(f"🏁 [大模型结束]")
        except Exception:
            logger.info(f"🏁 [大模型结束]")

#LangChain 是如何知道“工具开始”就要找 on_tool_start？因为源码里定死了标准事件名：
#工具开始 → 触发 on_tool_start

#工具结束 → 触发 on_tool_end

#大模型开始 → 触发 on_llm_start

#大模型结束 → 触发 on_llm_end



import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"    #设置国内镜像，防止 HuggingFace 下载模型时超时。这行必须在所有 langchain 导入之前，否则不生效。

import asyncio
import time    #时间工具箱，用来计算接口耗时。
from contextlib import asynccontextmanager    #导入“异步上下文管理器”装饰器。这是 FastAPI 生命周期管理的关键工具。

from fastapi import FastAPI    #FastAPI 框架的核心类，用来创建 Web 服务。
from pydantic import BaseModel #Pydantic 的基类，用来定义请求/响应的数据结构（类似一个带类型校验的字典）。
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from core.config import settings

# ==========================================
# 全局变量（在启动时初始化，之后复用）
# ==========================================
# 为什么需要它？Agent 初始化很耗时（要加载模型、连接 MCP），不能在每次 HTTP 请求时都重新初始化一次。所以我们在服务启动时初始化一次，存进 app_state，之后每个请求直接拿。
app_state = {}   #：这是一个全局字典，用来存放“服务启动后初始化好的 Agent 对象”。


# ==========================================
# 启动/关闭生命周期
# ==========================================
@asynccontextmanager    #这是一个装饰器，它把一个异步函数变成“异步上下文管理器”。
async def lifespan(app: FastAPI):  #lifespan：函数名（FastAPI 规定的标准名字）;app: FastAPI：FastAPI 会自动把 app 实例传进来。这个函数会在服务启动时执行一次（yield 之前），在服务关闭时执行一次（yield 之后）。
    """服务启动时初始化 Agent，关闭时清理"""
    print("🚀 正在初始化 Agent 系统...")

    # 1. MCP 客户端
    client = MultiServerMCPClient({
        "my_life_tools": {
            "command": r"D:\pycharm\RAG_Project\venv\Scripts\python.exe",
            "args": ["mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    print(f"✅ 成功加载了 {len(tools)} 个工具：{[t.name for t in tools]}")

    # 2. LLM
    llm = ChatOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        streaming=True,  # 开启流式,告诉大模型“以后调用你时，可以一个字一个字地返回”，为后面的 SSE 流式输出做准备。
    )

    # 3. 三个专家 Agent
    weather_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "get_weather"],
        prompt="你是一个天气专家。用户问天气时，调用 get_weather 工具。",
        name="weather_agent"
    )
    knowledge_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "search_knowledge_base"],
        prompt="你是一个技术知识专家。用户问 RAG、大模型等概念时，调用 search_knowledge_base 工具。",
        name="knowledge_agent"
    )
    music_agent = create_react_agent(
        llm,
        tools=[t for t in tools if t.name == "get_music"],
        prompt="你是一个音乐推荐专家。用户让你推荐歌曲时，你【必须】调用 get_music 工具。如果工具没返回结果，你就说'抱歉我查不到'，严禁自己编造任何歌名！",
        name="music_agent"
    )

    # 4. 主管 Agent
    supervisor = create_supervisor(
        agents=[weather_agent, knowledge_agent, music_agent],
        model=llm,
        prompt=(
            "你是一个严厉的主管，负责调度专家。\n"
            "【绝对死命令】\n"
            "1. 如果用户问天气，你【必须】调用 transfer_to_weather_agent。\n"
            "2. 如果用户要求推荐歌曲，你【必须】调用 transfer_to_music_agent。\n"
            "3. 无论你脑子里知不知道答案，你都【绝对禁止】自己作答！\n"
            "4. 你必须等两个专家都返回了结果，才能生成最终回复。\n"
            "5. 在最终回复时，必须原样保留天气专家和音乐专家给你的具体内容，禁止自己编造！"
        )
    )


    memory = MemorySaver()
    agent_app = supervisor.compile(checkpointer=memory)   #编译成可运行的图。

    # 保存到全局，供接口调用
    app_state["agent"] = agent_app  #把初始化好的 Agent 存进全局字典 app_state。以后每个 HTTP 请求进来时，直接从 app_state["agent"] 拿，不用重新初始化。
    app_state["llm"] = llm          #把初始化好的 llm 存进全局字典 app_state。

    print("✅ Agent 系统初始化完成！")
    yield   #这是生命周期函数的分界线。yield 之前是启动逻辑，yield 之后是关闭逻辑。
    print("👋 服务关闭中...")
#服务运行时，会停在 yield 这一行，直到收到关闭信号，才会继续执行后面的代码。

# ==========================================
# FastAPI 应用
# ==========================================
app = FastAPI(lifespan=lifespan, title="RAG Agent API")  #创建 FastAPI 应用实例。lifespan=lifespan：把刚才定义的生命周期函数绑定到 app 上。title="RAG Agent API"：设置 API 文档的标题。
#FastAPI 是一个“Web 服务框架”，让你的 Python 代码能接收 HTTP 请求、返回 HTTP 响应。将智能体包装成一个网络接口，别的程序，前端网页，小程序，可以通过网络来调用它。

# ==========================================
# 请求/响应模型
# ==========================================
class ChatRequest(BaseModel):        #BaseModel 是 Pydantic 提供的类。继承它之后，就定义了一个“数据模型”。
    message: str                     #message: str：请求必须包含一个字符串类型的 message。
    thread_id: str = "default_user"  #thread_id: str = "default_user"：可选字段，默认值是 "default_user"。
#Pydantic 的作用：FastAPI 会自动校验请求体。如果前端传的 message 不是字符串，或者缺少必填字段，FastAPI 会自动返回 422 错误，不需要你写任何校验代码。
#BaseModel 自动做了什么？当 FastAPI 收到一个 HTTP 请求时：把请求体（JSON）自动解析成 ChatRequest 对象。如果 message 是数字 123，FastAPI 会自动转成字符串 "123"。如果缺少 message，FastAPI 会直接返回错误 422 Unprocessable Entity，告诉你“缺少必填字段”。
class ChatResponse(BaseModel):
    answer: str
    elapsed: float
#定义响应模型：返回的 JSON 里必须有 answer（字符串）和 elapsed（浮点数）。

# ==========================================
# 接口：普通问答
# ==========================================
@app.post("/chat", response_model=ChatResponse)     #@app.post("/chat")：装饰器，表示这个函数处理 POST /chat 请求。response_model=ChatResponse：告诉 FastAPI“返回值必须符合 ChatResponse 模型”，FastAPI 会自动序列化成 JSON。
async def chat(req: ChatRequest):   #异步函数，接收一个 ChatRequest 类型的参数。FastAPI 会自动把请求体反序列化成 ChatRequest 对象。
    agent_app = app_state["agent"]
    config = {"configurable": {"thread_id": req.thread_id},"callbacks": [AgentMonitorCallback()]}# 挂载黑匣子...config 是在你调用 app.ainvoke(..., config) 时，随手携带的一个“环境配置”。它就像是发给整个智能体所有节点的“对讲机频道”。

    start = time.time()    #记录开始时间。
    result = await agent_app.ainvoke({"messages": [("user", req.message)]}, config)
    elapsed = time.time() - start    #计算耗时。

    answer = result["messages"][-1].content   #从最终账本里取最后一条消息的文本内容。
    print(f"[LOG] 耗时 {elapsed:.2f}s | 问题: {req.message[:30]}...")   #可观测性初步：打印日志。{elapsed:.2f}：保留 2 位小数。{req.message[:30]}：取问题前 30 个字符，防止日志太长。

    return ChatResponse(answer=answer, elapsed=elapsed)   #构造响应对象并返回。


# ==========================================
# 接口：SSE 流式问答
# ==========================================
from sse_starlette.sse import EventSourceResponse    #导入 SSE 响应类。sse-starlette 是 FastAPI 生态里最常用的 SSE 库。


@app.post("/chat/stream")    #定义一个新的 POST 路由 /chat/stream。
async def chat_stream(req: ChatRequest):
    agent_app = app_state["agent"]
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_generator():    #这是一个异步生成器。它不像普通函数那样 return 一次，而是可以 yield 多次，每次 yield 一段数据。
        # 使用 astream_events 监听底层事件，agent_app.astream_events(...)：这是 LangChain 提供的显微镜级别的监听工具。它不仅监听节点流转，还能监听大模型底层的网络请求（比如调用了哪次 API、正在返回哪几个 Token）。
        async for event in agent_app.astream_events(
                {"messages": [("user", req.message)]},   #把用户的问题发出去。
                config,                                  #带上会话配置
                version="v2"  # 必须指定版本 v2。：LangChain 规定，用这个监听器必须声明使用 v2 版本的数据格式，否则返回的数据包结构会不一致。
        ):
            #在这个监听循环里，event 是个非常庞大的数据包。底层每发生一点小事（比如节点开始跑了、节点跑完了、大模型开始说话了、大模型说完了），都会抛出一个 event。我们需要从中提取两个关键指纹：
            kind = event["event"]   #kind（事件类型）：event["event"] 会返回类似 "on_chain_start"、"on_tool_start"、"on_chat_model_stream" 这样的字符串。
            #node_name = event["metadata"].get("langgraph_node", "")   #node_name（来源节点）：event["metadata"] 是一个字典，记录了是谁触发了这个事件。.get("langgraph_node", "") 意思是：“去找叫 langgraph_node 的键，如果找不到，就返回空字符串”。

            # 只捕获“模型正在输出文本”的事件，且节点必须是主管
            if kind == "on_chat_model_stream":#and node_name == "supervisor":   #kind == "on_chat_model_stream"：这意味着“大模型刚刚生产出了几个字（Token）”。如果大模型在调工具（on_tool_start），这个条件就不成立，直接被丢弃。node_name == "supervisor"：这意味着“此时说话的是主管”。
                node_name = event["metadata"].get("langgraph_node", "未知节点")
                print(f"\n👉 [底层事件] 当前节点 [{node_name}] 正在输出文字...")
                chunk = event["data"]["chunk"]   #chunk = event["data"]["chunk"]：event["data"] 里装着这次微事件的数据。chunk 就是大模型刚刚吐出来的那个“碎片”（比如只有“北京今天”这几个字）。
                if hasattr(chunk, "content") and chunk.content:   #hasattr(chunk, "content")：防御性检查，确认这个碎片里真的有文本内容。
                    # 直接把每一个字推给前端
                    yield f"{chunk.content}\n\n"

        yield "[DONE]\n\n"
    return EventSourceResponse(event_generator())     #EventSourceResponse(...)：它的作用是：把这个“生成器对象（水龙头）”，包装成符合 SSE（Server-Sent Events）协议的 HTTP 响应。当你 return 它时，FastAPI 不会一次性把所有数据打包发给前端，而是保持连接不断开，只要你的生成器 yield 一个数据，它就立刻推给前端。

# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    import uvicorn  #uvicorn 是 ASGI 服务器，用来跑 FastAPI 应用。

    uvicorn.run(app, host="0.0.0.0", port=8000)   #host="0.0.0.0"：监听所有网络接口（别人通过你的 IP 也能访问）。port=8000：监听 8000 端口。


#1.启动时初始化 MCP 客户端、加载工具、创建多智能体系统。

#2.暴露一个 /chat 接口，接收用户消息，返回 AI 回答。

#3.暴露一个 /chat/stream 接口，支持 SSE 流式输出。

#4.每次调用都记录日志（耗时、工具调用、Token 消耗）

# 用户 → HTTP 请求 → FastAPI 服务 → LangGraph Agent → 返回结果
#                          ↓
#                     日志记录（每次调用）


# 用户（浏览器/Postman）
#     ↓ 发送 POST /chat 请求，Body 是 JSON
# FastAPI 前台（app = FastAPI(...)）
#     ↓ 匹配路由 @app.post("/chat")
#     ↓ 把 JSON 解析成 ChatRequest（BaseModel 校验）
# 执行 chat(req) 函数
#     ↓ 调用 Agent
# 返回 ChatResponse
#     ↓ FastAPI 自动序列化成 JSON
# 用户收到响应



#“在做流式输出时，我发现由于 Supervisor 模式的框架封装，底层把所有 Agent 的节点都统一命名为了 agent。
# 这导致流式输出时，主管和专家的‘内心戏’、工具调用结果全被推给了前端，体验极差。
# 我通过切换 astream_events 到 astream，并针对主管节点（supervisor）做严格过滤，解决了这个问题。
# 如果要同时实现‘完美打字机’和‘精准过滤’，业界通常需要用 LangGraph 的 StateGraph 手写底层图，摆脱黑盒封装。”