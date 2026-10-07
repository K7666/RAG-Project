# mcp_server.py
from mcp.server.fastmcp import FastMCP

# 1. 创建一个 MCP 服务器实例，给它起个名字
mcp = FastMCP("MyLifeTools")     #FastMCP：用于构建MCP应用程序的python框架；本质是一个“工具封装器”。它让你能用熟悉的 Python 装饰器（如 @mcp.tool()）来定义工具，然后自动帮你处理与 MCP 客户端（比如你的 LangGraph Agent）之间的通信协议

# 2. 用 @mcp.tool() 注册工具，替换之前的 @tool
@mcp.tool()                      #@mcp.tool()：和 @tool 的功能一样，只是它属于 MCP 的标准，而不是 LangChain 的标准。
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。"""
    if city == "北京":
        return "北京今天晴朗，气温 25 度。"
    elif city == "上海":
        return "上海今天有小雨，气温 22 度。"
    return f"抱歉，没有找到{city}的天气数据。"

@mcp.tool()
def get_music(mood: str) -> str:
    """根据天气或心情推荐一首歌。"""
    mood_str = str(mood)
    if "雨" in mood_str:
        return "推荐歌曲：周杰伦《晴天》—— 适合在下雨天怀念阳光。"
    elif "晴" in mood_str or "阳光" in mood_str or "舒适" in mood_str:
        return "推荐歌曲：五月天《倔强》—— 适合阳光明媚的日子去郊游。"
    return "推荐歌曲：陈奕迅《稳稳的幸福》—— 适合任何时候听。"

# 3. 启动服务器
if __name__ == "__main__":
    print("MCP 服务器已启动...")
    mcp.run()         #mcp.run()：默认模式下，它会启动一个标准输入输出流（stdio）服务器。