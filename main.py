#修改后
# main.py
import os
#“我要下载模型了。先看看用户有没有设置名叫 HF_ENDPOINT 的环境变量。如果有，我就去这个地址下；如果没有，我就去默认的 huggingface.co。”
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"     # "HF_ENDPOINT": "这里是存放下载地址的地方"；"https://hf-mirror.com" → 把地址改成国内镜像站
from core.data_loader import DataLoader
from core.vector_store import VectorStoreManager
from core.llm import RAGChatbot
if __name__ == "__main__":
    # 1. 加载与切分文档（实例化数据加载器对象）
    loader = DataLoader()

    # 调用对象里的方法，拿到切分好的块
    chunks = loader.load_and_split()
    # 2. 构建/加载向量数据库（RAGChatbot 内部会自动完成）
    # 这里不需要再单独调用 VectorStoreManager 了，因为 RAGChatbot 内部会调
    # 3. 初始化 RAG 机器人
    print("\n4. 初始化 RAG 机器人...")
    bot = RAGChatbot()

    # 4. 开始提问
    print("\n5. 开始提问...")
    question = "什么是RAG？"
    answer, docs = bot.query(question)

    print("-" * 40)
    print(f"用户提问：{question}")
    print(f"AI回答：{answer}")
    print("-" * 40)
    print(f"参考了 {len(docs)} 个文档块")

#代码解释
# import os：拿出 Python 自带的“操作系统工具箱”。
#
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"：这是设置系统环境变量。
#
# 为什么写这个？ 因为我们之前踩过一个坑——下载 HuggingFace 模型时，默认会去国外的服务器，导致连接超时（WinError 10060）。
#
# 这就好比你给饭店的采购车装了一个“国内导航”，告诉它“别去国外绕远了，直接去 hf-mirror.com 这个国内仓库拿面粉（模型）”。
#
# ⚠️ 核心细节：这行代码必须在所有 langchain 导入之前执行，所以它放在了第一行。如果你把它放到 from core.data_loader 后面，导航就失效了。
#
# from core.data_loader import DataLoader：从你自己的模块（core/data_loader.py）里，拿出了“备菜部门”的图纸（DataLoader 类）。
#
# python
# if __name__ == "__main__":
# 这是 Python 中最经典的“大门守卫”代码。
#
# 翻译成人话：“只有当我直接运行 main.py 这个文件时，下面的代码才会执行。如果别的文件 import 这个文件，这里面的代码绝对不会瞎跑。”
#
# 为什么要这么写？ 以后你做测试，可能有别的脚本会引用 main.py 里的东西。没有这句话，一旦被引用，程序就会自动去加载文档、切分文档，造成严重的资源浪费。这就叫“避免误触发”。
#
# python
#     # 实例化数据加载器对象
#     loader = DataLoader()
# loader = DataLoader()：按照图纸，真的雇佣了一个叫 loader 的备菜厨师（实例化对象）。
#
# 重点看括号里是空的！为什么？因为 DataLoader 的 __init__ 方法里写了默认参数（docs_path=None）。所以 loader 会自动去 settings 配置里拿默认的路径（./docs）和块大小（500）。
#
# 如果明天你想临时处理另一个文件夹，只需要改成 loader = DataLoader(docs_path="./new_docs")，非常灵活。
#
# python
#     # 调用对象里的方法，拿到切分好的块
#     chunks = loader.load_and_split()
# loader.load_and_split()：你给这位备菜厨师下达命令：“开始干活！去 ./docs 拿文件，切碎它们！”
#
# 这个方法内部执行了我们昨天讲的那堆逻辑（DirectoryLoader 找文件、TextLoader 读文件、RecursiveCharacterTextSplitter 切块）。
#
# chunks：厨师干完活，把切好的 63 块食材（文本块）端出来，交到了老板（main.py）的手上。
