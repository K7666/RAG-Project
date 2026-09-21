#修改后
# core/llm.py
from langchain_openai import ChatOpenAI
from core.config import settings
from core.vector_store import VectorStoreManager


class RAGChatbot:
    def __init__(self):
        # 1. 初始化大模型（DeepSeek）
        self.llm = ChatOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )

        # 2. 初始化向量库管理员，并加载向量库
        self.vector_manager = VectorStoreManager()
        self.vectorstore = self.vector_manager.get_vectorstore()

        # 3. 把向量库包装成“检索器”，每次检索最相关的3块
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})#as_retriever()：把“仓库实体”升级成“检索前台”；search_kwargs={"k": 3}：这是个字典参数，意思是“每次检索，给我找出最相关的 3 个文本块”。
        #把一个“死仓库”变成了一个“按默认规则办事的活前台”。真正去书库里跑腿找那 3 本书的动作，是下一步的 invoke 干的。
    def query(self, question: str):
        """提问，返回AI回答和参考文档"""
        # 1. 去向量库里检索
        docs = self.retriever.invoke(question)     #.invoke(...)（方法/动作）这是 LangChain 框架里统一规定的“执行”动词。当你调用 self.retriever.invoke(question) 时，前台接待员（retriever）立刻行动：它拿到问题，用 Embedding 模型把问题转成向量，跑去向量数据库里计算相似度，然后把最相似的 3 个文本块（Document 对象）拿出来，打包成一个列表返回。
        #返回值 docs：是一个列表，里面装着 3 个 Document 对象
        # 2. 把检索到的块拼成上下文
        context = "\n\n".join([doc.page_content for doc in docs])  #[doc.page_content for doc in docs]：这是列表推导式。它的意思是：“遍历 docs 里的每一个 doc，把 doc.page_content（正文文本）拿出来，重新组合成一个新的列表。”
        #"\n\n".join(...)：这是 Python 字符串自带的 .join() 方法。它用 \n\n（两个换行符）作为分隔符，把刚才那个列表里的所有字符串拼接成一个大字符串。最后 context 就是一个包含了所有参考资料的纯文字。
        # 3. 构造提示词
        prompt = (                                  #每个字符串前面都有 f，这是 f-string 格式化。它会把 {context} 和 {question} 替换成真实的内容。
            f"请严格根据以下提供的上下文回答问题。"        #注意读一下这个提示词的内容：“请严格根据以下提供的上下文回答问题。如果找不到答案，就直接说不知道。” 这就是 RAG 减少幻觉的核心话术。
            f"如果上下文中找不到答案，请直接回答“我不知道”。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{question}"
        )

        # 4. 调用大模型
        response = self.llm.invoke(prompt)          #调用大模型。.invoke() 是 LangChain 的标准执行方法。#把拼好的 prompt 扔给 DeepSeek，它会返回一个 AIMessage 对象，我们存到 response 里。
        #self.llm.invoke(prompt) 会去调用 DeepSeek 的 API，把 prompt 发过去。DeepSeek 返回的结果是一个对象（不是纯字符串）。这个对象是 LangChain 内部定义好的，叫做 AIMessage。
        # 5. 返回回答和参考文档
        return response.content, docs               #因此response.content 是访问这个对象的 content 属性
        #Python 里逗号隔开就是元组。docs，检索到的 3 个原始文档块，用于前端溯源展示。