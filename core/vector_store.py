#修改后代码
# core/vector_store.py
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 提前贴好告示！

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from core.config import settings


class VectorStoreManager:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None

    def _init_embeddings(self, retries: int = 3):      #_init_embeddings：方法名。下划线开头是一种约定，告诉别人“这是内部的，外人不要随便调用”。
        """带有重试机制的模型初始化（也是真正下载模型的地方）"""
        for i in range(retries):
            try:
                print(f"正在初始化 Embedding 模型 (尝试 {i + 1}/{retries})...")
                self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")   #造一个 HuggingFaceEmbeddings 对象，让它去下载或加载这个模型，然后把造好的对象赋给当前对象的 embeddings 属性。（翻译机）
                return  # 成功了就退出循环
            except Exception as e:                    #Exception：Python 内置的异常类（所有报错的老祖宗）；as e：把捕获到的异常对象，赋值给变量 e。“如果 try 里面的代码报错了，抓住这个错误，把它存到 e 里面。
                print(f"❌ 初始化失败：{e}")
                if i < retries - 1:
                    time.sleep(2)  # 等2秒再试          #time：时间工具箱。sleep：函数，意思是“等待”。
                else:
                    raise Exception("重试多次依然失败，请检查网络或HF镜像设置！")

    def get_vectorstore(self, chunks=None):
        self._init_embeddings()               #调用前面定义的那个内部方法

        if os.path.exists(settings.CHROMA_PATH) and len(os.listdir(settings.CHROMA_PATH)) > 0:
            print("3. 发现本地已有向量数据库，直接加载...")
            self.vectorstore = Chroma(persist_directory=settings.CHROMA_PATH, embedding_function=self.embeddings)   #Chroma(...)：实例化仓库类。括号里传了两个关键字参数；persist_directory=...：告诉它数据库存在哪；embedding_function=self.embeddings：告诉它用哪个翻译机
            return self.vectorstore

        print("3. 开始向量化并存入向量数据库...")
        self.vectorstore = Chroma.from_documents(        #当一个方法在定义时，被加上了特殊的“标记”（@classmethod）。则调用这个方法不用先实例化在调用实例方法，直接在这个方法前面加上类名.即可；这里就是这样。普通实例方法：需要先有“对象”，用 对象.方法() 调用。类方法：不需要先有对象，直接用 类名.方法() 调用（比如 Chroma.from_documents）。
            documents=chunks,                     #切分好的文本块
            embedding=self.embeddings,            #翻译机
            persist_directory=settings.CHROMA_PATH#向量放在哪
        )
        print("✅ 向量数据库构建完成！")
        return self.vectorstore             #返回的是“一个可以用、可以操作的仓库实体”，而不是一句简单的通知