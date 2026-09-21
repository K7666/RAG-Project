#修改后
# core/data_loader.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import settings

class DataLoader:
    # ⚠️ 这里用到了你刚学的：默认参数和类型注解
    def __init__(self, docs_path: str = None, chunk_size: int = None):
        # 巧妙之处：如果外部没传 docs_path，就用 settings 里的默认值
        self.docs_path = docs_path or settings.DOCS_PATH
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        # 这个没设参数，直接去 settings 拿
        self.chunk_overlap = settings.CHUNK_OVERLAP

    # ⚠️ 这里用到了你刚学的：类型注解 -> list 代表返回一个列表
    def load_and_split(self) -> list:
        print("1. 开始加载文档...")
        # 注意：这里从 self 里拿路径，而不是去 config 里拿
        loader = DirectoryLoader(self.docs_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents = loader.load()
        print(f"成功加载了 {len(documents)} 个文件。")

        print("2. 开始切分文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"切分完毕！总共切出了 {len(chunks)} 个小块。")
        return chunks


#代码解释
# class DataLoader:
#     # ⚠️ 这里用到了你刚学的：默认参数和类型注解
#     def __init__(self, docs_path: str = None, chunk_size: int = None):
#         # 巧妙之处：如果外部没传 docs_path，就用 settings 里的默认值
#         self.docs_path = docs_path or settings.DOCS_PATH
#         self.chunk_size = chunk_size or settings.CHUNK_SIZE
#         # 这个没设参数，直接去 settings 拿
#         self.chunk_overlap = settings.CHUNK_OVERLAP
# 这短短几行，包含了你这几天学的所有硬核概念，非常精彩：
#
# class DataLoader:
# 你在绘制一张“数据加载部门”的图纸（类）。
#
# def __init__(self, docs_path: str = None, chunk_size: int = None):
#
# 这是部门成立时的“启动会议”（构造函数）。
#
# docs_path: str = None：这是你刚在 B站 补的类型注解 + 默认参数。意思是：“部门成立时，你可以给我传一个文档路径。如果传了，必须是字符串（: str）；如果你不传，默认就是空的（= None）。”
#
# 同样，chunk_size 也是这个意思。
#
# self.docs_path = docs_path or settings.DOCS_PATH （核心神句！）
#
# 这句话是在分配资源。怎么分配？
#
# 这里用到了 Python 的 or 语法（A or B：如果 A 是真东西，就用 A；如果 A 是空的（None），就用 B）。
#
# 翻译成人话：“如果调用我的时候，你给我传了 docs_path，我就把这个路径当成我的 self.docs_path；如果你没给我传，那我就去 settings 这个工具箱里，拿默认的路径（./docs）。”
#
# 这就是我们之前说过的：灵活性。你不传，我用默认的；你传了，我听你的。这就是为什么重构后代码突然变得很强大。
#
# self.chunk_size = chunk_size or settings.CHUNK_SIZE
#
# 同上。你没传块大小，我就去工具箱里拿 500。
#
# self.chunk_overlap = settings.CHUNK_OVERLAP
#
# 为什么这里没写 or？因为你在 __init__ 的参数里，根本没有给它留位置（没有定义 chunk_overlap 参数）。所以它没得选，必须去 settings 里拿默认值（50）。




# loader = DirectoryLoader(self.docs_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
# documents = loader.load()
# print(f"成功加载了 {len(documents)} 个文件。")
# DirectoryLoader（第三方库 LangChain 的类）：这是一个“大总管”。它专门负责批量处理文件。
#
# self.docs_path（参数）：告诉大总管去哪里找菜（也就是 ./docs 文件夹）。
#
# glob="**/*.md"（参数）：大总管问“你要找什么类型的文件？” 你说“只要后缀是 .md 的，在这个文件夹里（**/ 代表任意子文件夹）全都给我找出来。”
#
# loader_cls=TextLoader（参数）：大总管又问“找出来怎么读？” 你说“用 TextLoader 这个工具，把它当成纯文本（TXT）来读。”
#
# loader_kwargs={"encoding": "utf-8"}（参数）：大总管又问“遇到中文咋办？” 你说“强制用 utf-8 编码打开，别给我搞出乱码。”
#
# loader.load()（方法）：你把所有指令下达后，大总管开始干活（.load()），把文件一个个读进内存，打包成一个列表。
#
# documents（变量）：这个列表就叫 documents。里面装着 3 个“文档对象”，每个对象都包含文件内容和来源信息。
#
# print(f"...{len(documents)}...")：数一下，哦，搬来了 3 个文件。
#
# 🔪 第二部分：切文件（切菜工进场）
# python
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=self.chunk_size,
#     chunk_overlap=self.chunk_overlap,
#     separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
# )
# chunks = text_splitter.split_documents(documents)
# print(f"切分完毕！总共切出了 {len(chunks)} 个小块。")
# RecursiveCharacterTextSplitter（第三方库 LangChain 的类）：这是一个“智能切菜机”。专门把长文本切断，但又尽量不破坏句子意思。
#
# chunk_size=self.chunk_size（参数）：切菜机面板上的“厚度”旋钮。这里设置为 500，意思是“每一块最多 500 个字符”。
#
# chunk_overlap=self.chunk_overlap（参数）：切菜机面板上的“重叠”旋钮。设置为 50，意思是“切下一块的时候，把上一块的尾巴 50 个字也带进来”，这样防止一句话正好被切断，导致意思变了。
#
# separators=[...]（参数）：这是切菜机的“下刀优先级”。它是个列表。
#
# 先从最大的空隙切：两个换行 \n\n（段落之间）。
#
# 如果不够，找单换行 \n。
#
# 再不够，找中文句号 。、感叹号 ！、问号 ？、逗号 ，。
#
# 最后是空格，如果连空格都没有，那就硬切（""）。
#
# text_splitter.split_documents(documents)（方法）：切菜机开始干活。把前面那 3 个大文件 documents 喂进去，吐出来 63 个小块。
#
# chunks（变量）：把这 63 个小块存到 chunks 里，并打印出数量。
#
# 🎁 第三部分：交付（交接给下一道工序）
# python
# return chunks
# return（Python 内置关键字）：把切好的 63 个小块（chunks）作为这个函数的“最终产物”交出去。


#做实验
# 实例化时，你不传任何东西，它就默认用 settings 里的配置：
# loader1 = DataLoader()
# 如果你做实验，想测试 1000 的块大小，你只需要这样传参数：
# loader2 = DataLoader(chunk_size=1000)
# loader2 就会完全按照你的新要求去切分，而原来的 loader1 依然用默认的 500，互不干扰。