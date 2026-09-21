# app.py
import streamlit as st
from core.llm import get_rag_chain

# 设置网页标题
st.set_page_config(page_title="我的知识库助手", page_icon="📚")
st.title("📚 个人知识库问答助手")


# 【非常重要】@st.cache_resource 保证模型和向量库只加载一次
# 如果没这个，你在网页打一个字，它就会重新加载一次1.3G的模型，网页会卡死
@st.cache_resource
def load_chain():
    return get_rag_chain()


rag_query = load_chain()

# 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 每次刷新网页，重绘历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 接收用户在底部输入框的提问
if prompt := st.chat_input("请输入你的问题："):
    # 1. 在界面上显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. 显示“正在思考”的动画，并调用RAG
    with st.spinner("正在检索并思考..."):
        answer, docs = rag_query(prompt)

    # 3. 在界面上显示AI回答
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # 4. 【加分项】把参考来源折叠展示，方便溯源
    with st.expander("查看参考来源"):
        for i, doc in enumerate(docs):
            st.write(f"**片段 {i + 1}**: {doc.page_content[:150]}...")