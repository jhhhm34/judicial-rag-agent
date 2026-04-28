# E:\LLM\Langchain-Chatchat\app.py
# 这是 Agent 的独立可视化前端，运行在 agent 环境中

import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import agent

# ============================================================
# 第一部分：页面基础配置
# ============================================================
# page_title: 浏览器标签页的标题
# layout="wide": 页面用宽屏布局，聊天内容不会挤在中间窄条里
st.set_page_config(page_title="司法案例智能问答 Agent", layout="wide")

# 页面顶部的标题和副标题
st.title("司法典型案例智能问答 Agent")
st.caption("基于 LangGraph ReAct + RAG 混合检索 | 东南大学研究生科研项目")

# ============================================================
# 第二部分：侧边栏（展示系统信息）
# ============================================================
with st.sidebar:
    st.header("系统架构")
    st.markdown("""
    **Agent 层（LangGraph ReAct）**
    - 🔧 case_retrieval: 案例检索
    - 📝 case_summarizer: 结构化摘要
    - 📖 statute_lookup: 法条查询
    - ⚖️ case_comparator: 案例对比

    **RAG 层（Chatchat）**
    - 混合检索: BM25 + FAISS (RRF)
    - Embedding: bge-large-zh-v1.5
    - LLM: GLM-4-air
    """)
    st.divider()
    st.markdown("**示例问题：**")
    if st.button("农民工讨薪典型案例"):
        st.session_state.preset = "有没有农民工讨薪的法律援助典型案例？"
    if st.button("工伤案件典型性判断"):
        st.session_state.preset = "请判断以下案件是否具有典型性：某工厂工人李某在工作中因机器故障失去右手手指，工厂以李某操作不当为由拒绝赔偿工伤，李某申请劳动仲裁，最终获赔18万元。"
    if st.button("跨案例对比分析"):
        st.session_state.preset = "对比农民工工伤和交通事故两类案例的赔偿标准差异"

# ============================================================
# 第三部分：聊天历史管理
# ============================================================
# Streamlit 每次用户操作都会重新运行整个脚本
# 所以需要用 session_state 保存聊天历史，否则刷新就没了
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示之前的聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果这条消息有推理链，也要展示
        if "reasoning" in msg:
            with st.expander("查看推理过程", expanded=False):
                for step in msg["reasoning"]:
                    st.markdown(step)

# ============================================================
# 第四部分：处理用户输入
# ============================================================
# 检查是否有侧边栏的预设问题
user_input = None
if "preset" in st.session_state:
    user_input = st.session_state.preset
    del st.session_state.preset

# st.chat_input 会在页面底部显示一个输入框
if prompt := (user_input or st.chat_input("请输入您的问题...")):

    # 4.1 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4.2 调用 Agent 并展示推理过程
    with st.chat_message("assistant"):
        reasoning_steps = []  # 收集推理链

        # st.status 会显示一个可展开的状态框
        # 用户能实时看到 Agent 在做什么
        with st.status("Agent 正在推理...", expanded=True) as status:

            # ---- 核心调用 ----
            result = agent.invoke({
                "messages": [HumanMessage(content=prompt)]
            })
            # ---- 调用结束 ----

            # 遍历消息历史，展示每一步
            step_count = 0
            for msg in result["messages"]:
                msg_type = msg.__class__.__name__

                if msg_type == "AIMessage" and msg.tool_calls:
                    # LLM 决定调用工具
                    for tc in msg.tool_calls:
                        step_count += 1
                        step_text = f"**Step {step_count}** 🔧 调用工具: `{tc['name']}`"
                        st.markdown(step_text)

                        # 显示工具的输入参数（折叠）
                        args_str = str(tc["args"])
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "..."
                        st.code(args_str, language="json")

                        reasoning_steps.append(step_text)

                elif msg_type == "ToolMessage":
                    # 工具返回的结果
                    preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    with st.expander(f"📄 工具返回结果（点击展开）", expanded=False):
                        st.text(msg.content[:500])
                    reasoning_steps.append(f"📄 工具返回: {preview}")

            status.update(label=f"推理完成（共 {step_count} 次工具调用）", state="complete")

        # 4.3 显示最终回答
        final_answer = result["messages"][-1].content
        st.markdown(final_answer)

        # 4.4 保存到聊天历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "reasoning": reasoning_steps,
        })