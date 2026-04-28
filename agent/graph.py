# E:\LLM\Langchain-Chatchat\agent\graph.py

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from agent.config import LLM_MODEL, LLM_API_BASE, LLM_API_KEY
from agent.tools import case_retrieval, case_summarizer, statute_lookup, case_comparator

# ============================================================
# 1. 准备工具和 LLM
# ============================================================
tools = [case_retrieval, case_summarizer, statute_lookup, case_comparator]

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_base=LLM_API_BASE,
    openai_api_key=LLM_API_KEY,
    temperature=0.1,  # Agent 推理用低温度
    max_tokens=2000,
)

# bind_tools: 告诉 LLM "你有这4个工具可以调用"
# LLM 会在需要时返回 tool_calls，而不是纯文本
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 2. 系统提示词（定义 Agent 的行为边界）
# ============================================================
SYSTEM_PROMPT = """你是一个司法典型案例智能问答助手。你的任务是帮助用户查询、分析和判断司法典型案例。

你有以下工具可以使用：
1. case_retrieval: 从知识库检索案例（这是你获取信息的唯一途径）
2. case_summarizer: 对案例做结构化摘要
3. statute_lookup: 查询法条内容
4. case_comparator: 对比两个案例的异同

工作原则：
- 必须先用 case_retrieval 检索，再基于检索结果回答，绝不凭空编造
- 如果用户问的内容需要多步分析（如"对比"、"总结规律"），主动拆解为多步工具调用
- 如果检索结果中没有相关信息，如实回答"知识库中未找到相关案例"
- 引用案例时要注明来源
- 回答要专业、结构化、有理有据
"""


# ============================================================
# 3. 定义图的节点
# ============================================================
def agent_node(state: MessagesState):
    """
    Agent 节点：把消息历史发给 LLM，让它决定下一步。

    LLM 会返回两种情况：
    - 带 tool_calls → 说明它想调用工具
    - 不带 tool_calls → 说明它准备好直接回答了
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    """
    路由函数：决定 Agent 下一步去哪。

    这是 ReAct 循环的控制器：
    - 有 tool_calls → 去 tools 节点执行工具
    - 没有 tool_calls → 结束（Agent 认为可以回答了）
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ============================================================
# 4. 组装图
# ============================================================
def build_agent():
    """
    构建 Agent 的状态图。

    图结构：
        START → agent_node → (有tool_calls?) → tools → agent_node → ...
                            → (无tool_calls?) → END

    这个循环就是 ReAct 的 Think-Act-Observe 循环：
    - Think: agent_node 里 LLM 思考要做什么
    - Act: tools 节点执行工具
    - Observe: 工具结果作为 ToolMessage 回到 agent_node
    """
    graph = StateGraph(MessagesState)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # 设置入口
    graph.set_entry_point("agent")

    # 添加边
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")  # 工具执行完 → 回到 agent 继续思考

    return graph.compile()


# 全局 agent 实例
agent = build_agent()