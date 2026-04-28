# E:\LLM\Langchain-Chatchat\agent\run.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from agent.graph import agent


def run_agent(question: str):
    """运行 Agent 并打印每一步的推理过程"""
    print(f"\n{'=' * 60}")
    print(f"用户问题: {question}")
    print(f"{'=' * 60}\n")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # 打印推理链
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            print(f"[Step {i}] 👤 用户: {msg.content[:100]}")

        elif msg_type == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = str(tc["args"])[:150]
                    print(f"[Step {i}] 🤖 Agent 决定调用工具: {tc['name']}({args_str})")
            else:
                print(f"[Step {i}] 🤖 Agent 最终回答:")
                print(f"    {msg.content[:500]}")

        elif msg_type == "ToolMessage":
            print(f"[Step {i}] 🔧 工具返回: {msg.content[:200]}...")

        print()

    return result


if __name__ == "__main__":
    # ============================================================
    # 测试用例：从简单到复杂
    # ============================================================

    # 测试1：简单检索（应该只调一次 case_retrieval）
    # run_agent("有没有农民工讨薪的法律援助典型案例？")

    # 测试2：需要多步推理的复杂问题
    # run_agent("对比农民工工伤和交通事故两类案例的赔偿标准差异")

    # 测试3：典型性判断（你评估集里的题目）
    run_agent("请判断以下案件是否具有典型性：某工厂工人李某在工作中因机器故障失去右手手指，工厂以李某操作不当为由拒绝赔偿工伤，李某申请劳动仲裁，最终获赔18万元。")