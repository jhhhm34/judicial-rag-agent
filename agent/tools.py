# E:\LLM\Langchain-Chatchat\agent\tools.py

import requests
import json
from langchain_core.tools import tool
from agent.config import CHATCHAT_API_BASE, KNOWLEDGE_BASE_NAME, TOP_K, SCORE_THRESHOLD
from agent.reranker import rerank_docs

# @tool
# def case_retrieval(query: str) -> str:
#     """从司法典型案例知识库中检索相关案例。
#     当需要查找具体案例、查询某类案例是否存在时，使用此工具。
#
#     Args:
#         query: 检索问题，例如"农民工工伤赔偿典型案例"、"未成年人犯罪辩护"
#     """
#     # 调用你的 Chatchat search_docs API
#     # 这个接口就是你改过的混合检索（BM25+FAISS+RRF）
#     try:
#         response = requests.post(
#             f"{CHATCHAT_API_BASE}/knowledge_base/search_docs",
#             json={
#                 "query": query,
#                 "knowledge_base_name": KNOWLEDGE_BASE_NAME,
#                 "top_k": TOP_K,
#                 "score_threshold": SCORE_THRESHOLD,
#             },
#             timeout=30,
#         )
#         response.raise_for_status()
#         docs = response.json()
#     except Exception as e:
#         return f"检索失败: {str(e)}"
#
#     if not docs:
#         return "未在知识库中找到相关案例。"
#
#     results = []
#     for i, doc in enumerate(docs):
#         content = doc.get("page_content", "")
#         source = doc.get("metadata", {}).get("source", "未知来源")
#         # 截断过长的内容，避免超出 LLM 上下文
#         if len(content) > 1500:
#             content = content[:1500] + "...(内容已截断)"
#         results.append(f"【案例{i + 1}】(来源: {source})\n{content}")
#
#     return "\n\n---\n\n".join(results)

# agent/tools.py 中的 case_retrieval 函数，修改后：

@tool
def case_retrieval(query: str) -> str:
    """从司法典型案例知识库中检索相关案例..."""
    from agent.config import LLM_API_BASE, LLM_API_KEY

    # 第一步：调 Chatchat 混合检索，多取一些候选
    try:
        response = requests.post(
            f"{CHATCHAT_API_BASE}/knowledge_base/search_docs",
            json={
                "query": query,
                "knowledge_base_name": KNOWLEDGE_BASE_NAME,
                "top_k": 10,  # 多取一倍，给 Reranker 筛选空间
                "score_threshold": SCORE_THRESHOLD,
            },
            timeout=30,
        )
        response.raise_for_status()
        docs = response.json()
    except Exception as e:
        return f"检索失败: {str(e)}"

    if not docs:
        return "未在知识库中找到相关案例。"

    # 第二步：Reranker 精排
    docs = rerank_docs(query, docs, top_k=5)


    # 第三步：格式化返回
    results = []
    for i, doc in enumerate(docs):
        content = doc.get("page_content", "")
        source = doc.get("metadata", {}).get("source", "未知来源")
        if len(content) > 1500:
            content = content[:1500] + "...(已截断)"
        results.append(f"【案例{i + 1}】(来源: {source})\n{content}")

    return "\n\n---\n\n".join(results)


# def rerank_docs(query: str, docs: list, top_k: int = 5) -> list:
#     """
#     用智谱 API 做 Reranker 精排。
#
#     原理：把 query 和每个 doc 拼在一起，让 LLM 打一个相关性分数，
#     然后按分数从高到低排序，只返回 top_k 个。
#
#     这是一个轻量级的 Reranker 实现。
#     生产环境应该用专用的 Cross-Encoder 模型（如 bge-reranker-v2-m3）。
#     """
#     from agent.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
#
#     if len(docs) <= top_k:
#         return docs
#
#     # 构造打分请求
#     doc_texts = []
#     for doc in docs:
#         content = doc.get("page_content", "")[:500]  # 截断，节省 token
#         doc_texts.append(content)
#
#     # 让 LLM 对每个文档打相关性分数
#     scoring_prompt = f"""你是一个文档相关性评估专家。给定一个查询和多个候选文档，请为每个文档打一个0-10的相关性分数。
#
# 查询：{query}
#
# 请严格按以下 JSON 格式输出，只输出 JSON，不要其他内容：
# {{"scores": [分数1, 分数2, ...]}}
#
# 候选文档：
# """
#     for i, text in enumerate(doc_texts):
#         scoring_prompt += f"\n文档{i + 1}：{text}\n"
#
#     try:
#         response = requests.post(
#             f"{LLM_API_BASE}/chat/completions",
#             headers={"Authorization": f"Bearer {LLM_API_KEY}"},
#             json={
#                 "model": LLM_MODEL,
#                 "messages": [{"role": "user", "content": scoring_prompt}],
#                 "temperature": 0.0,  # 打分要确定性
#                 "max_tokens": 200,
#             },
#             timeout=30,
#         )
#         response.raise_for_status()
#         result_text = response.json()["choices"][0]["message"]["content"]
#
#         # 解析分数
#         import json, re
#         json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
#         if json_match:
#             scores = json.loads(json_match.group())["scores"]
#         else:
#             return docs[:top_k]  # 解析失败，退回原始排序
#
#         # 按分数排序
#         scored_docs = list(zip(docs, scores))
#         scored_docs.sort(key=lambda x: x[1], reverse=True)
#         return [doc for doc, score in scored_docs[:top_k]]
#
#     except Exception:
#         return docs[:top_k]  # 出错时退回原始排序


@tool
def case_summarizer(case_text: str) -> str:
    """对一段司法案例文本进行结构化摘要。
    提取案件概要、裁判要旨、典型价值、关键法条四个维度。
    当拿到检索结果后需要深入理解某个案例时使用。

    Args:
        case_text: 案例原文内容（从 case_retrieval 获取的结果）
    """
    prompt = f"""请对以下司法案例进行结构化摘要，严格按以下四个维度输出：

1. 案件概要：用2-3句话概括案件事实
2. 裁判要旨：核心裁判观点或调解结论
3. 典型价值：该案例的示范意义和参考价值
4. 关键法条：涉及的主要法律条文

案例内容：
{case_text[:3000]}
"""
    try:
        response = requests.post(
            f"{CHATCHAT_API_BASE}/v1/chat/completions",
            json={
                "model": "glm-4.5-air",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"摘要生成失败: {str(e)}"


@tool
def statute_lookup(statute_name: str) -> str:
    """查询法律法条的具体内容。
    当需要引用具体法律条文来支持分析时使用。

    Args:
        statute_name: 法条名称，如"工伤保险条例第十四条"、"消费者权益保护法第八条"
    """
    # 用 Chatchat 的知识库检索来找法条
    # 也可以换成独立的法条 JSON 文件
    try:
        response = requests.post(
            f"{CHATCHAT_API_BASE}/knowledge_base/search_docs",
            json={
                "query": statute_name,
                "knowledge_base_name": KNOWLEDGE_BASE_NAME,
                "top_k": 3,
                "score_threshold": SCORE_THRESHOLD,
            },
            timeout=30,
        )
        response.raise_for_status()
        docs = response.json()
    except Exception as e:
        return f"法条查询失败: {str(e)}"

    if not docs:
        return f"未找到与 '{statute_name}' 相关的法条内容。"

    results = []
    for doc in docs[:2]:
        content = doc.get("page_content", "")[:800]
        results.append(content)
    return "\n\n".join(results)


@tool
def case_comparator(aspect: str, case_a_summary: str, case_b_summary: str) -> str:
    """对比两个案例在特定维度上的异同。
    当用户要求跨案例对比分析时使用。

    Args:
        aspect: 对比的维度，如"赔偿金额"、"争议焦点"、"处理方式"
        case_a_summary: 第一个案例的摘要或关键内容
        case_b_summary: 第二个案例的摘要或关键内容
    """
    prompt = f"""请从"{aspect}"的角度，对比以下两个司法案例的异同：

【案例A】
{case_a_summary[:1500]}

【案例B】
{case_b_summary[:1500]}

请从以下维度进行对比分析：
1. 相同点
2. 不同点
3. 结论与启示
"""
    try:
        response = requests.post(
            f"{CHATCHAT_API_BASE}/v1/chat/completions",
            json={
                "model": "glm-4.5-air",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"对比分析失败: {str(e)}"