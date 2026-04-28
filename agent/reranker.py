# E:\LLM\Langchain-Chatchat\agent\reranker.py
"""
Reranker 精排模块
使用 bge-reranker-v2-m3 Cross-Encoder 对检索结果做精排

原理：
  1. 向量检索（Bi-Encoder）：query 和 doc 分开编码，算向量距离
     - 快（毫秒级），但 query 和 doc 之间没有直接交互
  2. Reranker（Cross-Encoder）：query 和 doc 拼在一起，同时送入模型
     - 慢（秒级），但模型能逐词理解 query 和 doc 的关系
     - Self-Attention 让 "第十四条"(query) 和 "第十四条第一款"(doc) 产生强关联

流程：
  粗召回（BM25+FAISS，取 top 10）→ 精排（Reranker，从 10 选 5）→ 生成回答
"""

from sentence_transformers import CrossEncoder
import os

# 全局变量，避免每次调用都重新加载模型（约 3 秒）
_reranker_model = None


def get_reranker():
    """懒加载 Reranker 模型（只在第一次调用时加载）"""
    global _reranker_model
    if _reranker_model is None:
        print("[Reranker] 正在加载 bge-reranker-v2-m3 模型（首次约需 30 秒）...")

        # 如果 HuggingFace 下载慢，可以改成本地路径
        model_name = "BAAI/bge-reranker-v2-m3"

        # 也可以用本地已下载的路径：
        # model_name = r"E:\models\bge-reranker-v2-m3"

        _reranker_model = CrossEncoder(
            model_name,
            max_length=512,   # 最大输入长度（query + doc 拼接后的总长度）
            device="cuda"   # 如果有 GPU 取消注释这行
        )
        print("[Reranker] 模型加载完成")
    return _reranker_model


def rerank_docs(query: str, docs: list, top_k: int = 5) -> list:
    """
    用 bge-reranker-v2-m3 对候选文档做精排。

    Args:
        query: 用户问题
        docs: 候选文档列表（每个元素是 dict，含 page_content 字段）
        top_k: 返回前 top_k 个最相关的文档

    Returns:
        精排后的文档列表（按相关性从高到低）

    工作原理：
        1. 把 (query, doc_content) 组成配对列表
        2. Cross-Encoder 对每一对独立打分（0-1 之间）
        3. 按分数从高到低排序
        4. 返回 top_k 个
    """
    if len(docs) <= top_k:
        return docs

    reranker = get_reranker()

    # 构造 (query, document) 配对
    # 截断到 500 字：Cross-Encoder 的 max_length=512 tokens
    # 中文大约 1 字 = 1-2 tokens，500 字约 500-1000 tokens
    # query 约占 50 tokens，留 450 给 document
    pairs = []
    for doc in docs:
        content = doc.get("page_content", "")[:450]
        pairs.append((query, content))

    # 逐对打分
    # scores 是一个 numpy 数组，如 [0.95, 0.31, 0.78, 0.12, ...]
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"[Reranker] 打分失败，回退原始排序: {e}")
        return docs[:top_k]

    # 按分数排序
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # 打印排序结果（调试用，生产环境可以注释掉）
    print(f"[Reranker] 精排完成，top {top_k} scores:", end=" ")
    for doc, score in scored_docs[:top_k]:
        source = doc.get("metadata", {}).get("source", "?")
        print(f"{source}({score:.3f})", end=" ")
    print()

    return [doc for doc, score in scored_docs[:top_k]]