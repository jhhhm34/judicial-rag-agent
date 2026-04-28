
from agent.reranker import rerank_docs
docs = [
    {'page_content': 'A', 'metadata': {'source': 'doc1'}},
    {'page_content': 'B', 'metadata': {'source': 'doc2'}},
    {'page_content': 'C', 'metadata': {'source': 'doc3'}},
]
docs[0]['page_content'] = '工伤保险制度是保障劳动者权益的重要制度'
docs[1]['page_content'] = '本案依据工伤保险条例第十四条第一款第六项认定为工伤'
docs[2]['page_content'] = '张某在超市购物时被误认为盗窃'
result = rerank_docs('涉及工伤保险条例第十四条的案例', docs, top_k=2)
for d in result:
    src = d['metadata']['source']
    txt = d['page_content'][:50]
    print(src, txt)
