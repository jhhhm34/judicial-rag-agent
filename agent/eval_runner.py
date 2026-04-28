# E:\LLM\Langchain-Chatchat\agent\eval_runner.py
# 评估测试脚本：一键运行 C / D / E 三种配置
#
# 使用方法：
#   conda activate agent
#   cd E:\LLM\Langchain-Chatchat
#
#   一键跑完CDE三种模式：
#     python -m agent.eval_runner --mode C D E
#
#   只跑某个维度：
#     python -m agent.eval_runner --mode C D E --dim 多步推理
#
#   只跑某道题：
#     python -m agent.eval_runner --mode C --qid Q33
#
#   指定知识库（测Baseline）：
#     python -m agent.eval_runner --mode C --kb judicial_baseline

import sys
import os
import json
import argparse
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from agent.config import CHATCHAT_API_BASE, KNOWLEDGE_BASE_NAME, LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from agent.reranker import rerank_docs

# 结果保存目录
OUTPUT_DIR = r"E:\LLM\Langchain-Chatchat\eval_result"

# ============================================================
# 评估题目（全部50题）
# ============================================================
QUESTIONS = {
    # 维度一：精确检索 (8题)
    "Q1": "有没有涉及盲人或视障人士金融服务权益的调解案例？",
    "Q2": "有没有农民工讨薪的法律援助典型案例？",
    "Q3": "有没有涉及工伤赔偿的仲裁典型案例？",
    "Q4": "有没有涉及未成年人犯罪辩护的法律援助案例？",
    "Q5": "有没有房屋买卖合同纠纷中关于面积缩水的典型案例？",
    "Q6": "有没有涉及《保障农民工工资支付条例》的法律援助案例？",
    "Q7": "知识库中有没有湖南省岳阳市的工伤保险待遇纠纷案例？",
    "Q8": "有没有涉及司法确认程序的人民调解案例？",

    # 维度二：语义理解 (8题)
    "Q9": "老人和子女之间因为钱产生纠纷，有什么典型案例？",
    "Q10": "公司不给员工买保险，出了事故不认账，有没有相关案例？",
    "Q11": "有没有弱势群体维权成功的典型案例？",
    "Q12": "租房押金不退，房东各种理由拖延，有没有类似案例？",
    "Q13": "消费者买到假冒伪劣商品，商家不赔偿，怎么维权？",
    "Q14": "有没有打工的人受伤了，老板不管不问的案例？",
    "Q15": "小孩犯了法，家里没钱请律师，国家能帮忙吗？",
    "Q16": "有没有那种两口子闹离婚，为了孩子抚养权打官司的案例？",

    # 维度三：典型性判断 (8题)
    "Q17": "请判断以下案件是否具有典型性：某工厂工人李某在工作中因机器故障失去右手手指，工厂以李某操作不当为由拒绝赔偿工伤，李某申请劳动仲裁，最终获赔18万元。请结合知识库中的典型案例说明判断依据。",
    "Q18": "请判断以下案件是否具有典型性：75岁老人王某将毕生积蓄50万存入银行，被银行员工误导购买了高风险理财产品，产品亏损后银行拒绝赔偿，老人申请调解，要求退还本金。",
    "Q19": "请判断以下案件是否具有典型性：外卖骑手张某送餐途中发生交通事故，平台以张某是'个人合作商'而非员工为由拒绝承担工伤赔偿责任。",
    "Q20": "请判断以下案件是否具有典型性：某小区业主发现开发商交付的房屋实际面积比合同少8平方米，开发商拒绝退款或补偿。",
    "Q21": "请判断以下案件是否具有典型性：聋哑人赵某在超市购物时，因无法与收银员正常沟通被误认为盗窃，遭到保安强制扣押，赵某要求超市公开道歉并赔偿。",
    "Q22": "请判断以下案件是否具有典型性：某快递员王某在分拣包裹时腰部受伤，快递公司以王某是'灵活用工'人员为由拒绝认定工伤，王某申请劳动仲裁。请结合知识库中的典型案例说明判断依据。",
    "Q23": "请判断以下案件是否具有典型性：农村老人张某（78岁）的三个子女因赡养费分担问题发生纠纷，大儿子以'父亲偏心'为由拒绝赡养，村委会调解未果后申请法律援助。",
    "Q24": "请判断以下案件是否具有典型性：某中学生在学校体育课上因设备老化导致骨折，学校以已购买校方责任险为由拒绝额外赔偿，家长申请调解。",

    # 维度四：抗干扰/拒答 (8题)
    "Q25": "有没有涉及加密货币诈骗的典型司法案例？",
    "Q26": "特斯拉汽车质量纠纷有没有典型调解案例？",
    "Q27": "网络主播因直播内容违规被平台封号，有哪些维权典型案例？",
    "Q28": "跨国婚姻纠纷中涉及外籍人士的典型调解案例有哪些？",
    "Q29": "人工智能侵权的典型司法案例有哪些？",
    "Q30": "有没有涉及区块链智能合约纠纷的典型案例？",
    "Q31": "请分析知识库中所有案例的平均赔偿金额是多少？",
    "Q32": "最高人民法院2024年发布的最新司法解释有哪些？",

    # 维度五：多步推理 (10题，Agent专属)
    "Q33": "对比农民工工伤赔偿和交通事故赔偿两类案例在赔偿标准上的差异。",
    "Q34": "检索所有涉及未成年人权益保护的案例，总结法律援助介入的共同模式。",
    "Q35": "找一个农民工讨薪案例和一个工伤赔偿案例，对比两种维权路径的成本和效率。",
    "Q36": "知识库中有哪些案例涉及《工伤保险条例》第十四条？请列举并说明该法条在不同案例中的适用差异。",
    "Q37": "一位法律工作者想了解'用人单位未缴纳社保导致的工伤赔偿纠纷'的典型处理方式。请检索相关案例，提取共同的争议焦点和解决路径。",
    "Q38": "请分别找一个调解成功和一个仲裁成功的劳动争议案例，对比两种解决方式的优缺点。",
    "Q39": "某基层法律援助中心想编写一份'农民工讨薪维权指南'，请基于知识库中的典型案例提炼3-5条实用建议。",
    "Q40": "请判断以下案件是否典型，并找到知识库中最相似的案例进行对比：某外卖骑手送餐途中被电动车撞伤，平台以骑手未购买商业保险为由拒绝赔偿。",
    "Q41": "比较知识库中'行业性、专业性人民调解案例'和'各类法律援助典型案例'两类案例在处理流程上的系统性差异。",
    "Q42": "一位法官希望了解：在工伤赔偿案例中，劳动关系认定通常依据哪些证据？请结合具体案例说明。",

    # 维度六：一致性检验 (8题 = 4对)
    "Q43a": "有没有盲人办银行业务被拒的案例？",
    "Q43b": "视障人士金融服务权益方面有什么典型调解案例？",
    "Q44a": "工人受伤老板不赔怎么办？",
    "Q44b": "劳动者因工致残，用人单位拒绝承担工伤保险责任的法律援助案例有哪些？",
    "Q45a": "请判断以下案件是否具有典型性：工厂工人因设备故障失去手指，工厂拒赔。",
    "Q45b": "请判断以下案件是否具有典型性：某制造企业员工操作机器时因设备缺陷导致手部伤残，企业以员工违规操作为由拒绝赔偿。",
    "Q46a": "有没有外卖骑手工伤赔偿的案例？",
    "Q46b": "知识库中关于平台经济从业者劳动权益保障的典型案例有哪些？",
}

# 维度-题号映射
DIM_RANGES = {
    "精确检索": [f"Q{i}" for i in range(1, 9)],
    "语义理解": [f"Q{i}" for i in range(9, 17)],
    "典型性判断": [f"Q{i}" for i in range(17, 25)],
    "抗干扰": [f"Q{i}" for i in range(25, 33)],
    "多步推理": [f"Q{i}" for i in range(33, 43)],
    "一致性": ["Q43a", "Q43b", "Q44a", "Q44b", "Q45a", "Q45b", "Q46a", "Q46b"],
}





# ============================================================
# 模式 C：纯 RAG（混合检索，无 Reranker）
# ============================================================
def run_mode_c(query, kb_name):
    """C模式：检索 + 直接生成（无Reranker）"""
    try:
        search_resp = requests.post(
            f"{CHATCHAT_API_BASE}/knowledge_base/search_docs",
            json={
                "query": query,
                "knowledge_base_name": kb_name,
                "top_k": 5,
                "score_threshold": 2.0,
            },
            timeout=30,
        )
        search_resp.raise_for_status()
        docs = search_resp.json()
    except Exception as e:
        return f"检索失败: {str(e)}"

    if not docs:
        return "未在知识库中找到相关案例。"

    context = "\n\n".join([d.get("page_content", "")[:1500] for d in docs[:5]])
    prompt = (
        f"【指令】根据已知信息，简洁和专业的来回答问题。"
        f"如果无法从中得到答案，请说\"根据已知信息无法回答该问题\"，"
        f"不允许在答案中添加编造成分，答案请使用中文。\n\n"
        f"【已知信息】{context}\n\n"
        f"【问题】{query}"
    )

    try:
        llm_resp = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=120,
        )
        llm_resp.raise_for_status()
        return llm_resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM生成失败: {str(e)}"


# ============================================================
# 模式 D：RAG + Reranker（检索 → 精排 → 生成）
# ============================================================
def run_mode_d(query, kb_name):
    """先检索，再 Reranker 精排，再调 LLM 生成"""
    try:
        search_resp = requests.post(
            f"{CHATCHAT_API_BASE}/knowledge_base/search_docs",
            json={
                "query": query,
                "knowledge_base_name": kb_name,
                "top_k": 10,
                "score_threshold": 2.0,
            },
            timeout=30,
        )
        search_resp.raise_for_status()
        docs = search_resp.json()
    except Exception as e:
        return f"检索失败: {str(e)}"

    if not docs:
        return "未在知识库中找到相关案例。"

    docs = rerank_docs(query, docs, top_k=5)

    context = "\n\n".join([d.get("page_content", "")[:1500] for d in docs])
    prompt = (
        f"【指令】根据已知信息，简洁和专业的来回答问题。"
        f"如果无法从中得到答案，请说\"根据已知信息无法回答该问题\"，"
        f"不允许在答案中添加编造成分，答案请使用中文。\n\n"
        f"【已知信息】{context}\n\n"
        f"【问题】{query}"
    )

    try:
        llm_resp = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=300,
        )
        llm_resp.raise_for_status()
        return llm_resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM生成失败: {str(e)}"


# ============================================================
# 模式 E：完整 Agent（LangGraph ReAct）
# ============================================================
def run_mode_e(query):
    """调用完整的 LangGraph Agent"""
    from langchain_core.messages import HumanMessage
    from agent.graph import agent

    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    final = result["messages"][-1].content

    tool_calls = []
    for msg in result["messages"]:
        name = msg.__class__.__name__
        if name == "AIMessage" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({str(tc['args'])[:80]})")

    return final, tool_calls


# ============================================================
# 运行单道题
# ============================================================
def run_single(qid, question, mode, kb_name):
    print(f"\n{'='*60}")
    print(f"[{qid}] 模式={mode} | {question[:60]}...")
    print(f"{'='*60}")

    start = time.time()
    tool_calls = []

    if mode == "C":
        answer = run_mode_c(question, kb_name)
    elif mode == "D":
        answer = run_mode_d(question, kb_name)
    elif mode == "E":
        answer, tool_calls = run_mode_e(question)
    else:
        answer = "未知模式"

    elapsed = time.time() - start

    if tool_calls:
        print(f"  工具调用链: {' -> '.join(tool_calls)}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  回答: {answer[:300]}...")
    print()

    return {
        "qid": qid,
        "mode": mode,
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
        "elapsed": round(elapsed, 1),
    }


# ============================================================
# 保存结果
# ============================================================
def save_results(results, mode, is_temp=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_temp" if is_temp else ""
    path = os.path.join(OUTPUT_DIR, f"eval_results_new_{mode}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="评估测试运行器")
    parser.add_argument("--mode", required=True, nargs="+", choices=["C", "D", "E"],
                        help="测试模式，可多选。C=混合检索RAG, D=+Reranker, E=+Agent")
    parser.add_argument("--qid", default=None,
                        help="只测一道题，如 --qid Q33")
    parser.add_argument("--dim", default=None,
                        choices=["精确检索", "语义理解", "典型性判断", "抗干扰", "多步推理", "一致性"],
                        help="只测某个维度")
    parser.add_argument("--kb", default=None,
                        help="知识库名（默认用config.py里的）")
    args = parser.parse_args()

    kb_name = args.kb or KNOWLEDGE_BASE_NAME

    if args.qid:
        test_ids = [args.qid]
    elif args.dim:
        test_ids = DIM_RANGES[args.dim]
    else:
        test_ids = list(QUESTIONS.keys())

    for mode in args.mode:
        print(f"\n{'#'*60}")
        print(f"# 开始测试模式: {mode}")
        print(f"# 知识库: {kb_name}")
        print(f"# 题目数: {len(test_ids)}")
        print(f"{'#'*60}\n")

        results = []
        for i, qid in enumerate(test_ids):
            if qid not in QUESTIONS:
                print(f"  [跳过] {qid} 不在题库中")
                continue

            try:
                result = run_single(qid, QUESTIONS[qid], mode, kb_name)
                results.append(result)
            except Exception as e:
                print(f"  [错误] {qid} 执行失败: {e}")
                results.append({
                    "qid": qid,
                    "mode": mode,
                    "question": QUESTIONS[qid],
                    "answer": f"执行失败: {str(e)}",
                    "tool_calls": [],
                    "elapsed": 0,
                })

            if (i + 1) % 5 == 0:
                save_results(results, mode, is_temp=True)
                print(f"  [自动保存] 模式{mode} 已完成 {i+1}/{len(test_ids)} 题")

        output_path = save_results(results, mode, is_temp=False)

        success = sum(1 for r in results if not r["answer"].startswith("执行失败"))
        failed = len(results) - success
        total_time = sum(r["elapsed"] for r in results)

        print(f"\n{'='*60}")
        print(f"模式 {mode} 测试完成！")
        print(f"  成功: {success} 题 / 失败: {failed} 题")
        print(f"  总耗时: {total_time:.0f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"  结果保存: {output_path}")
        print(f"{'='*60}")

    print(f"\n全部测试完成！请打开 {OUTPUT_DIR} 查看结果文件。")
    print(f"对照评分标准为每道题打 0-5 分，填入 xlsx 的 Sheet1 和 Sheet3。")


if __name__ == "__main__":
    main()
