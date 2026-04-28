import json
from collections import Counter

# ============================================================
# 归并映射表
# 核心设计：专家评析、推荐理由、典型意义等全部归入"典型价值"
# 因为756条全是典型案例，这些段落都在描述"为什么典型"
# ============================================================
SLOT_MAP = {
    # ── 案件情况 ──────────────────────────────────────────
    '案情简介':             '案件情况',
    '案例背景':             '案件情况',
    '案例基本情况':         '案件情况',
    '基本情况':             '案件情况',
    '案情概要':             '案件情况',
    '案例概要':             '案件情况',
    '罪犯基本情况':         '案件情况',
    '社区矫正对象基本情况': '案件情况',
    '基本案情':             '案件情况',
    '案件基本情况':         '案件情况',
    '案件情况':             '案件情况',
    '社区服刑人员基本情况': '案件情况',

    # ── 典型价值 ──────────────────────────────────────────
    # 这是检索的核心，所有描述"为什么典型"的段落都在这里
    '专家评析':   '典型价值',
    '案例评析':   '典型价值',
    '案例点评':   '典型价值',
    '案件点评':   '典型价值',
    '法律分析':   '典型价值',
    '分析说明':   '典型价值',
    '案例注解':   '典型价值',
    '编辑点评':   '典型价值',
    '推荐理由':   '典型价值',
    '典型意义':   '典型价值',
    '焦点问题评析': '典型价值',

    # ── 处理过程 ──────────────────────────────────────────
    '调解过程':                   '处理过程',
    '鉴定过程':                   '处理过程',
    '争议焦点':                   '处理过程',
    '代理意见':                   '处理过程',
    '律师代理思路':               '处理过程',
    '调查与处理':                 '处理过程',
    '方案制定':                   '处理过程',
    '实施情况':                   '处理过程',
    '任务措施':                   '处理过程',
    '罪犯教育改造方案的制定和实施': '处理过程',
    '活动概况':                   '处理过程',
    '重点宣传内容':               '处理过程',
    '案件办理过程':               '处理过程',
    '监督过程':                   '处理过程',
    '维权过程':                   '处理过程',
    '鉴定情况':                   '处理过程',
    '出庭作证':                   '处理过程',
    '相关证据':                   '处理过程',
    '处理过程':                   '处理过程',

    # ── 处理结果 ──────────────────────────────────────────
    '调解结果':       '处理结果',
    '裁决结果':       '处理结果',
    '判决结果':       '处理结果',
    '鉴定意见':       '处理结果',
    '裁判文书':       '处理结果',
    '处理依据及结果': '处理结果',
    '应对措施':       '处理结果',
    '案件结果概述':   '处理结果',
    '教育改造成效':   '处理结果',
    '处理情况':       '处理结果',
    '处理结果':       '处理结果',
    '审理结果':       '处理结果',
    '案件办理结果':   '处理结果',
    '裁判结果':       '处理结果',
    '监督结果':       '处理结果',
    '维权结果':       '处理结果',
    '处罚决定':       '处理结果',
    '处罚依据':       '处理结果',
    '给予行政处罚的依据':                   '处理结果',
    '被撤销仲裁裁决原因':                   '处理结果',
    '鉴定意见采信情况':                     '处理结果',
    '法律依据':                             '处理结果',
    '违法违规事实':                         '处理结果',
    '对社区矫正对象依法实施监督管理情况':   '处理结果',
    '对社区矫正对象依法实施教育帮扶情况':   '处理结果',
    '依法决定和接收社区矫正对象情况':       '处理结果',
    '对社区服刑人员依法实施监督管理情况':   '处理结果',
    '对社区服刑人员依法实施监督管理取得的效果': '处理结果',
    '对社区服刑人员执行禁止令情况':         '处理结果',
    '对社区矫正对象依法解除和终止社区矫正的情况': '处理结果',

    # ── 扩展内容 ──────────────────────────────────────────
    '案例思考':         '扩展内容',
    '结语和建议':       '扩展内容',
    '相关法律法规解读': '扩展内容',
    '相关法律规定解读': '扩展内容',
    '特点和效果':       '扩展内容',
    '活动特点和效果':   '扩展内容',
    '公证书格式':       '扩展内容',
    '小结':             '扩展内容',
    '小结（或反思）':   '扩展内容',
}

# 不归类的段落（内容太特殊，不适合放入检索库）
SKIP_SECTIONS = {
    '不予法律援助原因',
    '律师执业权利被侵害情况',
}


# ============================================================
# 读取并合并 head 和 body
# ============================================================
def load_and_merge(body_path, head_path):
    print(f'正在读取 {body_path} ...')
    with open(body_path, encoding='utf-8') as f:
        bodies = json.load(f)

    print(f'正在读取 {head_path} ...')
    with open(head_path, encoding='utf-8') as f:
        heads = json.load(f)

    head_map = {h['caseid']: h for h in heads}

    merged = []
    for body in bodies:
        caseid = body['caseid']
        head = head_map.get(caseid, {})
        merged.append({
            'caseid':      caseid,
            'title':       body['title'],
            'typename':    head.get('typename', ''),
            'areaname':    head.get('areaname', ''),
            'publishdate': head.get('publishdate', ''),
            'bodys':       body.get('bodys', []),
        })

    print(f'合并完成：{len(merged)}条案例')
    return merged


# ============================================================
# 按类别切块
# ============================================================
def process_cases(cases):
    all_chunks = []
    skipped_sections = set()

    for case in cases:
        caseid      = case['caseid']
        title       = case['title']
        typename    = case['typename']
        areaname    = case['areaname']

        slots = {}
        for body in case['bodys']:
            section_name = body['title']

            # 明确跳过的段落
            if section_name in SKIP_SECTIONS:
                continue

            slot = SLOT_MAP.get(section_name)
            if slot is None:
                skipped_sections.add(section_name)
                continue

            content = '\n'.join(
                p.strip() for p in body['content'] if p.strip()
            )
            if not content:
                continue

            # 同一类别多个段落合并，中间加分隔线
            if slot not in slots:
                slots[slot] = content
            else:
                slots[slot] += '\n\n' + content

        for slot_name, slot_content in slots.items():
            chunk = {
                'caseid':    caseid,
                'title':     title,
                'typename':  typename,
                'areaname':  areaname,
                'slot':      slot_name,
                'text': (
                    f'案例标题：{title}\n'
                    f'案件类型：{typename}\n'
                    f'所属地区：{areaname}\n'
                    f'内容类别：{slot_name}\n'
                    f'---\n'
                    f'{slot_content}'
                )
            }
            all_chunks.append(chunk)

    return all_chunks, skipped_sections


# ============================================================
# 打印统计
# ============================================================
def print_statistics(cases, all_chunks, skipped_sections):
    print(f'\n========== 处理结果统计 ==========')
    print(f'总案例数：{len(cases)}')
    print(f'总chunk数：{len(all_chunks)}')
    print(f'平均每案例chunk数：{len(all_chunks) / len(cases):.1f}')

    print(f'\n各类别chunk数量：')
    slot_cnt = Counter(c['slot'] for c in all_chunks)
    for slot, cnt in slot_cnt.most_common():
        avg_len = sum(
            len(c['text']) for c in all_chunks if c['slot'] == slot
        ) // cnt
        print(f'  {slot}：{cnt}个，平均{avg_len}字')

    if skipped_sections:
        print(f'\n仍未归类的段落名：')
        for s in sorted(skipped_sections):
            print(f'  {s}')
    else:
        print(f'\n所有段落名已全部归类，没有遗漏')

    print(f'\n示例chunk（典型价值类别）：')
    sample = next(
        (c for c in all_chunks if c['slot'] == '典型价值'), None
    )
    if sample:
        print(sample['text'][:400] + '...')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':

    # 第一个数据集
    print('===== 处理 lawdb（756条）=====')
    cases1 = load_and_merge('lawdb_case_body.json', 'lawdb_case_head.json')
    chunks1, skipped1 = process_cases(cases1)
    print_statistics(cases1, chunks1, skipped1)
    with open('chunks_lawdb.json', 'w', encoding='utf-8') as f:
        json.dump(chunks1, f, ensure_ascii=False, indent=2)
    print(f'已保存到 chunks_lawdb.json')

    # 第二个数据集
    print('\n===== 处理 lawdb2（大数据集）=====')
    cases2 = load_and_merge('lawdb2_case_body.json', 'lawdb2_case_head.json')
    chunks2, skipped2 = process_cases(cases2)
    print_statistics(cases2, chunks2, skipped2)
    with open('chunks_lawdb2.json', 'w', encoding='utf-8') as f:
        json.dump(chunks2, f, ensure_ascii=False, indent=2)
    print(f'已保存到 chunks_lawdb2.json')

    print('\n全部完成，原始文件未被修改。')