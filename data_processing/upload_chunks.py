import json
import requests
import os
import time

KB_NAME = "judicial_structured"
API_BASE = "http://127.0.0.1:7861"
CHUNKS_FILE = "chunks_lawdb.json"
TEMP_DIR = "temp_chunks"

def upload_chunks():
    # 读取chunks
    print(f"正在读取 {CHUNKS_FILE} ...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"共 {len(chunks)} 个chunk")

    # 建临时目录存txt文件
    os.makedirs(TEMP_DIR, exist_ok=True)

    success = 0
    failed = 0

    # 每次上传一批chunk，每批5个
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # 把这批chunk各自写成一个txt文件
        temp_files = []
        file_handles = []
        for j, chunk in enumerate(batch):
            filename = f"{TEMP_DIR}/chunk_{i+j}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chunk["text"])
            temp_files.append(filename)

        # 打开所有文件准备上传
        try:
            files = []
            handles = []
            for filename in temp_files:
                h = open(filename, "rb")
                handles.append(h)
                files.append(
                    ("files", (os.path.basename(filename), h, "text/plain"))
                )

            r = requests.post(
                f"{API_BASE}/knowledge_base/upload_docs",
                data={
                    "knowledge_base_name": KB_NAME,
                    "override": "true",
                    # 把切块大小设置很大，避免Chatchat再次切割
                    "chunk_size": "2000",
                    "chunk_overlap": "0",
                },
                files=files,
                timeout=120
            )

            # 关闭所有文件句柄
            for h in handles:
                h.close()

            if r.status_code == 200:
                result = r.json()
                if result.get("code") == 200:
                    success += len(batch)
                else:
                    print(f"第{i//batch_size+1}批失败：{result.get('msg')}")
                    failed += len(batch)
            else:
                print(f"第{i//batch_size+1}批HTTP错误：{r.status_code}，{r.text[:100]}")
                failed += len(batch)

        except Exception as e:
            print(f"第{i//batch_size+1}批异常：{e}")
            failed += len(batch)
            for h in handles:
                try:
                    h.close()
                except:
                    pass

        # 打印进度，每20批打印一次
        done = min(i + batch_size, len(chunks))
        if (i // batch_size) % 20 == 0:
            print(f"进度：{done}/{len(chunks)}，成功{success}，失败{failed}")

        time.sleep(0.3)

    print(f"\n上传完成！成功：{success}，失败：{failed}")
    print(f"临时文件保存在 {TEMP_DIR} 目录，确认无误后可以删除")

if __name__ == "__main__":
    upload_chunks()