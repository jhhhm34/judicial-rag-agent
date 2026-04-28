# E:\LLM\Langchain-Chatchat\agent\config.py

# Chatchat API 地址（你本地跑起来的）
CHATCHAT_API_BASE = "http://127.0.0.1:7861"

# 知识库名称（你实际用的知识库名）
KNOWLEDGE_BASE_NAME = "judicial_structured"  # ← 改成你的知识库名

# LLM 配置（用智谱，你已有 key）
LLM_MODEL = "glm-4.5-air"
LLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY = "your-api-key-here"  # 请替换为您API Key

# 检索参数
TOP_K = 5
SCORE_THRESHOLD = 2.0