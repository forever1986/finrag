from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import config

BEG_MODEL = SentenceTransformer('/root/autodl-tmp/model/AI-ModelScope/bge-large-zh')
TOP_K = 5
LLM = ChatOpenAI(
        temperature=0.01,
        model="glm-4",
        openai_api_key="05f9fe2491eef6c3a7c2659957880e86.V7LJzSgrEK7raanZ",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
TOKENIZER = AutoTokenizer.from_pretrained(config.model_tokenizer_path, trust_remote_code=True)
