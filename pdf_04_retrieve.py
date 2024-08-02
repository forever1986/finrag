import os
import json
import faiss
import numpy
import config
import pickle
import requests
import pandas as pd
from util import prompts
from rank_bm25 import BM25Okapi
from requests.adapters import HTTPAdapter
from util.instances import LLM, BEG_MODEL
from langchain_core.prompts import ChatPromptTemplate


class Query:

    def __init__(self, question, docs, top_k=5):
        super().__init__()
        self.question = question
        self.docs = docs
        self.top_k = top_k

    def to_dict(self):
        return {
            'question': self.question,
            'docs': self.docs,
            'top_k': self.top_k
        }


# 使用bm25进行检索
def bm25_retrieve(query, contents):
    bm25 = BM25Okapi(contents)
    # 对于每个文档，计算结合BM25
    bm25_scores = bm25.get_scores(query)
    # 根据得分排序文档
    sorted_docs = sorted(zip(contents, bm25_scores), key=lambda x: x[1], reverse=True)
    # print("通过bm25检索结果，查到相关文本数量：", len(sorted_docs))
    return sorted_docs


# 使用faiss向量数据库的索引进行查询
def embedding_retrieve(query, txt_file, model):
    embed_select_docs = []
    faiss_save_path = os.path.join("data/embedding_index", txt_file+'.faiss')
    if os.path.exists(faiss_save_path):
        index = faiss.read_index(faiss_save_path)
        query_embedding = numpy.array(model.encode(query))
        _, search_result = index.search(query_embedding.reshape(1, -1), 5)
        pkl_save_path = os.path.join(config.pkl_save_path, txt_file.split('.')[0] + '.pkl')
        with open(pkl_save_path, 'rb') as file:
            docs_dict = pickle.load(file)
        chunk_docs = list(docs_dict.keys())
        embed_select_docs = [chunk_docs[i] for i in search_result[0]]  # 存储为列表
        # print("通过embedding检索结果，查到相关文本数量：", len(embed_select_docs))
    else:
        print('找不到对于的faiss文件，请确认是否已经进行存储')

    return embed_select_docs


def search(query, model, llm, top_k=5):
    # 读取公司名称列表
    df = pd.read_csv(config.company_save_path)
    company_list = df['company'].to_numpy()

    # 使用大模型获得最终公司的名称
    prompt = ChatPromptTemplate.from_template(prompts.COMPANY_PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"company": company_list, "question": query})
    # print(response.content)
    company_name = response.content
    for name in company_list:
        if name in company_name:
            company_name = name
            break
    # print(company_name)

    # 通过bm25获取相似度最高的chunk
    pkl_file = os.path.join(config.pkl_save_path, company_name + '.pkl')
    with open(pkl_file, 'rb') as file:
        docs_dict = pickle.load(file)
        chunk_docs = list(docs_dict.keys())
    bm25_chunks = [docs_tuple[0] for docs_tuple in bm25_retrieve(query, chunk_docs)[:top_k]]
    # 通过embedding获取相似度最高的chunk
    embedding_chunks = embedding_retrieve(query, company_name, model)
    # 重排
    chunks = list(set(bm25_chunks + embedding_chunks))
    # print("通过双路检索结果：", len(chunks))
    arg = Query(question=query, docs=chunks, top_k=top_k)
    chunk_similarity = rerank_api(arg)
    # for r in chunk_similarity.items():
    #     print(r)

    # 获取父文本块
    result_docs = []
    pkl_dict_file = os.path.join(config.pkl_save_path, company_name + '_dict' + '.pkl')
    with open(pkl_dict_file, 'rb') as file:
        child_parent_dict = pickle.load(file)
    for key, _ in sorted(chunk_similarity.items(), key=lambda x: x[1], reverse=True):
        for child_txt, parent_txt in child_parent_dict.items():  # 遍历父文本块
            if key == child_txt:  # 根据匹配的子文本块找到父文本
                result_docs.append(parent_txt)
    # print("==========最终结果==============")
    # for d in result_docs:
    #     print(d)
    return result_docs


def rerank_api(query, url="http://127.0.0.1:8000/bge_rerank"):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(query.__dict__)
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=3))
    try:
        res = s.post(url, data=data, headers=headers, timeout=600)
        if res.status_code == 200:
            return res.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None


if __name__ == '__main__':
    user_query = '报告期内，华瑞电器股份有限公司人工成本占主营业务成本的比例分别为多少？'
    # 检索
    search(user_query, BEG_MODEL, LLM)
