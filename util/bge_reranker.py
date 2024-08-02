import uvicorn
from fastapi import FastAPI
from operator import itemgetter
from FlagEmbedding import FlagReranker
from pydantic import BaseModel


# 定义FastAPI
app = FastAPI()
# 定义BGE-reranker模型
reranker = FlagReranker('../model/Xorbits/bge-reranker-base')  # 修改为你模型文件路径


# 定义入参
class Query(BaseModel):
    question: str
    docs: list[str]
    top_k: int = 1

# 提供访问入口
@ app.post('/bge_rerank')
def bge_rerank(query: Query):
    scores = reranker.compute_score([[query.question, passage] for passage in query.docs])
    if isinstance(scores, list):
        similarity_dict = {passage: scores[i] for i, passage in enumerate(query.docs)}
    else:
        similarity_dict = {passage: scores for i, passage in enumerate(query.docs)}
    sorted_similarity_dict = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
    result = {}
    for j in range(query.top_k):
        result[sorted_similarity_dict[j][0]] = sorted_similarity_dict[j][1]
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

