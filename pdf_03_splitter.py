import os
import faiss
import numpy
import pickle
import config
from tqdm import tqdm
from util.instances import BEG_MODEL
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 将每个公司的txt文件进行分块，并将分别存储在本地文件和本地向量数据库
# 本地文件存为pkl，用于bm25的相似度查询
# 本地向量数据库，用于embedding的相似度查询
def splitter_doc(txt_file, model, splitter=False, doc_chunk_size=800, doc_chunk_overlap=100,
                 sub_chunk_size=150, sub_chunk_overlap=50):
    if not splitter:
        pkl_save_path = os.path.join(config.pkl_save_path, txt_file.split('.')[0] + '.pkl')
        if os.path.exists(pkl_save_path):
            print('当前文件已经初始化完成，无需再次初始化，如希望重新写入，则将参数splitter设为True')
            return

    # 第一步，读取txt文件
    cur_file_path = os.path.join('data/pdf_txt_file2', txt_file)
    with open(cur_file_path, 'r', encoding='utf-8') as file:
        file_doc = file.read()
    # 第二步，先将文档切块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=doc_chunk_size, chunk_overlap=doc_chunk_overlap,
                                                   separators=["\n"], keep_separator=True, length_function=len)
    parent_docs = text_splitter.split_text(file_doc)
    print(len(parent_docs))
    # 第三步，将切块再次切分小文本
    cur_text = []
    child_parent_dict = {}  # 子模块与父模块的dict
    for doc in parent_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=sub_chunk_size, chunk_overlap=sub_chunk_overlap,
                                                       separators=["\n", ], keep_separator=True, length_function=len)
        child_docs = text_splitter.split_text(doc)
        for child_doc in child_docs:
            child_parent_dict[child_doc] = doc
        cur_text += child_docs

    # 第四步，将文本向量化，返回一个key为文本，value为embedding的dict
    result_dict = dict()
    for doc in tqdm(cur_text):
        result_dict[doc] = numpy.array(model.encode(doc))
    # 第五步，将dict存储为.pkl文件，用于bm25相似度查询
    pkl_save_path = os.path.join(config.pkl_save_path, txt_file.split('.')[0] + '.pkl')
    if os.path.exists(pkl_save_path):
        os.remove(pkl_save_path)
        print('存在旧版本pkl文件，进行先删除，后创建')
    with open(pkl_save_path, 'wb') as file:
        pickle.dump(result_dict, file)
    print('完成pkl数据存储：', pkl_save_path)

    pkl_dict_save_path = os.path.join(config.pkl_save_path, txt_file.split('.')[0] + '_dict' + '.pkl')
    if os.path.exists(pkl_dict_save_path):
        os.remove(pkl_dict_save_path)
        print('存在旧版本pkl dict文件，进行先删除，后创建')
    with open(pkl_dict_save_path, 'wb') as file:
        pickle.dump(child_parent_dict, file)

    print('完成pkl dict数据存储：', pkl_dict_save_path)

    # 第六步，将dict中的向量化数据存储到faiss数据库
    result_vectors = numpy.array(list(result_dict.values()))
    dim = result_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(result_vectors)
    index.add(result_vectors)
    faiss_save_path = os.path.join(config.faiss_save_path, txt_file.replace('txt', 'faiss'))
    if os.path.exists(faiss_save_path):
        os.remove(faiss_save_path)
        print('存在旧版本faiss索引文件，进行先删除，后创建')
    faiss.write_index(index, faiss_save_path)
    print('完成faiss向量存储：', faiss_save_path)


if __name__ == '__main__':
    txt_file_name = '安徽黄山胶囊股份有限公司.txt'
    # 存储数据
    splitter_doc(txt_file_name, BEG_MODEL)
