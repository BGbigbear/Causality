import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from BCEmbedding.tools.langchain import BCERerank
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_vector_database(causality_data, vector_data_path):
    embedding_path = 'C:/Users/sibin/worksp/COMPETITION/Causality Extraction/models/bce-embedding-base_v1'
    # 加载embeddings模型
    embedding_model_name = embedding_path
    embedding_model_kwargs = {'device': "cuda" if torch.cuda.is_available() else "cpu"}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    # 加载数据库
    if os.path.exists(vector_data_path):
        # 加载向量数据库,可能需要重新构建向量数据库
        db = FAISS.load_local(vector_data_path, embeddings, allow_dangerous_deserialization=True,
                              distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    # 构建向量数据库
    else:
        # 对规则进行划分并入库
        chunks = []
        for i, data in enumerate(causality_data):
            chunks.append(f"{i}_{data['text']}")
        db = FAISS.from_texts(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        db.save_local(vector_data_path)
    return db


def load_retriever(causality_data, vector_path):
    db = create_vector_database(causality_data, vector_path)

    reranker_path = 'C:/Users/sibin/worksp/COMPETITION/Causality Extraction/models/bce-reranker-base_v1'
    # 加载排序模型
    reranker_args = {'model': reranker_path, 'top_n': 3, 'device': "cuda" if torch.cuda.is_available() else "cpu"}
    reranker = BCERerank(**reranker_args)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.4, "k": 50})
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    return compression_retriever
