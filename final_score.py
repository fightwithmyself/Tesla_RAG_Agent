# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------


import os
import pickle
import time
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from openai import OpenAI
from tqdm import tqdm


from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.constant import bge_reranker_tuned_model_path
from src.constant import qwen3_reranker_model_path 
from src.constant import text2vec_model_path 
from src.utils import merge_docs, post_processing


# warmstart
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
simModel = SentenceModel(model_name_or_path=text2vec_model_path, device='cuda:0')

BM25_RETRIEVE_SIZE = 5
MILVUS_RETRIEVE_SIZE = 10
RERANK_SIZE = 5
HYDE = 0


def calc_jaccard(list_a, list_b, threshold=0.3):
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


def report_score(result):
    idx = 0
    for item in result:
        question = item["question"]
        keywords = item["keywords"]
        gold = item["answer"]
        pred = item["pred"]["answer"]
        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            if not keywords:
                score = semantic_score
            else:
                score = 0.2 * keyword_score + 0.8 * semantic_score
        result[idx]["score"] = score
        idx += 1
        if score < 0.6:
            print(f"预测: {question}, 得分: {score}")

    return result



fd = open("data/qa_pairs/test_qa_pair_verify.json")
test_qa_pairs = json.load(fd)
result = []
for item in test_qa_pairs:
    query = item["question"].strip()
    if HYDE:
        hyde_query = request_hyde(query) 
        hyde_query = query + "\n" + hyde_query 
        bm25_docs = bm25_retriever.retrieve_topk(hyde_query, topk=BM25_RETRIEVE_SIZE)
        milvus_docs = milvus_retriever.retrieve_topk(hyde_query, topk=MILVUS_RETRIEVE_SIZE)
    else:
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=BM25_RETRIEVE_SIZE)
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=MILVUS_RETRIEVE_SIZE)
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=RERANK_SIZE)
    context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
    response = request_chat(query, context)
    answer = post_processing(response, ranked_docs)
    print("问题：", query)
    print("答案：", answer)
    print("="*100)
    item["pred"] = answer
    item["context"] = context
    result.append(item)

with open("data/qa_pairs/test_qa_pair_pred.json", "w") as fw:
    fw.write(json.dumps(result, ensure_ascii=False, indent=4))


with open("data/qa_pairs/test_qa_pair_pred.json") as fw:
    result = json.load(fw) 

results = report_score(result)
final_score = np.mean([item["score"] for item in results])
print("\n")
print(f"预测问题数：{len(results)}, 语义相似度+关键词加权得分：{final_score}")


"""
以下是RAG评估代码的扩展，利用Ragas框架来对问答系统输出的结果做评估。输入是query，生成的答案，参考答案，以及召回的上下文信息。
评估采用了精确率和召回率两个指标
"""

llm = ChatOpenAI(model=os.environ["DOUBAO_MODEL_NAME"], api_key=os.environ["DOUBAO_API_KEY"], base_url=os.environ["DOUBAO_BASE_URL"])

print("开始做RAGas评估...")
dataset = []
for g in result:
    query = g["question"] # 输入问题
    reference = g["answer"] # 参考答案
    response = g["pred"]["answer"] #生成的答案
    context = [g["context"]] # 上下文
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts": context,
            "response":response,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],llm=evaluator_llm)

# 系统输出得分
print("\n")
print("="*100)
print(f"预测问题数：{len(results)}, 语义相似度+关键词加权得分：{final_score}")
print(f"预测问题数：{len(results)}, LLM+RAGas综合得分：{result}")
print("="*100)
