# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  @丁师兄大模型
# --------------------------------------------

import os
import pickle
import time
import json
import re
import random
from tqdm import tqdm
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_chat_client import request_chat
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.reranker.qwen3_reranker_vllm import Qwen3ReRankervLLM 
from src.constant import bge_reranker_model_path
from src.constant import qwen3_4b_reranker_model_path
from src.utils import merge_docs, post_processing

random.seed(42)


LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}},{{引用编号2}},...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""


# warmstart
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
# bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_model_path)
qwen3_reranker = Qwen3ReRankervLLM(model_path=qwen3_4b_reranker_model_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)


fd = open("data/qa_pairs/train_qa_pair.json")
test_qa_pairs = json.load(fd)
output_handler = open("data/qa_pairs/train_data.json", "w")
for item in tqdm(test_qa_pairs):
    try:
        query = item["question"].strip()
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        ranked_docs = qwen3_reranker.rank(query, merged_docs, topk=5)
        context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
        response = request_chat(query, context)
        answer = post_processing(response, ranked_docs)
        context = [q.page_content for q in ranked_docs]
        all_docs = [q.page_content for q in merged_docs]
        info = {"query": query, "context": context, "response": response, "merged_docs": all_docs}
        info = json.dumps(info, ensure_ascii=False)
        output_handler.write(info+'\n')
        output_handler.flush()
    except:
        pass


MAX_INPUT_SIZE = 4096
RERANK_DEV_SIZE = 1000
TEST_RATE = 0.08

summary_train_handler = open("./data/summary_data/train.json", "w")
summary_test_handler = open("./data/summary_data/test.json", "w")
rerank_train_handler = open("./data/rerank_data/train.json", "w")
rerank_dev_handler = open("./data/rerank_data/dev.json", "w")
rerank_test_handler = open("./data/rerank_data/test.json", "w")

summary_train = []
summary_test = []
rerank_train = []
rerank_test = []
fd = open("data/qa_pairs/train_data.json")
for line in fd:
    info = json.loads(line)
    response = info["response"]
    all_cites = re.findall("[【](.*?)[】]", response)
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = sorted(list(set(cites)))
    cites = ",".join([str(c) for c in cites])
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)
    if cites:
        format_answer = answer + f"【{cites}】"
    else:
        format_answer = "无答案"
    context = "\n".join([str(idx+1) + "." + doc for idx, doc in enumerate(info["context"])])
    if len(context) > MAX_INPUT_SIZE:
        context = context[:MAX_INPUT_SIZE]

    query = info["query"].strip()
    instruction = LLM_CHAT_PROMPT.format(query=query, context=context)
    item = {
        "query": query,
        "context": context,
        "instruction": instruction,
        "input": "",
        "output": format_answer
    }
    neg_docs = [doc for doc in info["merged_docs"] if doc not in info["context"]]
    if random.random() < TEST_RATE:
        summary_test.append(item)

        if format_answer != "无答案":
            content_list = [info["context"][0], random.choice(info["context"][-2:])]
            if neg_docs:
                content_list.append(random.choice(neg_docs))
            rerank_test.append({"query": query, "content": content_list})

    else:
        summary_train.append(item)

        # rerank
        if format_answer != "无答案":
            positive = info["context"][0]
            middle = random.choice(info["context"][-2:])
            rerank_train.append({"query": query, "content": positive, "label": 2})
            rerank_train.append({"query": query, "content": middle, "label": 1})
            if neg_docs:
                negative = random.choice(neg_docs)
                rerank_train.append({"query": query, "content": negative, "label": 0})
        else:
            negative = random.choice(info["merged_docs"])
            rerank_train.append({"query": query, "content": negative, "label": 0})

rerank_train = [item for item in rerank_train if len(item["query"]) > 0 and len(item["content"]) > 0]
rerank_dev = rerank_train[-RERANK_DEV_SIZE:]
random.shuffle(rerank_train)
print("Rerank Train size:", len(rerank_train), "Rerank Test size:", len(rerank_test))

for item in rerank_train:
    rerank_train_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
for item in rerank_dev:
    rerank_dev_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
for item in rerank_test:
    rerank_test_handler.write(json.dumps(item, ensure_ascii=False) + "\n")


print("Summary Train size:", len(summary_train), "Summary Test size:", len(summary_test))
summary_train_handler.write(json.dumps(summary_train, ensure_ascii=False, indent=4))
summary_test_handler.write(json.dumps(summary_test, ensure_ascii=False, indent=4))
