# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

import re
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig

manual_collection = MongoConfig.get_collection("manual_text")


def merge_docs(docs1, docs2):
    merged_docs = []
    merged_ids = set()
    candidate_docs = docs1 + docs2
    for doc in candidate_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_mg = manual_collection.find_one({"unique_id": parent_id})
            unique_id = parent_mg["unique_id"]
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                merged_docs.append(parent_doc)
        else:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                merged_docs.append(doc)
    return merged_docs




def post_processing(response, docs):
    all_cites = re.findall("[【](.*?)[】]", response) 
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = list(set(cites))
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)

    related_images = []
    pages = []
    for index in cites:
        if index > len(docs):
            continue
        images = docs[index-1].metadata["images_info"]
        pages.append(docs[index-1].metadata["page"])
        for image in images:
            if image["title"]:
                related_images.append(image)
    pages = sorted(list(set(pages)))
    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }
