# Tesla RAG Agent · 车书问答系统

面向 **Tesla Model 3 用户手册** 的检索增强问答系统。输入一句自然语言提问，返回答案 + **手册原始页码** + **相关插图**，让回答可被人工核对。

```
输入：走开的时候车怎么自己锁门？

输出：
  answer         带着手机钥匙或配对的遥控钥匙离开时，车门和行李箱会自动锁定；
                 锁闭时外部车灯会闪烁一次。可通过 控制 > 车锁 > 离车后自动上锁 开启。
  cite_pages     [7]
  related_images [{ page: 7, image_path: "...page7_img1.jpeg", title: "车锁设置界面" }]
```

技术栈：`PyMuPDF` · `MongoDB` · `Milvus` · `BM25` · `BGE-M3` · `BGE-Reranker` · `LLaMA-Factory` · `LoRA` · `AWQ` · `vLLM`

---

## 1. 系统架构

### 离线：文档 → 知识库

```
Tesla_Manual.pdf
    │
    ├─ PyMuPDF 解析          逐页取文本；裁掉页脚 50px 去页码干扰；
    │                        过滤封面与目录（仅保留第 4–247 页）；抽取插图并识别图题
    │
    ├─ LLM 清洗              断句接顺 + 用 ### 标记小节标题（补回 PDF 丢失的结构信息）
    │
    ├─ 复合切分              ① 有 ### → 按小节切
    │                        ② 无标题、段落 ≤ 10 → 按空行切
    │                        ③ 段落过多 → m3e-small 编码 + 层次聚类（余弦/average）
    │                        ④ 分组结果为【父块】，再按 256 token / 50 overlap 切【子块】
    │
    ├─ MongoDB               父块 + 子块全量入库（page_content + metadata）
    │                        子块 metadata 带 parent_id，指向所属父块
    │
    └─ 双索引                BM25（jieba 分词，pickle 持久化）
                             Milvus（BGE-M3 稠密 + 稀疏双向量）
```

### 在线：提问 → 带引用的答案

```
query
    │
    ├─ BM25 召回 top10               负责专有名词的字面精确匹配
    ├─ Milvus 召回 top10             BGE-M3 稠密 + 稀疏，库内 RRF 融合，负责同义表述
    │
    ├─ merge_docs                    两路取并集去重；命中子块则回查 MongoDB 回填父块
    │                                （用小块找、用大块答）
    │
    ├─ BGE-Reranker 精排 top5        Cross-Encoder，经 Qwen3-Reranker 教师蒸馏微调
    │
    ├─ 拼装带编号上下文 →  vLLM       Qwen3-8B（LoRA SFT + AWQ INT4），OpenAI 兼容接口
    │                                Prompt 强制「答案【引用编号】」，无依据则回答「无答案」
    │
    └─ post_processing               正则解析引用编号 → 映射回 page 与 images_info
                                     返回 { answer, cite_pages, related_images }
```

---

## 2. 关键设计

### 2.1 父子块双粒度切分

召回与生成对块大小的诉求相反：召回要**短而语义集中**，生成要**长而上下文完整**。单一尺寸无法兼顾。

本项目让两种粒度同时进入检索集，命中子块后通过 `parent_id` 回查 MongoDB 换成父块，再送入精排与生成：

| | 粒度 | 作用 |
|---|---|---|
| 子块 | 256 token / 50 overlap | 向量语义集中，召回精度高 |
| 父块 | 语义分组结果（< 512 字符进检索集） | 上下文完整，生成不截断 |

切分长度按 `tiktoken (cl100k_base)` 的 **token 数**计量，而非字符数。

### 2.2 多路混合召回 + RRF

| 通路 | 擅长 | 短板 |
|---|---|---|
| BM25（jieba） | 专有名词、精确匹配 | 同义表述召不回 |
| BGE-M3 稠密 | 语义相似、口语化提问 | 容易把专有名词泛化 |
| BGE-M3 稀疏 | 带上下文感知的词权重 | — |

稠密与稀疏在 Milvus 内用 `RRFRanker` 融合。采用 RRF（`1/(k+rank)` 求和）而非加权求和，是因为 BM25 分数与向量内积量纲不同，加权方案需要反复调参，而 RRF 只依赖名次、无需调参。

BM25 与 Milvus 两路之间直接取并集而不做二次融合——粗排阶段的目标是「别漏」，排序交由后续 Cross-Encoder 统一完成。

### 2.3 精排模型蒸馏

通用重排模型在汽车手册领域排序不准，而效果更好的大模型重排器推理过慢，无法上线。因此做离线蒸馏：

- **教师**：Qwen3-Reranker-4B（vLLM 推理，仅离线使用）
- **学生**：bge-reranker-v2-m3（约 568M，线上使用）
- **标签**：由教师精排的名次映射为分级标签

| label | 来源 | 含义 |
|---|---|---|
| 2 | 教师 top1 | 高度相关 |
| 1 | 教师 top5 中靠后者 | 中等相关 |
| 0 | 进入粗排候选但未进 top5 | **难负样本** |

label 0 取自 `merged_docs - context`，这些片段能被召回说明字面或语义相近，却被教师判定不相关，正处于模型决策边界上，训练价值最高。

训练使用 pointwise + `BCEWithLogitsLoss`，标签经 `(label-min)/(max-min)` 归一化为 0 / 0.5 / 1 的 soft target。配置见 [`configs/rag_retrieval/training_bert.yaml`](configs/rag_retrieval/training_bert.yaml)。

### 2.4 训练推理一致的 SFT 数据构造

常规做法是「让教师模型看着整页原文生成问答对」，但线上推理时模型看到的是**检索精排后的 5 个片段**，两者输入分布不一致。

本项目在 `generate_sft_data.py` 中**重跑一遍真实检索链路**，由教师模型在真实上下文下生成带引用的答案，训练时的 `instruction` 与线上 Prompt 完全一致。

同一份数据同时派生两套训练集：

```
train_qa_pair.json ──→ 真实检索链路 ──→ train_data.json
                                            ├─→ data/summary_data/   生成模型 SFT
                                            └─→ data/rerank_data/    精排模型蒸馏
```

### 2.5 引用格式与拒答

**引用编号是运行时按精排位置临时分配的序号**（不存数据库），由五层机制保证端到端可靠：

1. **Prompt 模板**给出 `{{答案}}【{{引用编号}}】` 的确切结构
2. **标签规范化**——教师输出的引用统一去重、排序、逗号归一，1000 条样本格式完全一致
3. **LoRA 微调**将格式固化进权重，不依赖 Prompt 约束
4. **近贪心解码**（`temperature=0.001`、`top_k=1`）保证输出确定性
5. **后处理正则**兜住格式变体，并将编号映射回 `page` / `images_info`

拒答能力的训练数据有两个来源：`data/ut/raw_general_chats.txt` 中的闲聊 query（如「成都的市花是什么花」）标注为「无答案」；以及教师回答中未产生任何引用编号的样本。评测时拒答单独做 0/1 判定，不与语义相似度混算。

### 2.6 LoRA 微调 + AWQ 量化部署

单张 48G 显存无法承载 8B 全参微调（AdamW 混合精度约需 16 bytes/参数 ≈ 128 GB），故采用 LoRA：

| 阶段 | 做法 | 显存 / 体积 |
|---|---|---|
| 训练 | LoRA `rank=8`、`lora_target=all`、bf16、等效 batch 8 | 基座冻结，省去约 96 GB 优化器状态 |
| 合并 | `llamafactory-cli export`，CPU 上完成 | bf16 约 16 GB |
| 量化 | AWQ INT4，`q_group_size=128`、`zero_point=True` | 约 5.8 GB |
| 部署 | vLLM，`--max-model-len 8192 --gpu-memory-utilization 0.7` | 余量留给检索与精排模型 |

AWQ 的校准数据直接复用 SFT 训练集并套上 chat template——激活分布决定了哪些权重通道显著，用领域内真实样本校准可最小化量化误差。见 [`configs/llama_factory/awq_quant.py`](configs/llama_factory/awq_quant.py)。

---

## 3. 目录结构

```
.
├── build_index.py              离线主流程：解析 → 清洗 → 切分 → 建双索引
├── generate_sft_data.py        造数主流程：真实检索链路 → SFT 数据 + 精排蒸馏数据
├── infer.py                    在线问答（命令行交互）
├── final_score.py              评测：语义相似度 + 关键词覆盖，并接入 Ragas
├── config.ini.example          环境变量与服务启动脚本模板
│
├── src/
│   ├── constant.py             全部路径与模型位置集中配置
│   ├── utils.py                merge_docs（去重 + 父块回填）、post_processing（引用解析）
│   ├── parser/
│   │   ├── pdf_parse.py        PyMuPDF 解析、复合切分、写入 MongoDB
│   │   └── image_handler.py    插图抽取与图题识别（字号/粗体/位置加权打分）
│   ├── server/
│   │   └── semantic_chunk.py   语义切分服务（FastAPI :6000，m3e-small + 层次聚类）
│   ├── client/
│   │   ├── llm_clean_client.py     文档清洗（云端教师模型）
│   │   ├── llm_chat_client.py      造数答案生成（云端教师模型）
│   │   ├── llm_local_client.py     线上生成（vLLM :8000，含问答 Prompt）
│   │   ├── llm_hyde_client.py      HyDE 查询改写（可选，默认关闭）
│   │   ├── semantic_chunk_client.py
│   │   └── mongodb_config.py       MongoDB 单例连接
│   ├── retriever/
│   │   ├── bm25_retriever.py       BM25 + jieba，pickle 持久化
│   │   ├── milvus_retriever.py     BGE-M3 稠密 + 稀疏，RRF 融合
│   │   └── faiss_retriever.py / tfidf_retriever.py   对照实验
│   ├── reranker/
│   │   ├── bge_m3_reranker.py      线上精排（微调后的 Cross-Encoder）
│   │   └── qwen3_reranker_vllm.py  离线教师重排器
│   ├── fields/                 Pydantic 数据模型（ManualInfo / ManualImages）
│   └── gen_qa/run.py           QA 合成、同义扩写、关键词抽取、闲聊负样本
│
├── configs/                    训练与部署配置快照（第三方框架本体不入库）
│   ├── llama_factory/          LoRA 训练 / 合并 / 推理 / AWQ 量化 / vLLM 启动
│   └── rag_retrieval/          精排蒸馏训练配置 + accelerate 配置
└── scripts/download_models.sh  ModelScope 模型下载
```

---

## 4. 环境

### 硬件

单张 NVIDIA GPU **48GB**（AutoDL 实例，如 A6000 / L40）。训练与部署分时复用同一张卡。

### 软件

| 类别 | 版本 |
|---|---|
| OS / Python | Ubuntu 22.04 · Python 3.10 |
| CUDA | 12.6 |
| 深度学习 | torch 2.7.0 · transformers 4.51.1 · accelerate 1.7.0 |
| 推理 | vllm 0.9.0.1 · autoawq 0.2.9 |
| 检索 | pymilvus · FlagEmbedding · sentence-transformers · jieba · tiktoken |
| 解析 / 存储 | PyMuPDF · pymongo · MongoDB 7.0 |
| 服务 | FastAPI · uvicorn |

```bash
pip install -r requirements.txt
```

### 需自行准备的外部依赖

仓库不含模型权重、PDF 原件与数据产物（见 `.gitignore`），需自行准备：

```bash
# 1. 模型权重（约 30GB）
bash scripts/download_models.sh

# 2. 第三方训练框架
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory-main
git clone https://github.com/NLPJCL/RAG-Retrieval.git RAG-Retrieval
# 随后把 configs/ 下的配置覆盖回各自目录

# 3. MongoDB 7.0（解压即用）
# 4. Tesla Model 3 用户手册 PDF → data/Tesla_Manual.pdf
```

---

## 5. 快速开始

### 配置

```bash
cp config.ini.example config.ini
vim config.ini          # 填入 API Key、调整路径
source config.ini       # 同时会拉起切分服务 / vLLM / MongoDB
```

`src/constant.py` 中的 `base_dir` 默认为 `/root/autodl-tmp/RAG/`，请改为自己的项目根路径。

### 离线建库

```bash
python build_index.py
```

依次完成：PDF 解析 → LLM 清洗 → 复合切分 → 写入 MongoDB → 构建 BM25 与 Milvus 索引。中间结果落盘为 pickle，可断点续跑。

### 在线问答

```bash
python infer.py
# 输入—> 走开的时候车怎么自己锁门
```

### 评测

```bash
python final_score.py
```

输出「语义相似度 + 关键词覆盖」加权分（拒答样本单独 0/1 判定）与 Ragas 的 context recall / precision。

### 训练复现

```bash
# 1. 造数（真实检索链路 → SFT 数据 + 精排蒸馏数据）
python generate_sft_data.py

# 2. 精排蒸馏
cd RAG-Retrieval/rag_retrieval/train/reranker && bash train.sh

# 3. 生成模型 LoRA 微调
cd LLaMA-Factory-main
cp ../data/summary_data/train.json data/summary_train.json      # 注册名 faq_summary
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 4. 合并 LoRA → AWQ INT4 → vLLM
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
python awq_quant.py
bash vllm_serve.sh
```

---

## 6. 关键参数

| 参数 | 值 | 位置 |
|---|---|---|
| 正文页范围 | 第 4–247 页 | `pdf_parse.py` |
| 页脚裁剪 | 50 px | `pdf_parse.py` |
| 子块长度 / 重叠 | 256 / 50 token | `pdf_parse.py` |
| 父块入检索集上限 | 512 字符 | `pdf_parse.py` |
| 语义分组目标大小 | 10 段 | `pdf_parse.py` |
| BM25 / Milvus 召回量 | 10 / 10 | `infer.py` |
| 精排保留 | top 5 | `infer.py` |
| 精排训练 | pointwise BCE · lr 5e-5 · bs 8 × acc 2 · fp16 | `configs/rag_retrieval/training_bert.yaml` |
| LoRA | rank 8 · target all · lr 2e-5 · 3 epoch · bf16 · cutoff 4500 | `configs/llama_factory/train_lora_qwen3.yaml` |
| AWQ | w_bit 4 · group 128 · zero_point | `configs/llama_factory/awq_quant.py` |
| vLLM | max-model-len 8192 · gpu-util 0.7 | `configs/llama_factory/vllm_serve.sh` |

---

## 7. 已知局限与改进方向

对照代码梳理的待改进点，供复现者参考：

| 位置 | 问题 | 改进 |
|---|---|---|
| `pdf_parse.py` | 父子块在**单页范围内**构建，跨页小节仍会被割裂 | 清洗后按 `###` 跨页重组为逻辑小节再切分，`page` 改为页码列表 |
| `generate_sft_data.py` | 精排验证集由训练集尾部切出且未从训练集移除，验证指标偏乐观 | 先划分再落盘 |
| `generate_sft_data.py` | label 1 在 `context[-2:]` 中随机选取，候选不足时可能与 label 2 重复 | 改用教师打分做连续 soft target，并加判重 |
| `generate_sft_data.py` | 上下文按**字符数**截断，可能切坏引用编号对应关系 | 改为按 token 截断，文档整条保留或整条丢弃 |
| `generate_sft_data.py` | 裸 `except: pass` 导致数据静默丢失 | 记录异常并统计失败率 |
| `mongodb_config.py` | 未对 `unique_id` 建索引，`find_one` 为全集合扫描 | `create_index("unique_id", unique=True)` |
| `infer.py` | 上下文编号格式（`【1】`）与造数时（`1.`）不一致 | 统一模板 |
| `utils.py` | `post_processing` 仅校验引用编号上界 | 补充 `index < 1` 的下界校验 |
| 评测 | 测试集全部由模型合成，缺人工标注金标准 | 人工标注一批作为基准 |

---

## 8. 声明

- 本仓库为个人学习与技术复现项目，源码结构参考公开课程实现，不用于商业用途。
- Tesla Model 3 用户手册版权归特斯拉所有，仓库不包含手册原件及其衍生数据。
- 所有 API Key 均通过环境变量注入，请勿将 `config.ini` 提交到版本库。
