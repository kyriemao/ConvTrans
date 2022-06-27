# ConvTrans
This repository contains the code (primary version) of our EMNLP 2022 submission: "_ConvTrans: Transforming Web Search Sessions for Conversational Dense Retrieval_"

## Environment

The main running enviroment is:
- python 3.7.11
- pytorch 1.7.1
- transformers 4.18.0
- numpy: 1.19.2
- scipy: 1.1.0
- tensorboard: 2.4.0


## Datasets
- Raw web search sessions: [MSMARCO search sessions and relevance labels](https://microsoft.github.io/msmarco/)
```
# raw web search sessions
wget https://msmarco.blob.core.windows.net/conversationalsearch/ann_session_dev.tar.gz 

# relevance labels
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv

# queries
https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
```

- [MSMARCO passage collections](https://microsoft.github.io/msmarco/)

- CAsT evaluation sets: [CAsT-19, CAsT-20](https://www.treccast.ai/).The needed raw files include:
  - 2020_automatic_evaluation_topics_v1.0.json
  - 2020qrels.txt
  - evaluation_topics_v1.0.txt
  - 2019qrels.txt

## Code
- Transforming raw web search sessions into pseudo conversational search sessions as training data for the conversational session encoder.
```
python gen_session.py --config=config/gen_session.toml
```
(1) generating session graphs from the raw web search sessions.
(2) performing query transformation on the session graphs
(3) performing random walk sampling to generate the final pseudo conversational search sessions to be used as training data.


- Using the data to train the conversational session encoder.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
train_model.py --config=./config/train_model.toml

```

- Evaluation the trained model on CAsT datasets
```
python test_model.py --config=config/test_model.toml
```

- Other files:
  - processing passage collections, generating passage (doc) embeddings in advance for testing: gen_tokenized_doc.py, gen_cast_collection.py, gen_doc_embedding.py
  - training and inference files of the conversational query rewriter of ConvTrans: train_kw_to_nl_t5.py, train_nl_to_cnl_t5.py, inference_kw_to_nl_t5.py, inference_nl_to_cnl_t5.py