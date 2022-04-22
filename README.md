# Biaffine-JointPN

Knowledge Graph Question Answering Based on Bi-affine Transformation and Pointer Network Joint ModelingJoint Modeling

- backup_code是平时备份代码的文件夹，用不到
- notebook是一些jupyter写的文件，用于调试，用不到
- biaffine_ner 这个文件夹的内容是利用双仿射变换(Biaffine)作为输出层进行实体识别
- joint_with_mrc 这个文件夹的内容是联合实体消歧与关系匹配

## 下载知识图谱和预训练语言模型
```bash
mkdir chinese-roberta-wwm-ext
mkdir knowledge_graph
```

### 下载预训练语言模型
[模型地址](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main)

将下载的文件存放在chinese-roberta-wwm-ext文件夹下，确保chinese-roberta-wwm-ext文件夹含有如下4个文件：
- config.json
- pytorch_model.bin
- tokenizer.json
- vocab.txt

### 下载知识图谱和问答数据
[问答数据地址](https://github.com/CLUEbenchmark/KgCLUE/tree/main/datasets)
[知识图谱地址](https://github.com/CLUEbenchmark/KgCLUE#%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D)
将下载的文件存放在knowledge_graph文件夹下，确保knowledge_graph文件夹含有如下



## 训练

### 基于BERT-Biaffine的实体识别模型
```bash
cd biaffine_ner
export knowledge_graph='knowledge_graph'
export bert_model_path='chinese-roberta-wwm-ext'
export checkpoints='checkpoints'

python main.py \
    --knowledge_graph $knowledge_graph \
    --checkpoints $checkpoints \
    --bert_model_path $bert_model_path
```

参数说明：
- knowledge_graph 表示下载的知识图谱和问答数据集存放的文件夹路径
- bert_model_path 表示下载的预训练语言模型的存放路径
- checkpoints 表示模型存放的路径

训练过程的日志保存在log.txt中

### 基于BERT-PointNetwork联合建模

```bash
cd jointPN
export knowledge_graph='knowledge_graph'
export bert_model_path='chinese-roberta-wwm-ext'
export checkpoints='checkpoints2'

python main.py \
    --knowledge_graph $knowledge_graph \
    --checkpoints $checkpoints \
    --bert_model_path $bert_model_path
```