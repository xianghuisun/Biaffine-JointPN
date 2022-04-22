import os
from posixpath import split
import sys
from transformers import AutoTokenizer
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata
from collections import defaultdict
from tqdm import tqdm
import re

def get_tokenizer(bert_model_path):
    """添加特殊中文字符和未使用的token【unused1】"""
    tokenizer=AutoTokenizer.from_pretrained(bert_model_path)
    return tokenizer

def generate_label2id(file_path):
    with open(file_path) as f:
        lines=f.readlines()
    label2id={}
    for line in lines:
        line_split=line.strip().split()
        if len(line_split)>1:
            label2id[line_split[-1]]=len(label2id)
    return label2id

def process_nerlabel(label2id):
    #label2id,id2label,num_labels = tools.load_schema_ner()
    #Since different ner dataset has different entity categories, it is inappropriate to pre-assign entity labels
    new_={}
    new_={'O':0}
    for label in label2id:
        if label!='O':
            label='-'.join(label.split('-')[1:])
            if label not in new_:
                new_[label]=len(new_)
    return new_

class token_rematch:
    def __init__(self):
        self._do_lower_case = True


    @staticmethod
    def stem(token):
            """获取token的“词干”（如果是##开头，则自动去掉##）
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """控制类字符判断
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """判断是不是有特殊含义的符号
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def load_schema_ner():
    
    id2label={1: 'ent'}
    label2id={k:v for v,k in id2label.items()}
    
    num_labels=len(label2id)
    
    return label2id,id2label,num_labels

def batch_to_device(tensor_dicts,device):
    for key in tensor_dicts.keys():
        tensor_dicts[key].to(device)


def read_kg(kg_path,kg):
    #'/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt'
    with open(kg_path) as f:
        lines=f.readlines()

    print('The number of triples: {}'.format(len(lines)))

    sub_map = defaultdict(set)#每一个头实体作为key，对应的所有一跳路径内的(关系，尾实体)作为value
    # so_map=defaultdict(set)#每一对(头实体，尾实体)作为key，对应的所有一跳的关系作为value
    sp_map=defaultdict(set)#每一个（头实体，关系）作为key，对应的仅能有一个答案

    alias_map=defaultdict(set)
    ent_to_relations=defaultdict(set)
    bad_line=0
    spliter='|||' if kg=='nlpcc' else '\t'
    for i in tqdm(range(len(lines))):
        line=lines[i]
        l = line.strip().split(spliter)
        s = l[0].strip()
        p = l[1].strip()
        o = l[2].strip()
        if s=='' or p=='' or o=='':
            bad_line+=1
            continue
        sub_map[s].add((p, o))
    #     so_map[(s,o)].add(p)
        sp_map[(s,p)].add(o)

        ent_to_relations[s].add(p)

        entity_mention=s
        if kg.lower()=='kgclue' and ('（' in s and '）' in s):
            entity_mention=s.split('（')[0]
            alias_map[entity_mention].add(s)
        if kg.lower()=='nlpcc' and ('(' in s and ')' in s):
            entity_mention=s.split('(')[0]
            alias_map[entity_mention].add(s)

        if p in ['别名','中文名','英文名','昵称','中文名称','英文名称','别称','全称','原名']:
            alias_map[entity_mention].add(o)
    return alias_map,sub_map