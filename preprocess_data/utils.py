import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
from copy import deepcopy
import re

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data

def read_data_nlpcc(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    i=0
    examples=[]
    while i+2<len(lines):
        question=lines[i].strip().split('\t')[1]
        i+=1
        triple=lines[i].strip().split('\t')[1]
        i+=1
        try:
            answer=lines[i].strip().split('\t')[1]
        except:
            print(lines[i])
        i+=2
        if triple.split('|||')[2].strip()==answer and len(answer)>=1:
            examples.append({"question":question,'answer':triple})
    print("The number examples: {} from {}".format(len(examples),data_path))
    return examples
    
SPIECE_UNDERLINE = '▁'
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
    
def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def is_fuhao(c):
    if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
            or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
            or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
            or c == '‘' or c == '’':
        return True
    return False
    
def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
            if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                output.append(SPIECE_UNDERLINE)
            output.append(char)
            output.append(SPIECE_UNDERLINE)
        else:
            output.append(char)
    return "".join(output)

def read_kg(kg_path,kg):
    #'/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt'
    with open(kg_path) as f:
        lines=f.readlines()
    assert kg.lower() in ['kgclue','nlpcc']
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