from collections import defaultdict
import imp
from tqdm import tqdm
import re
import json
from copy import deepcopy
import random
import torch
import sys,os
import pandas as pd
import numpy as np
from collections import Counter
import argparse

from copy import deepcopy


def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data
    
SPIECE_UNDERLINE = '▁'
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False

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


def convert_to_bio(train_data,write_file,repeat_limit=3):
    examples=[]
    for example in train_data:
        question,answer=example['question'],example['answer']
        triplet=answer.strip().split('|||')
        assert len(triplet)==3
        entity=triplet[0]
        assert type(question)==str
        question=question.strip()
        entity=entity.strip()    
        if '（' in entity and entity.endswith('）'):
            entity=entity.split('（')[0]
            
        context = question
        context_chs = _tokenize_chinese_chars(context)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_chs:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            if c != SPIECE_UNDERLINE:
                char_to_word_offset.append(len(doc_tokens) - 1)

        ans_text = entity

        start_position_final = None
        end_position_final = None
        count_i = 0
        start_position = question.find(ans_text)
        if start_position==-1:
            #print(example)
            continue
            #raise Exception("check")
            
        end_position = start_position + len(ans_text) - 1
        while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
            start_position -= 1
            end_position -= 1
            count_i += 1

        while context[start_position] == " " or context[start_position] == "\t" or \
                context[start_position] == "\r" or context[start_position] == "\n":
            start_position += 1

        start_position_final = char_to_word_offset[start_position]
        end_position_final = char_to_word_offset[end_position]

        if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
            start_position_final += 1

        examples.append({'doc_tokens': doc_tokens,
                        'question': question,
                        'answer': ans_text,
                        'start_position': start_position_final,
                        'end_position': end_position_final})

    print("{} samples will be written in {}".format(len(examples),write_file))
    with open(write_file,'w') as f:
        for example in examples:
            doc_tokens,start_position,end_position=example['doc_tokens'],example['start_position'],example['end_position']
            seq_out=['O']*len(doc_tokens)
            seq_out[start_position]='B-ent'
            for i in range(start_position+1,end_position):
                seq_out[i]='I-ent'
            seq_out[end_position]='E-ent'
            if start_position==end_position:
                seq_out[start_position]='S-ent'
            for token,tag in zip(doc_tokens,seq_out):
                f.write(token+'\t'+tag+'\n')
            f.write('\n')

def preprocess_to_bio(knowledge_graph):
    train_file=os.path.join(knowledge_graph,'train.json')
    dev_file=os.path.join(knowledge_graph,'dev.json')
    test_public_file=os.path.join(knowledge_graph,'test_public.json')

    train_data=read_data(train_file)
    dev_data=read_data(dev_file)
    test_data=read_data(test_public_file)

    convert_to_bio(train_data,write_file=os.path.join(knowledge_graph,'train.txt'))
    convert_to_bio(dev_data,write_file=os.path.join(knowledge_graph,'dev.txt'))
    convert_to_bio(test_data,write_file=os.path.join(knowledge_graph,'test_public.txt'))

# if __name__=="__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--knowledge_graph", type=str, required=True)
#     args = parser.parse_args()

#     train_file=os.path.join(args.knowledge_graph,'train.json')
#     dev_file=os.path.join(args.knowledge_graph,'dev.json')
#     test_public_file=os.path.join(args.knowledge_graph,'test_public.json')

#     train_data=read_data(train_file)
#     dev_data=read_data(dev_file)
#     test_data=read_data(test_public_file)

#     convert_to_bio(train_data,write_file=os.path.join(args.knowledge_graph,'train.txt'))
#     convert_to_bio(dev_data,write_file=os.path.join(args.knowledge_graph,'dev.txt'))
#     convert_to_bio(test_data,write_file=os.path.join(args.knowledge_graph,'test_public.txt'))


