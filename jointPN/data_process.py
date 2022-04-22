import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
from copy import deepcopy
import re,os

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



def to_choice_format(example):
    question,candidate_rels,rel=example
    if rel not in candidate_rels:
        #答案提供的关系不在这个别名实体一级子图关系内，那么这个别名实体与问题构成负样本
        rel='不匹配'
    i=1
    context=''
    for i in range(len(candidate_rels)):
        context+='({})'.format(i+1)+candidate_rels[i]
    #question=question+'。'+context
    #answer_start=question.find(rel)
    answer_start=context.find(rel)
    return {"question":question,'context':context,'answer_start':answer_start,'answer':rel}

def get_choice_examples(examples,alias_map,sub_map,kg='kgclue'):
    bad_examples=[]
    new_examples=[]
    for example in examples:
        question,triple=example['question'],example['answer']
        head,rel,tail=triple.split('|||')
        rel=rel.strip()
        head=head.strip()
        topic_entity=head

        if kg=='kgclue' and ('（' in head and '）' in head):
            topic_entity=head.split('（')[0]
        if kg=='nlpcc' and ('(' in head and ')' in head):
            topic_entity=head.split('(')[0]

        if topic_entity not in question:
            bad_examples.append(example)#找不到主题实体
            
        alias_entity_list=list(alias_map[topic_entity])
        if topic_entity not in alias_entity_list:
            alias_entity_list.append(topic_entity)
            
        for alias_entity in alias_entity_list:
            candidate_rels=[]
            for r,t in sub_map[alias_entity]:
                candidate_rels.append(r)
            candidate_rels.append('不匹配')
            random.shuffle(candidate_rels)
            if candidate_rels!=['不匹配']:
                new_examples.append([question.replace(topic_entity,alias_entity),candidate_rels,rel])

    choice_examples=[]
    for example in new_examples:
        choice_examples.append(to_choice_format(example))

    return choice_examples

# {"question":'你知道守望星光（盛一伦演唱歌曲）是哪家公司出品的吗？。(1)歌曲时长(2)发行时间(3)填词(4)谱曲(5)中文名(6)歌曲语言(7)不匹配(8)歌曲原唱(9)编曲',
# "context":(1)歌曲时长(2)发行时间(3)填词(4)谱曲(5)中文名(6)歌曲语言(7)不匹配(8)歌曲原唱(9)编曲,
# "answer_start":43, "answer":"不匹配"}

# {'question': '你知道守望星光（韩玉玲、刘东考演唱歌曲）是哪家公司出品的吗？。(1)歌曲时长(2)谱曲(3)歌曲语言(4)所属专辑(5)出品公司(6)发行日期(7)音乐风格(8)中文名(9)导演(10)歌曲原唱(11)不匹配(12)填词',
# 'context': '(1)歌曲时长(2)谱曲(3)歌曲语言(4)所属专辑(5)出品公司(6)发行日期(7)音乐风格(8)中文名(9)导演(10)歌曲原唱(11)不匹配(12)填词',
# 'answer_start': 60,
# 'answer': '出品公司'}

# {"doc_tokens": ["(", "1", ")", "填", "词", "(", "2", ")", "歌", "曲", "时", "长", "(", "3", ")", "导", "演", "(", "4", ")", "发", "行"      , "日", "期", "(", "5", ")", "中", "文", "名", "(", "6", ")", "出", "品", "公", "司", "(", "7", ")", "歌", "曲", "语", "言", "(", "8",       ")", "音", "乐", "风", "格", "(", "9", ")", "所", "属", "专", "辑", "(", "10", ")", "歌", "曲", "原", "唱", "(", "11", ")", "不", "匹
#       ", "配", "(", "12", ")", "谱", "曲"], "question": "你知道守望星光（韩玉玲、刘东考演唱歌曲）是哪家公司出品的吗？", "answer": "出品公司"      , "start_position": 33, "end_position": 36}

def process_choice_examples(data, max_seq_length=386, repeat_limit=3):
    # to examples
    ############################################Convert to examples######################
    examples=[]
    max_length=0
    for example in data:
        question,answer_start,ans_text=example['question'],example['answer_start'],example['answer']
        assert type(question)==str
        question=question.strip()
        #context = question
        context = example['context']
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

        start_position_final = None
        end_position_final = None
        count_i = 0
        #start_position = question.find(ans_text)
        start_position = context.find(ans_text)
        assert start_position==answer_start

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

        if len(doc_tokens)>max_length:
            max_length=len(doc_tokens)
    
    #max_seq_length=min(max_length+20,max_seq_length)
    print('examples num:', len(examples),' max_seq_length: ', max_seq_length)

    for _ in range(10):
        idx=random.randint(a=0,b=len(examples)-1)
        for key,value in examples[idx].items():
            print(key,' : ',value)
        print('-'*100)
    
    return examples

def write_to_file(src_path,des_path,alias_map,sub_map,kg='kgclue',is_examples=False):
    if is_examples==False:
        train_examples=read_data(src_path)
    else:
        train_examples=src_path

    choice_examples=get_choice_examples(train_examples,alias_map,sub_map,kg=kg)
    examples=process_choice_examples(choice_examples, max_seq_length=386, repeat_limit=3)
    with open(des_path,'w') as f:
        for e in examples:
            f.write(json.dumps(e,ensure_ascii=False)+'\n')


def preprocess_to_mrc(knowledge_graph,kg='kgclue'):
    alias_map,sub_map=read_kg(os.path.join(knowledge_graph,'Knowledge.txt'),kg=kg)

    write_to_file(src_path=os.path.join(knowledge_graph,'train.json'),
                des_path=os.path.join(knowledge_graph,'train_mrc.txt'),
                alias_map=alias_map,
                sub_map=sub_map,
                kg=kg)
    write_to_file(src_path=os.path.join(knowledge_graph,'dev.json'),
                des_path=os.path.join(knowledge_graph,'dev_mrc.txt'),
                alias_map=alias_map,
                sub_map=sub_map,
                kg=kg)
    write_to_file(src_path=os.path.join(knowledge_graph,'test_public.json'),
                des_path=os.path.join(knowledge_graph,'test_public_mrc.txt'),
                alias_map=alias_map,
                sub_map=sub_map,
                kg=kg)