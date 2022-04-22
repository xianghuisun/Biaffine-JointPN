import json
from lib2to3.pgen2 import token
from tqdm import tqdm
import torch
import numpy as np
import random
import string
import logging
logger = logging.getLogger('main.utils')

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line.strip()))
    logger.info("原始数据{}中有{}个样本".format(data_path,len(data)))
    return data


SPIECE_UNDERLINE = '▁'


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

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

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False


def json2features(examples, tokenizer, max_seq_length=386, is_test=False):
    # {"doc_tokens": ["(", "1", ")", "填", "词", "(", "2", ")", "歌", "曲", "时", "长", "(", "3", ")", "导", "演", "(", "4", ")", "发", "行"      , "日", "期", "(", "5", ")", "中", "文", "名", "(", "6", ")", "出", "品", "公", "司", "(", "7", ")", "歌", "曲", "语", "言", "(", "8",       ")", "音", "乐", "风", "格", "(", "9", ")", "所", "属", "专", "辑", "(", "10", ")", "歌", "曲", "原", "唱", "(", "11", ")", "不", "匹", "配", "(", "12", ")", "谱", "曲"], 
    # "question": "你知道守望星光（韩玉玲、刘东考演唱歌曲）是哪家公司出品的吗？", 
    # "answer": "出品公司"      , "start_position": 33, "end_position": 36}
    ############################################Convert to features######################
    features=[]
    for example in tqdm(examples,total=len(examples),unit='example'):
        query_tokens=tokenizer.tokenize(example['question'])
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            #doc_tokens就是context_choice的list
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
        if example['end_position'] < len(example['doc_tokens']) - 1:
            tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
            #tok_start_position和tok_end_position只会比原始的start_position和end_position变的更大

        tokens = []
        segment_ids = []
        ######question
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ######choice
        for token in all_doc_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        ######convert to input_ids,attention_mask,token_type_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except:
            input_ids=input_ids[:max_seq_length]
            input_mask=input_mask[:max_seq_length]
            segment_ids=segment_ids[:max_seq_length]
            if end_position>=len(input_ids) or start_position>=len(input_ids):
                logger.info("Label position longer than sentence, discard the example")
                continue

        start_position = tok_start_position + 2 + len(query_tokens)
        end_position = tok_end_position + 2 + len(query_tokens)

        features.append({'tokens': tokens,
                        'input_ids': input_ids,
                        'attention_mask': input_mask,
                        'token_type_ids': segment_ids,
                        'start_position': start_position,
                        'end_position': end_position})

    logger.info('features num:', len(features))
    for _ in range(30):
        idx=random.randint(a=0,b=len(features)-1)
        tokens_idx=features[idx]['tokens']
        start_position_idx=features[idx]['start_position']
        end_position_idx=features[idx]['end_position']
        logger.info('tokens: {}'.format(str(tokens_idx)))
        logger.info('start_position and end_position: ({},{})'.format(start_position_idx,end_position_idx))
        logger.info('answer choice: {}'.format(str(tokens_idx[start_position_idx:end_position_idx+1])))
        logger.info('-'*100)
    #json.dump(features, open(output_file, 'w'))

    # if is_test==True:
    #     #对于测试集合，过滤其中含有英文的example
    #     tmp_features=[]
    #     for feature in features:
    #         tokens=feature['tokens']
    #         has_en_char=False
    #         for word in tokens:
    #             if word in ['[CLS]','[SEP]']:
    #                 continue
    #             word=word.replace('##','')
    #             for char in word:
    #                 if char in string.ascii_letters:
    #                     has_en_char=True
    #                     break

    #             if has_en_char==True:
    #                 break

    #         if has_en_char==False:
    #             tmp_features.append(feature)
    #     logger.info("In testing example, original count: {}, after filtering examples with en count: {}".format(len(features),len(tmp_features)))
    #     features=tmp_features
    
    logger.info("The number of output from json2features: {}".format(len(features)))
    return features

def restore_question_ans_choice(tokens):
    assert type(tokens)==list
    tokens=tokens[1:-1]
    question_tokens=tokens[:tokens.index('[SEP]')]
    choice_tokens=tokens[tokens.index('[SEP]')+1:]
    question=''
    for char in question_tokens:
        question+=char.replace('##','')
    choice=''
    for char in choice_tokens:
        choice+=char.replace('##','')
    return question,choice

def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    logger.info("missing keys:{}".format(missing_keys))
    logger.info('unexpected keys:{}'.format(unexpected_keys))
    logger.info('error msgs:{}'.format(error_msgs))