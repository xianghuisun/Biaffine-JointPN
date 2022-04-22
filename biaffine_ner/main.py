from cProfile import label
from email.policy import strict
from importlib.resources import path
import json
from operator import imod
import os, importlib
import sys
from typing import Any
from transformers import AdamW,get_linear_schedule_with_warmup 
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import argparse
import data_conll,tools
import random
from model import myModel,Span_loss,metrics_span
from data_process import preprocess_to_bio

import logging
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG) 
file_handler = logging.FileHandler('log.txt',mode='w')
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(
        logging.Formatter(
                fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        )
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
        logging.Formatter(
                fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        )
logger.addHandler(console_handler)

from tqdm import tqdm
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# System based
random.seed(seed)
np.random.seed(seed)

#global setting
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
device = torch.device('cuda')

def train(args,
        tar_train_data,
        tar_test_data,
        ner_label2id,
        tokenizer):

    kg_path=os.path.join(args.knowledge_graph,'Knowledge.txt')
    logger.info('Reading knowledge graph from {}'.format(kg_path))
    alias_map,sub_map=tools.read_kg(kg_path,kg=args.kg)
    ner_num_label=len(ner_label2id)
    model = myModel(args,ner_num_label=ner_num_label,device=device).to(device)

    model.to(device)
    # #---optimizer---
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    num_training_steps=len(tar_train_data)*args.batch_size*args.epoch
    warmup_steps=num_training_steps*args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=num_training_steps)

    #---loss function---
    ner_span_loss_func = Span_loss(ner_num_label,class_weight=[1]+[4]*(ner_num_label-1)).to(device)
    span_acc = metrics_span().to(device)

    global_step=0
    best=0
    training_loss=0.0
    count=0
    for epoch in range(args.epoch):
        model.train()
        model.zero_grad()
        for tar_item in tqdm(tar_train_data,total=len(tar_train_data),unit='batches'):
            global_step+=1
            tar_input_ids, tar_attention_mask, tar_token_type_ids = tar_item["input_ids"], tar_item["attention_mask"], tar_item["token_type_ids"]
            tar_ner_span_label, tar_ner_span_mask = tar_item['span_label'], tar_item["span_mask"]
            tar_ner_span_logits = model( 
                input_ids = tar_input_ids.to(device), 
                attention_mask = tar_attention_mask.to(device),
                token_type_ids = tar_token_type_ids.to(device),
            )
            ner_loss = ner_span_loss_func(tar_ner_span_logits, tar_ner_span_label.to(device), tar_ner_span_mask.to(device))
            loss=ner_loss.float().mean().type_as(ner_loss)
            training_loss+=loss.item()
            count+=1
            loss.backward()
            #training_loss+=loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        model.eval()
        recall,precise,span_f1=evaluate_ner(args=args,
                                            model=model,
                                            label2id=ner_label2id,
                                            tokenizer=tokenizer,
                                            alias_map=alias_map,
                                            sub_map=sub_map)
        model.train()

        logger.info('Evaluating the model...')
        logger.info('epoch %d, loss %.4f, recall %.4f, precise %.4f, span_f1 %.4f'% (epoch,training_loss/count,recall,precise,span_f1))
        training_loss=0.0
        count=0
        
        if best < span_f1:
            best=span_f1
            torch.save(model.state_dict(), f=os.path.join(args.checkpoints,'best-model.pt'))
            logger.info('save the best model in {}'.format(args.checkpoints))

def evaluate_ner(args,model,label2id,tokenizer,alias_map,sub_map):
    id2label={k:v for v,k in label2id.items()}
    ner_num_label=len(label2id)
    sentences,entities=data_conll.load_ner_data(os.path.join(args.knowledge_graph,'test_public.txt'))
    logger.info("Evaluating the model...")

    examples=[]
    for sentence,entity in tqdm(zip(sentences,entities),total=len(sentences),unit='sentence'):
        assert len(sentence.split(' '))==len(entity)
        example={'sentence':sentence.split(' ')}#,'entity':entity}
        inputs=tokenizer(text=sentence)
        input_ids,attention_mask,token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        #example['input_ids']=input_ids
        #example['attention_mask']=attention_mask
        example['wordpiece_sentence']=tokenizer.convert_ids_to_tokens(input_ids)
        span_label,ner_relation=data_conll.get_span_label(sentence,tokenizer,attention_mask,relation=entity,label2id=label2id)
        #example['span_label']=span_label
        example['ner_relation']=ner_relation
        piece_length=len(attention_mask)
        example['piece_length']=piece_length
        with torch.no_grad():
            input_ids=torch.LongTensor([input_ids])
            attention_mask=torch.LongTensor([attention_mask])
            token_type_ids=torch.LongTensor([token_type_ids])
            ner_span_logits= model( 
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                ) 
        #ner_span_logits=ner_span_logits[0]
        ner_span_logits=torch.nn.functional.softmax(ner_span_logits[0],dim=-1)
        assert ner_span_logits.size()==(piece_length,piece_length,ner_num_label)
        predict_ids=torch.argmax(ner_span_logits,dim=-1)
        assert predict_ids.size()==(piece_length,piece_length)
        predict_ids=predict_ids.cpu().tolist()
        example['predict_span']=[]
        tmp_records={}
        for start_id in range(1,piece_length-1):
            for end_id in range(start_id,piece_length-1):
                if predict_ids[start_id][end_id]!=0:
                    example['predict_span'].append((start_id,
                                                    end_id,
                                                    id2label[predict_ids[start_id][end_id]],
                                                    float(ner_span_logits[start_id,end_id,predict_ids[start_id][end_id]])))
        
        examples.append(example)

    with open("./examples.txt",'w') as f:
        for example in examples:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')

    examples=[]
    with open("./examples.txt") as f:
        lines=f.readlines()
        for line in lines:
            examples.append(json.loads(line.strip()))

    new_examples=[]
    bad_examples=[]
    for example in examples:
        if example['wordpiece_sentence'][1:-1]==example['sentence']:
            new_examples.append(example)
        else:
            bad_examples.append(example)

    number_span_right=0
    number_span_wrong=0
    number_span_totoal=0

    number_span_totoal_2=0
    number_span_right_2=0
    number_span_wrong_2=0
    #############################################处理句子中含有英文和数字的情况######################
    for example in bad_examples:
        ner_relation=example['ner_relation']
        predict_span=example['predict_span']
        assert len(ner_relation)<=1
        if example['ner_relation']==[]:
            continue
        ner_relation=example['ner_relation'][0]
        if ner_relation[-1]=='O':
            continue
        number_span_totoal_2+=1
        for each in predict_span:
            s,e,score,_=each
            each=[s,e,score]
            if each == ner_relation:
                number_span_right_2 += 1#当前预测的span完全正确
            else:
                number_span_wrong_2 += 1

    ##############################################处理正常的全是中文的情况######################
    for example in new_examples:
        if example['ner_relation']==[]:
            continue
        ner_relation=example['ner_relation'][0]
        wordpiece_sentence=example['wordpiece_sentence']
        if ner_relation[2]!='ent':
            continue
        number_span_totoal+=1
        for start_id,end_id,_,_ in example['predict_span']:
            ent_tokens=''.join(wordpiece_sentence[start_id:end_id+1])
            if ent_tokens in alias_map or ent_tokens in sub_map:
                number_span_right+=1
            else:
                number_span_wrong+=1
    ###########################################################################################
    recall=(number_span_right+number_span_right_2)/(number_span_totoal+number_span_totoal_2)
    precision=(number_span_right+number_span_right_2)/(number_span_right+number_span_wrong+number_span_right_2+number_span_wrong_2)
    f1=2*precision*recall/(precision+recall)

    logger.info('recall : {}, precision : {}, f1 : {}'.format(recall,precision,f1))
    return recall,precision,f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_graph", type=str, required=True,help="train file")
    parser.add_argument("--kg", type=str, default="kgclue")
    
    parser.add_argument("--checkpoints", type=str, required=True,help="output_dir")
    parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=512,help="lstm_hidden_size")
    parser.add_argument("--to_biaffine_size", type=int, default=128,help="to_biaffine_size")
    parser.add_argument("--max_length", type=int, default=196,help="max_length")
    parser.add_argument("--epoch", type=int, default=100,help="epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,help="learning_rate")
    parser.add_argument("--finetune_path", type=str, default="",help="output_dir")
    parser.add_argument("--bert_model_path", type=str, required=True,help="bert_model_path")
    parser.add_argument("--clip_norm", type=float, default=1,help="clip_norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,help="warmup proportion")
    parser.add_argument("--num_workers", type=int, default=8,help='num_workers')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    args = parser.parse_args()
    preprocess_to_bio(knowledge_graph=args.knowledge_graph)
    ner_train_path=os.path.join(args.knowledge_graph,'train.txt')
    ner_test_path=os.path.join(args.knowledge_graph,'test_public.txt')

    for k,v in args.__dict__.items():
        logger.info("{} : {}".format(str(k),str(v)))

    tokenizer=tools.get_tokenizer(bert_model_path=args.bert_model_path)
    ner_label2id=tools.generate_label2id(file_path=ner_train_path)
    ner_label2id=tools.process_nerlabel(label2id=ner_label2id)
    logger.info("Ner label2id : {}".format(json.dumps(ner_label2id)))

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints,exist_ok=True)

    tar_train_data  = data_conll.yield_data(args=args,
                                            file_path=ner_train_path, 
                                            tokenizer=tokenizer, 
                                            mode='ner', 
                                            label2id=ner_label2id) #ner_train -> whole span, sub span, non span 
    tar_test_data  = data_conll.yield_data(args=args,
                                            file_path=ner_test_path, 
                                            tokenizer=tokenizer, 
                                            mode='ner', 
                                            label2id=ner_label2id, 
                                            is_training=False) #ner_test -> whole span, non span

    train(args=args,
        tar_train_data=tar_train_data,
        tar_test_data=tar_test_data,
        ner_label2id=ner_label2id,
        tokenizer=tokenizer)

if __name__=="__main__":
    main()
