from ast import arg
import torch.nn as nn
from models import BertForQuestionAnswering
import json,random
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import json2features, read_data, restore_question_ans_choice
from data_process import preprocess_to_mrc

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

seed_=42
random.seed(seed_)
np.random.seed(seed_)
torch.manual_seed(seed_)
torch.cuda.manual_seed_all(seed_)
device=torch.device('cuda')

def evaluate(model, args, eval_features):
    logger.info("***** Eval({} examples are evaluating...) *****".format(len(eval_features)))
    all_input_ids = torch.LongTensor([f['input_ids'] for f in eval_features])
    all_attention_mask = torch.LongTensor([f['attention_mask'] for f in eval_features])
    all_token_type_ids = torch.LongTensor([f['token_type_ids'] for f in eval_features])
    all_start_positions = [f['start_position'] for f in eval_features]
    all_end_positions = [f['end_position'] for f in eval_features]
    
    all_tokens=[f['tokens'] for f in eval_features]

    eval_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_start_predictions=[]
    all_end_predictions=[]
    logger.info("Start evaluating")
    for step, batch in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader),unit='batch'):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids = batch
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch_start_logits+=(1.0-attention_mask)*(-1e5)
            batch_end_logits+=(1.0-attention_mask)*(-1e5)

        batch_start_probs=torch.softmax(batch_start_logits,dim=1)
        batch_end_probs=torch.softmax(batch_end_logits,dim=1)#(batch_size,max_seq_length)
        for i in range(len(batch_start_probs)):
            pos_s=torch.argmax(batch_start_probs[i]).item()
            end_probs=batch_end_probs[i]
            for j in range(0,pos_s+1):
                end_probs[j]+=(-1e5)
            pos_e=torch.argmax(end_probs).item()
            assert pos_e>pos_s
            all_start_predictions.append(pos_s)
            all_end_predictions.append(pos_e)

    correct=0
    assert len(all_start_positions)==len(all_start_predictions)
    with open('predict.txt','w') as f:
        f.write("golden_start_pos"+'\t'+'golden_end_pos'+'\t'+'predict_start_pos'+'\t'+'predict_end_pos'+'\n')
        for golden_start_pos,golden_end_pos,predict_start_pos,predict_end_pos,tokens in zip(all_start_positions,
                                                                                        all_end_positions,
                                                                                        all_start_predictions,
                                                                                        all_end_predictions,
                                                                                        all_tokens):
            question,choice=restore_question_ans_choice(tokens=tokens)
            tmp={"question": question,
                "choice": choice,
                'golden_positions':[golden_start_pos,golden_end_pos],
                'golden_answer': ''.join(tokens[golden_start_pos:golden_end_pos+1]),
                'predict_positions':[predict_start_pos,predict_end_pos],
                'predict_answer': ''.join(tokens[predict_start_pos:predict_end_pos+1]),}
            f.write(json.dumps(tmp,ensure_ascii=False)+'\n')

    for i in range(len(all_start_positions)):
        start_pos=all_start_positions[i]
        end_pos=all_end_positions[i]

        start_pred=all_start_predictions[i]
        end_pred=all_end_predictions[i]

        if start_pos<=start_pred and end_pred<=end_pos:
            correct+=1

    return correct/len(all_start_positions)



def train(args, model, optimizer, scheduler, train_dataloader, dev_features, steps_per_epoch):

    logger.info('***** Training *****')
    model.train()
    global_steps = 1
    best_acc=0.70
    for i in range(int(args.epochs)):
        logger.info('Starting epoch {}'.format(i + 1))
        total_loss = 0
        iteration = 1
        model.train()
        with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
                loss = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            start_positions=start_positions, 
                            end_positions=end_positions)

                total_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                pbar.update(1)
                loss.backward()

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1
                iteration += 1

        acc=evaluate(model, args, eval_features=dev_features)
        if acc>best_acc:
            logger.info('Previous best acc is {}, current acc is {}'.format(best_acc,acc))
            best_acc=acc
            torch.save(model.state_dict(), f=args.checkpoints)
            logger.info('save the best model in {}'.format(args.checkpoints))
            
        logger.info("Accuracy: {}".format(acc))


def main(args):
    train_data=read_data(os.path.join(args.knowledge_graph,'train_mrc.txt'))
    dev_data=read_data(os.path.join(args.knowledge_graph,'dev_mrc.txt'))
    test_data=read_data(os.path.join(args.knowledge_graph,'test_public_mrc.txt'))
    tokenizer=BertTokenizer.from_pretrained(args.bert_model_path)
    train_features=json2features(examples=train_data,tokenizer=tokenizer,max_seq_length=args.max_seq_length)
    dev_features=json2features(examples=dev_data,tokenizer=tokenizer,max_seq_length=args.max_seq_length,is_test=True)
    test_features=json2features(examples=test_data,tokenizer=tokenizer,max_seq_length=args.max_seq_length,is_test=True)

    # with open("/home/xhsun/Desktop/graduate_saved_files/joint_choice/kgclue/train_features.json",'w') as f:
    #     for feature in train_features:
    #         f.write(json.dumps(feature,ensure_ascii=False)+'\n')

    bert_config=BertConfig.from_pretrained(args.bert_model_path)
    bert_config.__dict__.update({"num_labels":2})
    model=BertForQuestionAnswering(config=bert_config)
    if args.finetune_path!="":
        logger.info("Loading finetune_path from {}".format(args.finetune_path))
        load_res=model.load_state_dict(torch.load(args.finetune_path,map_location='cpu'))
        logger.info(str(load_res))

    model.to(device)
    logger.info("Evaluate acc in first time: {}".format(evaluate(model, args, test_features)))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    #train_features=test_features
    
    steps_per_epoch = len(train_features) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps*args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler =  get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps=warmup_steps,
                                    num_training_steps=total_steps)

    all_input_ids = torch.LongTensor([f['input_ids'] for f in train_features])
    all_attention_mask = torch.LongTensor([f['attention_mask'] for f in train_features])
    all_token_type_ids = torch.LongTensor([f['token_type_ids'] for f in train_features])
    all_start_positions = torch.LongTensor([f['start_position'] for f in train_features])
    all_end_positions = torch.LongTensor([f['end_position'] for f in train_features])

    train_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_start_positions, all_end_positions)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    train(args, model, optimizer, scheduler, train_dataloader, dev_features=test_features, steps_per_epoch=steps_per_epoch)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    ## As the dataset or task changes
    parser.add_argument("--knowledge_graph", type=str, required=True,help="train file")
    parser.add_argument("--kg", type=str, default="kgclue")

    parser.add_argument("--checkpoints", type=str, required=True,help="output_dir")
    parser.add_argument("--finetune_path", type=str, default="",help="output_dir")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", default=386, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
                        
    ## Maybe will change
    parser.add_argument("--bert_model_path", type=str, required=True,help="bert_model_path")
    parser.add_argument("--do_lowercase", default=True)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()

    preprocess_to_mrc(knowledge_graph=args.knowledge_graph,kg=args.kg)

    main(args)
