import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
logger=logging.getLogger("main.model")

class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        '''
        (bsz,max_length,dim) x.size()==y.size()
        '''
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        #(bsz,max_length,max_length,num_labels)
        return bilinar_mapping

class myModel(nn.Module):
    def __init__(self, args, ner_num_label, device = torch.device('cuda')):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained(args.bert_model_path)
        #print(len(tokenizer))
        
        self.lstm_hidden_size=args.lstm_hidden_size
        self.ner_start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*args.lstm_hidden_size, out_features=args.to_biaffine_size),
                                            torch.nn.ReLU())
        self.ner_end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*args.lstm_hidden_size, out_features=args.to_biaffine_size),
                                            torch.nn.ReLU())        
        

        self.ner_biaffne_layer = biaffine(args.to_biaffine_size, out_size=ner_num_label)

        self.lstm=torch.nn.LSTM(input_size=self.bert_encoder.config.hidden_size,hidden_size=args.lstm_hidden_size, \
                        num_layers=1,batch_first=True, \
                        dropout=0.5,bidirectional=True)
        
        self.relu=torch.nn.ReLU()
        self.device=device
        #self.ner_logits_layer=torch.nn.Linear(in_features=768, out_features=ner_num_label)
        

    def forward(self, input_ids, attention_mask, token_type_ids, word_seq_lens=None):
        '''
        word_seq_len指的是单词级别的句子长度，不是wordpiece，需要传入给LSTM使用. size()==(batch_size,)
        '''
        bert_output = self.bert_encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            token_type_ids=token_type_ids) 
        encoder_rep = bert_output[0]
        batch_size,max_length,_=encoder_rep.size()
        #################################################LSTM####################################################
        if word_seq_lens is None:
            word_seq_lens=torch.sum(attention_mask,dim=1)
            max_word_len=word_seq_lens.long().max().item()
        #print('encoder_rep size : ',encoder_rep.size())
        #print(word_seq_lens)
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)#sorted_seq_len代表排序后的句子长度，permIdx代表最长句子的id
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_encoder_rep = encoder_rep[permIdx]
        packed_words=pack_padded_sequence(sorted_encoder_rep,sorted_seq_len.cpu(),batch_first=True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True) 

        lstm_out = lstm_out[recover_idx]#(batch_size,max_word_len)
        #print('lstm out size : ', lstm_out.size())
        if max_word_len<max_length:
            pad_embeddings=torch.zeros(batch_size,max_length-max_word_len,self.lstm_hidden_size*2).to(self.device)
            lstm_out=torch.cat((lstm_out,pad_embeddings),dim=1)
        assert lstm_out.size()==(batch_size,max_length,self.lstm_hidden_size*2)
        ###########################################################################################################
        
        ner_start_rep = self.ner_start_layer(lstm_out) 
        ner_end_rep = self.ner_end_layer(lstm_out) 
        
        ner_span_logits = self.ner_biaffne_layer(ner_start_rep, ner_end_rep)
        ner_span_logits = ner_span_logits.contiguous()        
        # ner_span_logits = self.ner_biaffne_layer(ner_start_rep, ner_end_rep)
        # ner_span_logits = ner_span_logits.contiguous()      
        # print("input ids : ",input_ids.size())#(bsz,max_length)
        # print("input mask : ",attention_mask.size())
        # print("input seg : ",token_type_ids.size())
        # print("encoder rep : ",encoder_rep.size())#(bsz,max_length,bert_config.hidden_size*2)，因为是双向的
        # print("dp_start_rep : ",dp_start_rep.size())#(bsz,max_length,out_features=128)
        # print("dp_end_rep : ",dp_end_rep.size())
        # print("ner_start_rep : ",ner_start_rep.size())#(bsz,max_length,out_features=128)
        # print("ner_end_rep : ",ner_end_rep.size())
        # print("concat_ner_start_rep : ",concat_ner_start_rep.size())#(bsz,max_length,out_features*3)
        # print("concat_ner_end_rep : ",concat_ner_end_rep.size())
        # print("dp_span_logits : ",dp_span_logits.size())#(bsz,)
        # print("ner_span_logits : ",ner_span_logits.size())

        return ner_span_logits
        #return None, ner_span_logits, None, None
        # span_logits = self.relu(span_logits)
        # span_logits = self.logits_layer(span_logits)

        #span_prob = torch.nn.functional.softmax(span_logits, dim=-1)


class Span_loss(nn.Module):
    def __init__(self, num_label, class_weight=None):
        super().__init__()
        self.num_label = num_label
        if class_weight != None:
            self.class_weight=torch.FloatTensor(class_weight)
            logger.info("Class weight : {}".format(self.class_weight.cpu().tolist()))
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)#reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()#reduction='mean'

    def forward(self,span_logits,span_label,span_mask):
        '''
        span_logits.size()==(bsz,max_length,max_length,num_labels)
        span_label.size()==span_mask.size()==(bsz,max_length,max_length)
        span_label只有(i,j)位置是1,(i,j)代表句子的第i个位置与第j个位置之间的span是一个ent
        span_mask左上半部分是1
        '''
        #print(span_mask.size(),'spanmask.size()')
        mask_pad=span_mask.view(-1)==1
        span_label = span_label.view(size=(-1,))[mask_pad]#(bsz*max_length*max_length,)
        span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]#(bsz*max_length*max_length,num_labels)
        span_loss = self.loss_func(input=span_logits, target=span_label)#(bsz*max_length*max_length,)
        
        # print("span_logits : ",span_logits.size())
        # print("span_label : ",span_label.size())
        # print("span_mask : ",span_mask.size())
        # print("span_loss : ",span_loss.size())

        # start_extend = span_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = span_mask.unsqueeze(1).expand(-1, seq_len, -1)
        # span_mask = span_mask.view(size=(-1,))#(bsz*max_length*max_length,)
        # span_loss *=span_mask
        #avg_se_loss = torch.sum(span_loss) / span_mask.size()[0]
        # avg_se_loss = torch.sum(span_loss) / torch.sum(span_mask).item()
        # # avg_se_loss = torch.sum(sum_loss) / bsz
        # return avg_se_loss
        return span_loss

class metrics_span(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, span_mask):
        '''
        logits.size()==(bsz,max_length,max_length,num_labels) .score for each span
        labels.size()==(bsz,max_length,max_length)
        span_mask.size()==(bsz,max_length,max_length)#下三角全1是，上三角超出句子实际长度的位置也是1
        '''
        (bsz,max_length,max_length,num_labels)=logits.size()
        assert labels.size()==(bsz,max_length,max_length)==span_mask.size()
        #print(bsz,max_length,max_length,num_labels)

        span_mask.unsqueeze_(-1)#(bsz,max_length,max_length,1)
        assert span_mask.size()==(bsz,max_length,max_length,1)
        logits*=span_mask
        logits = torch.argmax(logits,dim=-1)#(bsz,max_length,max_length)
        assert logits.size()==(bsz,max_length,max_length)#label_id for each span

        logits=logits.view(size=(-1,)).float()
        labels=labels.view(size=(-1,)).float()

        ones=torch.ones_like(logits)
        zero=torch.zeros_like(logits)
        y_pred=torch.where(logits<1,zero,ones)
        y_pred=torch.triu(y_pred)#extract upper triangle matrix

        ones=torch.ones_like(labels)
        zero=torch.zeros_like(labels)
        y_true=torch.where(labels<1,zero,ones)#only golden span position is 1, otherwise positions are zeros

        corr=torch.eq(logits,labels).float()
        corr=torch.mul(corr,y_true)
        
        recall=torch.sum(corr)/(torch.sum(y_true)+1e-8)
        
        precision=torch.sum(corr)/(torch.sum(y_pred)+1e-8)
        
        f1=2*recall*precision/(recall+precision+1e-8)

        if torch.sum(labels)==0 and torch.sum(logits)==0:
            return 1.0,1.0,1.0

        return recall.item(), precision.item(), f1.item()
