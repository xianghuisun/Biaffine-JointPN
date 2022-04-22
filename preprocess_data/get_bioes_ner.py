import json
from utils import _tokenize_chinese_chars,is_whitespace, SPIECE_UNDERLINE,read_data,read_data_nlpcc

def convert_to_bioes_examples(data,file_path,repeat_limit=3):
    examples=[]
    for example in data:
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

    with open(file_path,'w') as f:
        for example in examples:
            doc_tokens,start_position,end_position=example['doc_tokens'],example['start_position'],example['end_position']
            seq_out=['O']*len(doc_tokens)
            seq_out[start_position]='B-ent'
            for i in range(start_position+1,end_position):
                seq_out[i]='I-ent'
            seq_out[end_position]='I-ent'
            if start_position==end_position:
                seq_out[start_position]='B-ent'#BIO or BIOES
            for token,tag in zip(doc_tokens,seq_out):
                f.write(token+'\t'+tag+'\n')
            f.write('\n')
        print("The number of examples : {}".format(len(examples)))

if __name__=="__main__":
    # train_data=read_data('../kgclue/train.json')
    # dev_data=read_data('../kgclue/dev.json')
    # test_data=read_data('../kgclue/test_public.json')
    # convert_to_bioes_examples(train_data,'/home/xhsun/Desktop/graduate_saved_files/bioes_ner/kgclue/train.txt')
    # convert_to_bioes_examples(dev_data,'/home/xhsun/Desktop/graduate_saved_files/bioes_ner/kgclue/dev.txt')
    # convert_to_bioes_examples(test_data,'/home/xhsun/Desktop/graduate_saved_files/bioes_ner/kgclue/test.txt')

    train_data=read_data_nlpcc('../nlpcc2018/nlpcc-iccpol-2016.kbqa.training-data')
    test_data=read_data_nlpcc('../nlpcc2018/nlpcc-iccpol-2016.kbqa.testing-data')
    convert_to_bioes_examples(train_data,'/home/xhsun/Desktop/graduate_saved_files/bioes_ner/nlpcc/train.txt')
    convert_to_bioes_examples(test_data,'/home/xhsun/Desktop/graduate_saved_files/bioes_ner/nlpcc/test.txt')