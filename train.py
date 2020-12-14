import random
import json
from nltk.tokenize import TweetTokenizer


def train_model():
    label_list = []
    response_list = []

    context_list = []

    label_response_dict = {}

    total_sarcasm_tokens = 0
    total_non_sarcasm_tokens = 0

    sarcasm_tokens = {}
    non_sarcasm_tokens = {}



    with open("data/train.jsonl", encoding="utf-8") as json_file:
        
        data = json.loads("[" + json_file.read().replace("}\n{", "},\n{") + "]")

        # print(type(data))   # <class 'list'>
        temp = 0
        tokenizer = TweetTokenizer()


        for p in data:#["label"]:
            #print(p)
            #print(type(p))  # <class 'dict'>
            temp += 1
            label_list.append(p["label"])



            response = p["response"].replace("@USER", "")
            response_list.append(response)
            
            
            words = tokenizer.tokenize(response.lower())

            if(p["label"] == "SARCASM"):
                for word in words:
                    total_sarcasm_tokens += 1
                    if word in sarcasm_tokens:
                        sarcasm_tokens[word] += 1
                    else:
                        sarcasm_tokens[word] = 1
            else:
                for word in words:
                    total_non_sarcasm_tokens += 1
                    if word in non_sarcasm_tokens:
                        non_sarcasm_tokens[word] += 1
                    else:
                        non_sarcasm_tokens[word] = 1

            
            context_list.append(p["context"])



    return sarcasm_tokens, non_sarcasm_tokens, total_sarcasm_tokens, total_non_sarcasm_tokens

    











