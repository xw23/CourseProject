import numpy as np
import math
import json

from nltk.tokenize import TweetTokenizer
from train import train_model

sarcasm_tokens = {}
non_sarcasm_tokens = {}
total_sarcasm_tokens = 0
total_non_sarcasm_tokens = 0

smoothing_parameter = 0.13
pos_prior = 0.78

response_list = []

context_list = []

label_response_dict = {}

word_frequency = {}
tokenizer = TweetTokenizer()

classified = []

sarcasm_tokens, non_sarcasm_tokens, total_sarcasm_tokens, total_non_sarcasm_tokens = train_model()


with open("data/test.jsonl", encoding="utf-8") as json_file:
    
    data = json.loads("[" + json_file.read().replace("}\n{", "},\n{") + "]")

    for p in data:
        response_list.append(p["response"].replace("@USER", ""))
        context_list.append(p["context"])
                   

for line in response_list:
    words = tokenizer.tokenize(line.lower())

    sarcasm_prob = 0
    non_sarcasm_prob = 0

    for word in words:
        sarcasm_count = 0
        non_sarcasm_count = 0

        if (word in sarcasm_tokens):
            sarcasm_count = sarcasm_tokens[word]

        sarcasm_prob += math.log((sarcasm_count + smoothing_parameter) / (total_sarcasm_tokens + (smoothing_parameter * len(sarcasm_tokens))))

        if (word in non_sarcasm_tokens):
            non_sarcasm_count = non_sarcasm_tokens[word]

        non_sarcasm_prob += math.log((non_sarcasm_count + smoothing_parameter) / (total_non_sarcasm_tokens + (smoothing_parameter * len(non_sarcasm_tokens))))


    sarcasm_prob += math.log(pos_prior)
    non_sarcasm_prob += math.log((1 - pos_prior))

    if (sarcasm_prob > non_sarcasm_prob):
        classified.append(1)
    else:
        classified.append(0)


num_of_output = len(classified)

YES = "SARCASM"
NO = "NOT_SARCASM"


output_file = open("answer.txt", "w")

for i in range(0, num_of_output):
    if (classified[i] == 1):
        output = "twitter_" + str(i+1) + "," + str(YES) + "\n"
    else:
        output = "twitter_" + str(i+1) + "," + str(NO) + "\n"

    output_file.write(output)

#output = "twitter_" + str(num_of_output) + "," + str(YES)

output_file.close()
