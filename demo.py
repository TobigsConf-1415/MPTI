import os
import sys
import argparse
import re
import pandas as pd
from numpy import argmax

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from utils import load, save, nameFromDB, toResponse, EOS, WELCOME, REGISTER, CAPITAL, PERSONALTXT, sample_sequence
from DialogRPT.src.dialogRPT import getIntegrated

from flask import Flask, request
app = Flask(__name__)


def resSherlock(inputs: str, params, name = "NoName", history = ""):
    history += inputs+ EOS
    if (history.count(EOS) > 4):
        history = history.split(EOS, maxsplit = 2)[-1]
    final, prob_gen, score_ranker, hyp = sherlockModel.predict(history, args.wt_ranker, params)[0]
    sherlock = hyp.replace("<NAME>", name)
    history += sherlock + EOS
    return sherlock, history

@app.route("/sherlock", methods=["POST"])
def sherlock():
	req = request.get_json()
	inputs = req["userRequest"]["utterance"]
	kakaoid = req["userRequest"]["user"]["id"]
	if inputs.startswith("!register"):
		name = inputs.split(" ")[-1]
		db[kakaoid] = (name, "")
		output = WELCOME.format(name)
	else:
		name, history = nameFromDB(db, kakaoid)
		if name==None:
			output = REGISTER
		else:
			output, history = resSherlock(inputs, params, name, history)
			db[kakaoid] = (name, history)
	return toResponse(output)
    

def response_ironman(inputs: str):
    
    input_vec = vectorizer.transform([inputs])
    similarity = cosine_similarity(input_vec, tfidf)[0]
    if max(similarity) > 0.5 :
        return rule_based_df['answer'][argmax(similarity)]
    else:
        history = [tokenizer_ironman.encode(inputs)]
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer_ironman, model_ironman)
        
        response = tokenizer_ironman.decode(out_ids, skip_special_tokens=True)
        response = re.split("([.?!’”]) ", response)
        response = [response[i-1] + response[i] for i in range(1, len(response), 2)] + [response[-1]]
        response = re.split("([.?!’,”]) ", ' '.join([r.capitalize() for r in response]))
        response_fin = []
        for word in [w for s in response for w in s.split()]:
            punc = ''
            if word.endswith(('.', '!','?')):
                punc = word[-1]; word = word[:-1]
            if word in CAPITAL.keys():
                response_fin.append(CAPITAL[word])
            else: response_fin.append(word)
            response_fin.append(punc)
            
        return re.sub(r' (?=\W)', '', ' '.join(response_fin))

@app.route("/ironman", methods=["POST"])
def ironman():
	req = request.get_json()
	inputs = req["userRequest"]["utterance"]
	output = response_ironman(inputs)
	return toResponse(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', '-db', type=str, default = 'db.pickle')
    parser.add_argument('--port', '-p', type=int, default = 3305)
    parser.add_argument('--ip', '-i', type=str, default = '0.0.0.0')
    parser.add_argument('--path_generator', '-pg', type=str, default = 'DialoGPT/output')
    parser.add_argument('--path_ranker', '-pr', type=str, default = "DialogRPT/restore/ensemble.yml")
    parser.add_argument('--path_transfo', '-pt', type=str, default = "TransferTransfo/default_gpt2")
    parser.add_argument('--path_rule', type=str, default = "TransferTransfo/rule_based_df.json")
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--beam', type=int, default=3)
    parser.add_argument('--wt_ranker', type=float, default=0.4)
    parser.add_argument('--topp', type=float, default=0.8)
    parser.add_argument('--max_t', type=int, default=15)
    parser.add_argument('--cpu', action='store_true', help='enables CUDA training')
    args = parser.parse_args()
    
    cuda = False if args.cpu else torch.cuda.is_available()
    tokenizer_ironman = GPT2Tokenizer.from_pretrained(args.path_transfo)
    model_ironman = GPT2LMHeadModel.from_pretrained(args.path_transfo)
    device = torch.device('cuda' if cuda else 'cpu')
    model_ironman.to(device)
    add_special_tokens_(model_ironman, tokenizer_ironman)
    personality = [tokenizer_ironman.encode(text) for text in PERSONALTXT]
    rule_based_df = pd.read_json(args.path_rule)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(rule_based_df['question']).toarray()

    
    db = load(args.dbname)
    sherlockModel = getIntegrated(args.path_ranker, args.path_generator, cuda)
    params = {'topk': args.topk, 'beam': args.beam, 'topp': args.topp, 'max_t':args.max_t}
    save(db, args.dbname)
    
    app.run(host=args.ip, port=args.port)
