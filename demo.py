import os
import sys
import argparse
import re
import app
import pandas as pd
from numpy import argmax

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from train_ironman import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import load, save, nameFromDB, toResponse, EOS, WELCOME, REGISTER, CAPITAL, PERSONALTXT
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

@app.route("/ironman", methods=["POST"])
def ironman():
	req = request.get_json()
	inputs = req["userRequest"]["utterance"]
	output = app.response_ironman(inputs)
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
