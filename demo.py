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

from TransferTransfo.train_ironman import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import load, save, nameFromDB, toResponse, EOS, WELCOME, REGISTER, CAPITAL, PERSONALTXT
from DialogRPT.src.dialogRPT import getIntegrated

from flask import Flask, request
app = Flask(__name__)


def response_sherlock(inputs: str, params, name = "NoName", history = ""):
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
			output, history = response_sherlock(inputs, params, name, history)
			db[kakaoid] = (name, history)
	return toResponse(output)
    
    
# for ironman
def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits
def sample_sequence(personality, history, tokenizer, model, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(20):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device="cuda").unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device="cuda").unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids).logits  # modified
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / 0.7
        logits = top_filtering(logits, top_k=0, top_p=0.9)
        probs = F.softmax(logits, dim=-1)
        ## no greedy decoding, do sampling
        prev = torch.multinomial(probs, 1)
        if i < 1 and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)
        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    return current_output


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
