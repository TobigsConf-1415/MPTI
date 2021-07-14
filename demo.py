import os
import sys
import pickle
import threading
import time
import datetime
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from train_ironman import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import pandas as pd
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from DialogRPT.src.dialogRPT import getIntegrated
sherlockModel = getIntegrated()
EOS = "<|endoftext|>"

def load(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def get_from_db(uid):
    if uid not in db:
        return (None, None)
    return db[uid]

def response_sherlock(inputs: str, name = "Jinkyoung", history = ""):
    params = {'topk':3, 'beam': 3, 'topp': 0.8, 'max_t':15}
    history += inputs+ EOS
    if (history.count(EOS) > 4):
        history = history.split(EOS, maxsplit = 2)[-1]
    final, prob_gen, score_ranker, hyp = sherlockModel.predict(history, 0.4, params)[0]
    sherlock = hyp.replace("<NAME>", name)
    history += sherlock + EOS
    return sherlock, history
    
    
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



tokenizer_ironman = GPT2Tokenizer.from_pretrained('default_gpt2')
model_ironman = GPT2LMHeadModel.from_pretrained('default_gpt2')
model_ironman.to("cuda")
add_special_tokens_(model_ironman, tokenizer_ironman)
personality_text = [
        'my name is tony stark .',
        'i am iron man .',
        'i am a billionaire industrialist .',
        'i am a superhero .',
        'i have a daughter named morgan .',
        'i had saved the world countless times .',
        'i like American cheeseburger .',
        'i am married with pepper potts .',
        'i killed thanos .', 
        'i am a genius .', 
        'i own the stark industries .', 
        'i programmed JARVIS .',
        'i put on my armored suit to protect the world as Iron Man .',
        'i was born on may 29, 1970 .' , 
        'i am a genius inventor .',
        'i was born in manhattan, new york .', 
        'my father is howard stark .', 
        'i am the founding member of the Avengers .'
        'i love to be the center of attention .' 
    ]
personality = [tokenizer_ironman.encode(text) for text in personality_text]
rule_based_df = pd.read_json('rule_based_df.json')
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(rule_based_df['question']).toarray()

def response_ironman(inputs: str):
    
    input_vec = vectorizer.transform([inputs])
    similarity = cosine_similarity(input_vec, tfidf)[0]
    
    if max(similarity) > 0.5 :
        return rule_based_df['answer'][argmax(similarity)]
    else:
        history = [tokenizer_ironman.encode(inputs)]
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer_ironman, model_ironman)
        
        capital = {'i': 'I', "i'll": "I'll", "i'm": "I'm", "i'd": "I'd", "i've":"I've",
           'tony': 'Tony',
           'morgan': 'Morgan',
           'pepper': 'Pepper',
           'potts': 'Potts',
           'ironman': 'Ironman', 'iron': 'Iron',
           'stark': 'Stark',
           'avengers': 'Avengers', 
           'thanos':'Thanos',
           'jarvis': 'JARVIS',
           'manhattan':'Manhattan',
           'new':'New','york':'York',
           'california':'California',
           'usa':'USA',
           'howard':'Howard',
           'may':'May',
           }
        response = tokenizer_ironman.decode(out_ids, skip_special_tokens=True)
        response = re.split("([.?!’”]) ", response)
        response = [response[i-1] + response[i] for i in range(1, len(response), 2)] + [response[-1]]
        response = re.split("([.?!’,”]) ", ' '.join([r.capitalize() for r in response]))
        response_fin = []
        for word in [w for s in response for w in s.split()]:
            punc = ''
            if word.endswith(('.', '!','?')):
                punc = word[-1]; word = word[:-1]
            if word in capital.keys():
                response_fin.append(capital[word])
            else: response_fin.append(word)
            response_fin.append(punc)
            
        return re.sub(r' (?=\W)', '', ' '.join(response_fin))



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
	return "snubob api server"

@app.route("/hello", methods=['POST'])
def hello():
	msg = "TEST_hello."
	req = request.get_json()
	res = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": msg               }
            }
        ]
    }
}
	return jsonify(res)

@app.route("/sherlock", methods=["POST"])
def sherlock():
	msg = "This is Sherlock Holmes."
	req = request.get_json()
	inputs = req["userRequest"]["utterance"]
	kakaoid = req["userRequest"]["user"]["id"]
	if inputs.startswith("!register"):
		name = inputs.split(" ")[-1]
		db[kakaoid] = (name, "")
		output = "Welcome to MPTI-Sherlock Chatbot, {}🙌 Enjoy your chatting with the great Sherlock Holmes. \n\n❗️Please Do not enter your personal information as recent conversation records are saved in the server.❗️".format(name)
	else:
		name, history = get_from_db(kakaoid)
		if name==None:
			output = "Please register your name first. 🧐\nYou can regiser your name with command🙏: !register [YOUR NAME HERE]"
		else:
			output, history = response_sherlock(inputs, name, history)
# 			print(history)
			db[kakaoid] = (name, history)
	res2 = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": output
                }
            }
        ]
    }
}
	return  jsonify(res2)

@app.route("/ironman", methods=["POST"])
def ironman():
	msg = "I am Iron Man."
	req = request.get_json()
	inputs = req["userRequest"]["utterance"]
# 	kakaoid = req["userRequest"]["user"]["id"]
# 	if not get_from_db(kakaoid)[0]:
# 		name = input("Welcome to MPTI-Ironman Chatbot🙌 Enjoy your chatting with Ironman. \n\n❗️Please register your name first. Enter your name.")
# 		history = ""
# 		db_ironman[kakaoid] = (name, history)
# 	else:
# 		history = get_from_db(kakaoid)[1]
# 		history = json.loads(history)
# 		history.append(tokenizer.encode(inputs))
# 	output = response_ironman(str(history))
	output = response_ironman(inputs)
	res = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": output
                }
            }
        ]
    }
}
	return  jsonify(res)

# @app.route("/save", methods=["POST", "GET"])
def save(filename):
    now_time = '({})'.format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'))
    output = now_time + " !!!!!!Something is Wrong!!!!!!"
    with open(filename, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output = now_time + " Successfully Saved "+filename
    print(output)
    threading.Timer(600, save, args = [filename]).start()


if __name__ == "__main__":
	filename = "db.pickle"
	db = load(filename)
	device = torch.device("cuda")
	save(filename)
	app.run(host='0.0.0.0', port=3305)
