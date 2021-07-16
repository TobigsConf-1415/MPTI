import os
import pickle
import datetime
import threading
from flask import jsonify
from TransferTransfo.train_ironman import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_

EOS = "<|endoftext|>"
WELCOME = "Welcome to MPTI-Sherlock Chatbot, {}🙌 Enjoy your chatting with the great Sherlock Holmes. \n\n❗️Please Do not enter your personal information as recent conversation records are saved in the server.❗️"
REGISTER = "Please register your name first. 🧐\nYou can regiser your name with command🙏: !register [YOUR NAME HERE]"
CAPITAL =  {'i': 'I', "i'll": "I'll", "i'm": "I'm", "i'd": "I'd", "i've":"I've",
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
PERSONALTXT= [
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

def load(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    
def nameFromDB(db, uid):
    if uid not in db:
        return (None, None)
    return db[uid]

def save(db, filename):
    now_time = '({})'.format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'))
    output = now_time + " !!!!!!Something is Wrong!!!!!!"
    with open(filename, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output = now_time + " Successfully Saved "+filename
    print(output)
    threading.Timer(600, save, args = [filename]).start()

def toResponse(output):
    res = {
    "version": "2.0",
    "template": {
        "outputs": [
                        {
                            "simpleText": {"text": output}
                        }
                    ]
                }
            }
    return jsonify(res)


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