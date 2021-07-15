import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
import pandas as pd
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        input_ids = torch.tensor(instance["input_ids"], device="cpu").unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device="cpu").unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids)
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


def run():
    model_checkpoint = 'default_gpt2/'

    model = 'gpt2'
    max_history = 2
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint)
    model.to("cpu")
    add_special_tokens_(model, tokenizer)
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
    personality = [tokenizer.encode(text) for text in personality_text]



    rule_based_df = pd.read_json('rule_based_df.json')
    # question_embed = rule_based_df['question'].apply(tokenizer.encode)
    # question_embed = pd.Series(question_embed.map(lambda x: x + [0] * (20-len(x))))  # padding
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(rule_based_df['question']).toarray()


    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        
        
        # input_vec = tokenizer.encode(input_vec)
        # input_vec = input_vec + [0] * (20-len(input_vec))
        # similarity = question_embed.map(lambda x: dot(x, input_vec)/(norm(x)*norm(input_vec)))  # cosine similarity
        # similarity = question_embed.map(lambda x: len(set(x).intersection(set(input_vec)))/len(set(x).union(set(input_vec))))  # jaccard similarity
        input_vec = vectorizer.transform([raw_text])
        similarity = cosine_similarity(input_vec, tfidf)[0]  # tfidf + cosine similarity

        if max(similarity) > 0.5 :
            print(rule_based_df['answer'][argmax(similarity)])
        else :
        
            history = [tokenizer.encode(raw_text)]
            with torch.no_grad():
                out_ids = sample_sequence(personality, history, tokenizer, model)
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(out_text)


if __name__ == "__main__":
    run()
