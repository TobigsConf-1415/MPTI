import os
import pickle
import datetime
import threading
from flask import jsonify

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