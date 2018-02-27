#Code by Lisa Anne Hendricks for "Generating Visual Explanations"
#
#@article{hendricks2016generating,
#  title={Generating Visual Explanations},
#  author={Hendricks, Lisa Anne and Akata, Zeynep and Rohrbach, Marcus and Donahue, Jeff and Schiele, Bernt and Darrell, Trevor},
#  journal={arXiv preprint arXiv:1603.08507},
#  year={2016}
#}

import json
import sys
import re

UNK_IDENTIFIER = '<unk>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)
  print("Wrote json file to %s" % save_name)

def open_txt(t_file):
  txt_file = open(t_file).readlines()
  return [t.strip() for t in txt_file]

def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  if sentence[-1] != '.':
    return sentence
  return sentence[:-1]

def tokenize_text(sentence, vocabulary, leave_out_unks=False):
 sentence = [s.strip() for s in split_sentence(sentence)]
 token_sent = []
 for w in sentence:
   try:
     token_sent.append(vocabulary[w])
   except:
     if not leave_out_unks:
       try:
         token_sent.append(vocabulary['<unk>'])
       except:
         pass
     else:
       pass
 if not leave_out_unks:
   token_sent.append(vocabulary['EOS'])
 return token_sent
