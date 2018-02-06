#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import sys
import subprocess
import tempfile
import itertools

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "??", "!", "!!", ",", ":", "-", "--", "...", ";", "-lrb-", "-rrb-",
        "-lcb-", "-rcb-", "-LSB-", "-RSB-", "-lsb-", "-rsb-"]

class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def tokenize_caption(self, caption):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(caption.encode())
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE)
        token_line = p_tokenizer.communicate(input=caption.rstrip())[0]
        token_line = token_line.decode()
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create tokenized caption
        # ======================================================
        tokenized_caption = [w for w in token_line.rstrip().split(' ') \
                    if w not in PUNCTUATIONS]
        return tokenized_caption

    def tokenize(self, anns):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_captions_for_image = {}
        #cid = [item['id'] for item in captions_for_image]
        #sentences = [item['caption'].replace('\n', ' ') for item in captions_for_image]
        #sentences = '\n'.join(sentences)
        ids = anns.keys()
        sentences = [anns[i]['caption'].replace('\n', ' ') for i in ids]
        sentences = '\n'.join(sentences)


        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        assert len(list(ids)) == len(lines)
        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for i, line in zip(ids, lines):
            """
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            """
            assert i not in final_tokenized_captions_for_image
            tokenized_caption = [w for w in line.rstrip().split(' ') \
                    if w not in PUNCTUATIONS]
            final_tokenized_captions_for_image[i] = tokenized_caption

            #final_tokenized_captions_for_image += "{} {}\n".format(k, tokenized_caption)
        return final_tokenized_captions_for_image
