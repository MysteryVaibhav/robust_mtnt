# -*- coding: utf-8 -*-
# usage: 
# python artificial_noise.py "test.fr" "test.en" "test-noisy.fr" "test-noisy.en"  "0.04,0.007,0.002,0.015" <----(Spl = 0.04, Prof = 0.007, Emo = 0.002, grammar = 0.015)
import emoji
import sys
# from stop_words import get_stop_words
import codecs
import numpy as np


SWEAR_FILE = "./swear_words.txt"
FR_EN_STOPS = "./fr_en_stops.txt"


def add_err_spl(fr_word, en_word):
    char_pos = np.random.choice(np.arange(0, len(fr_word)))
    chars = list(fr_word)
    print("chosen character {}".format(chars[char_pos]))
    if np.random.random() < 0.5:
        # drop character
        del chars[char_pos]
        fr_word = "".join(chars)
        # print("fr_word after drop:{}".format(fr_word))
    else:
        # insert twice repeated characters
        repeat_char = "".join([chars[char_pos]] * 2)
        chars.insert(char_pos, repeat_char)
        fr_word = "".join(chars)
    print("word after spell fr:{}, en:{}".format(fr_word, en_word))
    return fr_word, en_word


def add_err_prof(fr_word, en_word):
    swear_pairs = []
    # there aren't many swears in the file, so read into a list
    with codecs.open(SWEAR_FILE, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            swear_pairs.append(line)
    # swear_choice = np.random.choice(np.arange(0, len(swear_pairs)))

    choice = np.random.choice(swear_pairs)
    # print(swear_pairs)
    # print(choice)
    fr_swear, en_swear = choice.split("\t")
    fr_word += " " + fr_swear.strip()
    en_word += " " + en_swear.strip()

    return fr_word, en_word


def add_err_emo(fr_word, en_word):
    emoji_list = list(emoji.UNICODE_EMOJI.keys())
    emoji_choice = "".join([np.random.choice(emoji_list)] * 3)  # inserting emoticon repeated thrice
    fr_word += " " + emoji_choice
    en_word += " " + emoji_choice

    return fr_word, en_word


def add_err_gram(fr_word, en_word):
    fr_en_stop_pairs = []
    # there aren't many swears in the file, so read into a list
    with codecs.open(FR_EN_STOPS,'r',encoding='utf-8') as f:
        for line in f.readlines():
            fr_en_stop_pairs.append(line)
            # swear_choice = np.random.choice(np.arange(0, len(swear_pairs)))
    choice = np.random.choice(fr_en_stop_pairs)
    # print(fr_en_stop_pairs)
    # print(choice)
    fr_stop, en_stop = choice.split("\t")
    fr_word += " " + fr_stop.strip()
    en_word += " " + en_stop.strip()
    return fr_word, en_word

def add_noise(input_file_src, input_file_tgt, output_file_src, output_file_tgt, noise_proba, noise_funcs):
    with codecs.open(input_file_src, 'r', encoding='utf-8') as f_src_in, \
            codecs.open(input_file_tgt, 'r', encoding='utf-8') as f_tgt_in, \
            codecs.open(output_file_src, 'w', encoding='utf-8') as f_src_out, \
            codecs.open(output_file_tgt, 'w', encoding='utf-8') as f_tgt_out:
        for fr_line, en_line in zip(f_src_in.readlines(), f_tgt_in.readlines()):
            noisy_fr = []
            noisy_en = []
            # print("fr:{} , en:{}".format(fr_line, en_line))
            word_posn = 0
            for fr_word, en_word in zip(fr_line.split(), en_line.split()):
                noise_id = np.random.choice(np.arange(0, len(noise_proba)), p=noise_proba)
                if not noise_funcs[noise_id] is None:
                    fr_word, en_word = noise_funcs[noise_id](fr_word, en_word)
                    # if fr_word is None:
                    #     print(noise_id)
                noisy_fr.append(fr_word.strip())
                noisy_en.append(en_word.strip())
                word_posn+=1
            # print("fr: {} en:{}".format(noisy_fr,noisy_en))
            if len(fr_line) > len(en_line):
                # add remaining words in fr
                fr_remaining = fr_line.split()[word_posn:len(fr_line)-1]
                noisy_fr += fr_remaining
            else:
                en_remaining = en_line.split()[word_posn:len(en_line) - 1]
                noisy_en += en_remaining
            f_src_out.write(" ".join(noisy_fr) + "\n")
            f_tgt_out.write(" ".join(noisy_en) + "\n")

if __name__ == "__main__":
    input_file_src = sys.argv[1]
    input_file_tgt = sys.argv[2]
    output_file_src = sys.argv[3]
    output_file_tgt = sys.argv[4]
    # Noise probability in the order Spl = 0.04, Prof = 0.007, Emo = 0.002, grammar = 0.015
    noise_proba = [float(x) for x in sys.argv[5].split(',')]
    pure_proba = 1 - sum(noise_proba)
    noise_proba.append(pure_proba)
    noise_funcs = [add_err_spl, add_err_prof, add_err_emo, add_err_gram, None]
    add_noise(input_file_src, input_file_tgt, output_file_src, output_file_tgt, noise_proba, noise_funcs)
