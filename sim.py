#!/usr/bin/python3.5

import os
import re
import sys

import librosa
import librosa.display
import imagehash
import bitstring
import scipy

from PIL import Image
from dtw import dtw
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from prettytable import PrettyTable


def get_audio_sim_score(file1, file2):
    max_dist_value = 1000

    y1, sr1 = librosa.load(file1)
    y2, sr2 = librosa.load(file2)

    mfcc1 = librosa.feature.mfcc(y1, sr1)  # Computing MFCC values
    mfcc2 = librosa.feature.mfcc(y2, sr2)
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

    return 1 - dist / max_dist_value


def get_images_sim_score(file1, file2):
    h1 = imagehash.phash(Image.open(file1))
    h2 = imagehash.phash(Image.open(file2))
    hash_bit_size = 64

    return 1 - (h1 - h2) / hash_bit_size


def get_docs_sim_score(filename1, filename2):
    with open(filename1) as file1, open(filename2) as file2:
        text1 = file1.read()
        text2 = file2.read()

        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform([text1, text2])

        return (tfidf * tfidf.T).A[1][0]


def get_bin_sim_score(filename1, filename2):
    with open(filename1, mode='rb') as file1, open(filename2, mode='rb') as file2:
        arr1 = bitstring.BitArray(file1)
        arr2 = bitstring.BitArray(file2)

        len1, len2 = len(arr1), len(arr2)
        if len1 > len2:
            arr2 = [False] * (len1 - len2) + arr2
        elif len2 > len1:
            arr1 = [False] * (len2 - len1) + arr1

        return 1 - scipy.spatial.distance.hamming(arr1, arr2)


def get_abs_filenames(root):
    names = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            names.append(path + '/' + name)

    return names


def get_filenames_by_pattern(fnames, pattern):
    matching_files = [n for n in fnames if re.fullmatch(pattern, n)]
    fnames -= set(matching_files)

    return matching_files


def get_ext_handle_dict(filenames):
    patn = lambda exts: '|'.join(['.*\.' + ext for ext in exts])

    ext_dict = {
        'text': (['odt', 'txt', 'doc', 'rtf'], get_docs_sim_score),
        'image': (['tiff', 'jpeg', 'gif', 'png', 'jpg'], get_images_sim_score),
        'audio': (['wav', 'flac', 'm4a', 'mp3'], get_audio_sim_score)
    }

    fnames = set(filenames)

    for ftype in ext_dict:
        exts, sim_func = ext_dict[ftype]
        ftype_files = get_filenames_by_pattern(fnames, patn(exts))
        ext_dict[ftype] = ftype_files, sim_func,

    ext_dict['others'] = list(fnames), get_bin_sim_score

    return ext_dict


def get_sim_list(ext_dict, ftype, sim_score, eps=0.05):
    filenames, sim_func = ext_dict[ftype]
    bp = lambda path: os.path.basename(path)
    sim_list = []

    for f1, f2 in combinations(filenames, 2):
        score = sim_func(f1, f2)
        if (score + eps) >= sim_score:
            sim_values = bp(f1), bp(f2), "%.2f" % score
            sim_list.append(sim_values)

    return sim_list


DEFAULT_SIM_SCORE = 0.7
def output_sim_tables(ext_dict, sim_score):
    for ftype in ext_dict:
        res = get_sim_list(ext_dict, ftype, sim_score)
        table = PrettyTable()
        table.field_names = ["file1", "file2", "sim_score"]
        for f1, f2, sc in res:
            table.add_row([f1, f2, sc])

        print(ftype, table, '\n', sep='\n')


if __name__ == '__main__':
    args = sys.argv
    args_len = len(args)

    if args_len == 2:
        sim_score = DEFAULT_SIM_SCORE
        root = args[1]
    elif args_len == 3:
        sim_score = float(args[1][1:])
        root = args[2]
    else:
        raise ValueError('wrong arg. amount')

    handling_files = get_abs_filenames(root)
    ext_dict = get_ext_handle_dict(handling_files)

    output_sim_tables(ext_dict, sim_score)