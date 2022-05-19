# -*- coding: utf-8 -*-

# 消除所有警告
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import random

import tensorflow as tf
from keras.optimizers import Adam

from utils import cos_np_for_normalized, normalize
from models import *

random.seed(42)
import configs as configs
import codecs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def get_session(gpu_fraction=0.1):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class CodeSearcher:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self.path = self.conf.get('workdir', '../data/github/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())

        self.vocab_methname = self.load_pickle(self.path + self.data_params['vocab_methname'])
        self.vocab_apiseq = self.load_pickle(self.path + self.data_params['vocab_apiseq'])
        self.vocab_sbt = self.load_pickle(self.path + self.data_params['vocab_sbt'])
        self.vocab_tokens = self.load_pickle(self.path + self.data_params['vocab_tokens'])
        self.vocab_desc = self.load_pickle(self.path + self.data_params['vocab_desc'])

        self._eval_sets = None

        self._code_reprs = None
        self._code_base = None
        self._code_base_chunksize = 2000000

    @staticmethod
    def load_pickle(filename):
        return pickle.load(open(filename, 'rb'))

        ##### Data Set #####

    def load_use_data(self):
        methodname = pickle.load(open(self.path + self.data_params['valid_methname'], 'rb'))
        apiseqs = pickle.load(open(self.path + self.data_params['valid_apiseq'], 'rb'))
        tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        sbt = pickle.load(open(self.path + self.data_params['valid_sbt'], 'rb'))
        return methodname, apiseqs, tokens, sbt

    def load_codebase(self):
        if self._code_base is None:
            codebase = []
            codes = codecs.open(self.path + self.data_params['use_codebase'], encoding='utf8',
                                errors='replace').readlines()
            for i in range(0, len(codes)):
                codebase.append(codes[i])
            self._code_base = codebase

    @staticmethod
    def convert(vocab, words):
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]

    @staticmethod
    def revert(vocab, indices):
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    @staticmethod
    def pad(data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        chunk_codes = [self._code_base[i][k] for k in maxinds]
        chunk_sims = chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

    @staticmethod
    def postproc(codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes = [code for code in codes_]
        sims = [sim for sim in sims_]
        final_codes = []
        final_sims = []
        n = len(codes_sims)
        for i in range(n):
            is_dup = False
            for j in range(i):
                if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
                    is_dup = True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes, final_sims)

    def search(self, model, query, n_results=10):
        methnames, apiseqs, tokens, sbt = self.load_use_data()
        # print('loading complete')
        padded_methnames = self.pad(methnames, self.data_params['methname_len'])
        padded_apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
        padded_tokens = self.pad(tokens, self.data_params['tokens_len'])
        padded_sbt = self.pad(sbt, self.data_params['sbt_len'])

        data_len = len(tokens)
        desc = self.convert(self.vocab_desc, query)  # convert desc sentence to word indices
        padded_desc = self.pad([desc] * data_len, self.data_params['desc_len'])
        sims = model.predict([padded_methnames, padded_apiseqs, padded_tokens, padded_sbt, padded_desc],
                             batch_size=1000).flatten()  # 是否需要加batchsize ?

        negsims = np.negative(sims)
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        codes_out = [self._code_base[k] for k in maxinds]
        sims_out = sims[maxinds]

        return codes_out, sims_out

    def load_model_epoch(self, model, epoch):
        model.load("{}models/{}/epo{:d}_sim.h5".format(self.path, self.model_params['model_name'], epoch))


def call_module(query, n_results):
    K.set_session(get_session(0.1))
    conf = getattr(configs, "get_config")()
    codesearcher = CodeSearcher(conf)

    model = eval(conf['model_params']['model_name'])(conf)
    model.build()
    optimizer = Adam(clipnorm=0.1)
    model.compile(optimizer=optimizer)
    codesearcher.load_model_epoch(model, conf['training_params']['reload'])
    codesearcher.load_codebase()
    codes, sims = codesearcher.search(model, query, n_results)
    zipped = zip(codes, sims)
    zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
    zipped = codesearcher.postproc(zipped)
    K.clear_session()

    print("******")
    tl = list(zipped)
    for i in range(len(tl)):
        print(tl[i][0], end='')
        print(tl[i][1])

    # return tl


# if __name__ == '__main__':
#     call_module("convert int to string", 5)

import sys

if __name__ == '__main__':
    # path = "D:\\ChenGe\\Codefiles\\srtp\\Hu_TabCS\\model_code"
    # command = "setx WORK1 %s /m" % path
    # os.system(command)
    # os.environ['WORKON_HOME'] = "D:\\ChenGe\\Codefiles\\srtp\\Hu_TabCS\\model_code"
    # res = os.path.dirname(os.path.dirname(__file__))
    # sys.path.append(res)
    # sys.path.append("D:\\ChenGe\\Codefiles\\srtp\\Hu_TabCS\\model_code")

    os.chdir("D:\\ChenGe\\Codefiles\\srtp\\Hu_TabCS\\model_code")
    query = " ".join(sys.argv[1:-1])
    num = int(sys.argv[-1])
    # print(query, num)
    call_module(query, num)
