# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import pickle
import random
import traceback

import tensorflow as tf
from keras.optimizers import Adam

from models import *

random.seed(42)
import configs as configs
import codecs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_session(gpu_fraction=0.1):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
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

    def load_pickle(self, filename):
        return pickle.load(open(filename, 'rb'))

        ##### Data Set #####

    def load_use_data(self):
        logger.info('Loading use data..')
        logger.info('methname')
        methodname = pickle.load(open(self.path + self.data_params['valid_methname'], 'rb'))
        logger.info('apiseq')
        apiseqs = pickle.load(open(self.path + self.data_params['valid_apiseq'], 'rb'))
        logger.info('tokens')
        tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        logger.debug('sbt')
        sbt = pickle.load(open(self.path + self.data_params['valid_sbt'], 'rb'))
        return methodname, apiseqs, tokens, sbt

    def load_codebase(self):
        """load codebase
        codefile: h5 file that stores raw code
        """
        logger.info('Loading codebase ...')
        if self._code_base is None:
            codebase = []
            # codes=codecs.open(self.path+self.data_params['use_codebase']).readlines()
            codes = codecs.open(self.path + self.data_params['use_codebase'], encoding='utf8',
                                errors='replace').readlines()
            # use codecs to read in case of encoding problem
            for i in range(0, len(codes)):
                codebase.append(codes[i])
            self._code_base = codebase


    ##### Converting / reverting #####
    def convert(self, vocab, words):
        """convert words into indices"""
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]

    def revert(self, vocab, indices):
        """revert indices into words"""
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    # ##### Model Loading / saving #####
    # def save_model_epoch(self, model, epoch):
    #     if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
    #         os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
    #     model.save("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
    #                "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch),
    #                overwrite=True)
    #
    # def load_model_epoch(self, model, epoch):
    #     assert os.path.exists(
    #         "{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch)) \
    #         , "Weights at epoch {:d} not found".format(epoch)
    #     assert os.path.exists(
    #         "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch)) \
    #         , "Weights at epoch {:d} not found".format(epoch)
    #     model.load("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
    #                "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch))


    def search(self, model, query, n_results=10):
        methnames, apiseqs, tokens, sbt = self.load_use_data()
        print('loading complete')
        padded_methnames = self.pad(methnames, self.data_params['methname_len'])
        padded_apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
        padded_tokens = self.pad(tokens, self.data_params['tokens_len'])
        padded_sbt = self.pad(sbt, self.data_params['sbt_len'])

        data_len = len(tokens)
        desc = self.convert(self.vocab_desc, query)  # convert desc sentence to word indices
        padded_desc = self.pad([desc] * data_len, self.data_params['desc_len'])
        sims = model.predict([padded_methnames, padded_apiseqs, padded_tokens, padded_sbt, padded_desc],
                             batch_size=1000).flatten()  # 是否需要加batchsize ?
        # desc_repr=model.repr_desc([padded_methnames,padded_apiseqs,padded_tokens,padded_desc])
        # desc_repr=desc_repr.astype('float32')

        codes_out = []
        sims_out = []
        negsims = np.negative(sims)
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        codes_out = [self._code_base[k] for k in maxinds]
        sims_out = sims[maxinds]

        return codes_out, sims_out

    def postproc(self, codes_sims):
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


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--proto", choices=["get_config"], default="get_config",
                        help="Prototype config to use for config")
    parser.add_argument("--mode", choices=["train", "eval", "repr_code", "search"], default='search',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set "
                             " The `repr_code/repr_desc` mode computes vectors"
                             " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


def call_module(query, n_results):
    K.set_session(get_session(0.1))  # using 80% of total GPU Memory
    args = parse_args()
    conf = getattr(configs, args.proto)()
    codesearcher = CodeSearcher(conf)

    ##### Define model ######
    logger.info('Build Model')
    model = eval(conf['model_params']['model_name'])(conf)  # initialize the model
    model.build()
    # optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    optimizer = Adam(clipnorm=0.1)
    model.compile(optimizer=optimizer)

    args.mode = 'search'

    # codesearcher.load_code_reprs()
    codesearcher.load_codebase()  # 把raw code 存至一个list里面


    try:
        query = input('Input Query: ')
        n_results = int(input('How many results? '))
    except Exception:
        print("Exception while parsing your input:")
        traceback.print_exc()
        exit(1)


    codes, sims = codesearcher.search(model, query, n_results)
    zipped = zip(codes, sims)
    zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
    zipped = codesearcher.postproc(zipped)
    K.clear_session()

    tl = list(zipped)
    return tl



if __name__ == '__main__':
    call_module()
