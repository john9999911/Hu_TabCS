# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import math
import pickle
import random
import traceback
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scipy.stats import rankdata

from models import *

random.seed(42)
import tables
import configs as configs
import codecs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from model_code.utils import normalize, cos_np_for_normalized

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

    def load_training_data_chunk(self):
        logger.debug('Loading a chunk of training data..')
        logger.debug('methname')
        chunk_methnames = pickle.load(open(self.path + self.data_params['train_methname'], 'rb'))
        logger.debug('apiseq')
        chunk_apiseqs = pickle.load(open(self.path + self.data_params['train_apiseq'], 'rb'))
        logger.debug('tokens')
        chunk_tokens = pickle.load(open(self.path + self.data_params['train_tokens'], 'rb'))
        logger.debug('sbt')
        chunk_sbt = pickle.load(open(self.path + self.data_params['train_sbt'], 'rb'))
        logger.debug('desc')
        chunk_descs = pickle.load(open(self.path + self.data_params['train_desc'], 'rb'))
        return chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_sbt, chunk_descs

    def load_valid_data_chunk(self):
        logger.debug('Loading a chunk of validation data..')
        logger.debug('methname')
        chunk_methnames = pickle.load(open(self.path + self.data_params['valid_methname'], 'rb'))
        logger.debug('apiseq')
        chunk_apiseqs = pickle.load(open(self.path + self.data_params['valid_apiseq'], 'rb'))
        logger.debug('tokens')
        chunk_tokens = pickle.load(open(self.path + self.data_params['valid_tokens'], 'rb'))
        logger.debug('sbt')
        chunk_sbt = pickle.load(open(self.path + self.data_params['valid_sbt'], 'rb'))
        logger.debug('desc')
        chunk_descs = pickle.load(open(self.path + self.data_params['valid_desc'], 'rb'))
        return chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_sbt, chunk_descs

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
        if self._code_base == None:
            codebase = []
            # codes=codecs.open(self.path+self.data_params['use_codebase']).readlines()
            codes = codecs.open(self.path + self.data_params['use_codebase'], encoding='utf8',
                                errors='replace').readlines()
            # use codecs to read in case of encoding problem
            for i in range(0, len(codes)):
                codebase.append(codes[i])
            self._code_base = codebase

    ### Results Data ###
    def load_code_reprs(self):
        logger.debug('Loading code vectors (chunk size={})..'.format(self._code_base_chunksize))
        if self._code_reprs == None:
            """reads vectors (2D numpy array) from a hdf5 file"""
            codereprs = []
            h5f = tables.open_file(self.path + self.data_params['use_codevecs'])
            vecs = h5f.root.vecs
            for i in range(0, len(vecs), self._code_base_chunksize):
                codereprs.append(vecs[i:i + self._code_base_chunksize])
            h5f.close()
            self._code_reprs = codereprs
        return self._code_reprs

    def save_code_reprs(self, vecs):
        npvecs = np.array(vecs)
        fvec = tables.open_file(self.path + self.data_params['use_codevecs'], 'w')
        atom = tables.Atom.from_dtype(npvecs.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
        ds[:] = npvecs
        fvec.close()

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

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch),
                   overwrite=True)

    def load_model_epoch(self, model, epoch):
        assert os.path.exists(
            "{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        assert os.path.exists(
            "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch))

    # ##### Training #####
    # def train(self, model):
    #     if self.train_params['reload'] > 0:
    #         self.load_model_epoch(model, self.train_params['reload'])
    #     valid_every = self.train_params.get('valid_every', None)
    #     save_every = self.train_params.get('save_every', None)
    #     batch_size = self.train_params.get('batch_size', 128)
    #     nb_epoch = self.train_params.get('nb_epoch', 50)
    #     split = self.train_params.get('validation_split', 0)
    #
    #     val_loss = {'loss': 1., 'epoch': 0}
    #     f1 = open('/data/shuaijianhang/TSACS-TASF/model_code/results/training_results.txt', 'a', encoding='utf-8',
    #               errors='ignore')
    #     for i in range(self.train_params['reload'] + 1, nb_epoch):
    #         print('Epoch %d :: \n' % i, end='')
    #         logger.debug('loading data chunk..')
    #         chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_sbt, chunk_descs = self.load_training_data_chunk()
    #         logger.debug('padding data..')
    #         chunk_padded_methnames = self.pad(chunk_methnames, self.data_params['methname_len'])
    #         chunk_padded_apiseqs = self.pad(chunk_apiseqs, self.data_params['apiseq_len'])
    #         chunk_padded_tokens = self.pad(chunk_tokens, self.data_params['tokens_len'])
    #         chunk_padded_sbt = self.pad(chunk_sbt, self.data_params['sbt_len'])
    #         chunk_padded_good_descs = self.pad(chunk_descs, self.data_params['desc_len'])
    #         chunk_bad_descs = [desc for desc in chunk_descs]
    #         random.shuffle(chunk_bad_descs)
    #         chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.data_params['desc_len'])
    #         early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min')
    #         hist = model.fit([chunk_padded_methnames, chunk_padded_apiseqs, chunk_padded_tokens, chunk_padded_sbt,
    #                           chunk_padded_good_descs, chunk_padded_bad_descs], epochs=1, batch_size=batch_size,
    #                          validation_split=split, callbacks=[early_stopping])
    #         if hist.history['val_loss'][0] < val_loss['loss']:
    #             val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
    #         print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))
    #
    #         if valid_every is not None and i % valid_every == 0:
    #             acc1, mrr = self.valid(model, 1)
    #             print(str(acc1))
    #             print(str(mrr))
    #             f1.write('epoch={},ACC1={}, MRR={}'.format(i, acc1, mrr) + '\n')
    #             # acc,mrr,map,ndcg=self.eval(model, 1000, 1)
    #
    #         if save_every is not None and i % save_every == 0:
    #             self.save_model_epoch(model, i)

    # def valid(self, model, K):
    #     """
    #     quick validation in a code pool.
    #     param:
    #         poolsize - size of the code pool, if -1, load the whole test set
    #     """
    #     # load test dataset
    #     if self._eval_sets is None:
    #         # self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
    #         methnames, apiseqs, tokens, sbt, descs = self.load_valid_data_chunk()
    #         self._eval_sets = dict()
    #         self._eval_sets['methnames'] = methnames
    #         self._eval_sets['apiseqs'] = apiseqs
    #         self._eval_sets['tokens'] = tokens
    #         self._eval_sets['sbt'] = sbt
    #         self._eval_sets['descs'] = descs
    #
    #     c_1, c_2 = 0, 0
    #     data_len = len(self._eval_sets['descs'])
    #     for i in range(data_len):
    #         bad_descs = [desc for desc in self._eval_sets['descs']]
    #         random.shuffle(bad_descs)
    #         descs = bad_descs
    #         descs[0] = self._eval_sets['descs'][i]  # good desc
    #         descs = self.pad(descs, self.data_params['desc_len'])
    #         methnames = self.pad([self._eval_sets['methnames'][i]] * data_len, self.data_params['methname_len'])
    #         apiseqs = self.pad([self._eval_sets['apiseqs'][i]] * data_len, self.data_params['apiseq_len'])
    #         tokens = self.pad([self._eval_sets['tokens'][i]] * data_len, self.data_params['tokens_len'])
    #         sbt = self.pad([self._eval_sets['sbt'][i]] * data_len, self.data_params['sbt_len'])
    #         n_good = K
    #
    #         sims = model.predict([methnames, apiseqs, tokens, sbt, descs], batch_size=data_len).flatten()
    #         r = rankdata(sims, method='max')
    #         max_r = np.argmax(r)
    #         max_n = np.argmax(r[:n_good])
    #         c_1 += 1 if max_r == max_n else 0
    #         c_2 += 1 / float(r[max_r] - r[max_n] + 1)
    #
    #     top1 = c_1 / float(data_len)
    #     # percentage of predicted most similar desc that is really the corresponding desc
    #     mrr = c_2 / float(data_len)
    #     print('Precision={}, MRR={}'.format(top1, mrr))
    #
    #     return top1, mrr
    #
    #     ##### Evaluation in the develop set #####

    # def eval(self, model, K):
    #     """
    #     validate in a code pool.
    #     param:
    #         poolsize - size of the code pool, if -1, load the whole test set
    #     """
    #
    #     def SUCCRATE(real, predict, n_results):
    #         sum = 0.0
    #         for val in real:
    #             try:
    #                 index = predict.index(val)
    #             except ValueError:
    #                 index = -1
    #             if index <= n_results: sum = sum + 1
    #         return sum / float(len(real))
    #
    #     def ACC(real, predict):
    #         sum = 0.0
    #         for val in real:
    #             try:
    #                 index = predict.index(val)
    #             except ValueError:
    #                 index = -1
    #             if index != -1: sum = sum + 1
    #         return sum / float(len(real))
    #
    #     def MAP(real, predict):
    #         sum = 0.0
    #         for id, val in enumerate(real):
    #             try:
    #                 index = predict.index(val)
    #             except ValueError:
    #                 index = -1
    #             if index != -1: sum = sum + (id + 1) / float(index + 1)
    #         return sum / float(len(real))
    #
    #     def MRR(real, predict):
    #         sum = 0.0
    #         for val in real:
    #             try:
    #                 index = predict.index(val)
    #             except ValueError:
    #                 index = -1
    #             if index != -1: sum = sum + 1.0 / float(index + 1)
    #         return sum / float(len(real))
    #
    #     def NDCG(real, predict):
    #         dcg = 0.0
    #         idcg = IDCG(len(real))
    #         for i, predictItem in enumerate(predict):
    #             if predictItem in real:
    #                 itemRelevance = 1
    #                 rank = i + 1
    #                 dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    #         return dcg / float(idcg)
    #
    #     def IDCG(n):
    #         idcg = 0
    #         itemRelevance = 1
    #         for i in range(n):
    #             idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    #         return idcg
    #
    #     # load valid dataset
    #     if self._eval_sets is None:
    #         methnames, apiseqs, tokens, sbt, descs = self.load_valid_data_chunk()
    #         self._eval_sets = dict()
    #         self._eval_sets['methnames'] = methnames
    #         self._eval_sets['apiseqs'] = apiseqs
    #         self._eval_sets['tokens'] = tokens
    #         self._eval_sets['sbt'] = sbt
    #         self._eval_sets['descs'] = descs
    #     data_len = len(self._eval_sets['descs'])
    #     numbers = codecs.open(
    #         '/data/shuaijianhang/Vocab3Hybrid-CARLCS_Hiera_Attention/model_code/results/bootstrap_nums.txt', 'a',
    #         errors='ignore', encoding='utf-8')
    #     for k in range(0, 10):
    #         succrate, acc, mrr, map, ndcg = 0, 0, 0, 0, 0
    #         print(str(k * 10000) + " to : " + str((k + 1) * 10000))
    #         bootstrap_num_list = np.random.choice(10000, 10000)
    #         numbers.write(str(bootstrap_num_list) + '\n')
    #         print('number of samples: ' + str(len(bootstrap_num_list)))
    #         for number in bootstrap_num_list:
    #             num = int(number)
    #             print(num)
    #             print('*****')
    #             desc = self._eval_sets['descs'][num]  # good desc
    #             descs = self.pad([desc] * data_len, self.data_params['desc_len'])
    #             methnames = self.pad(self._eval_sets['methnames'], self.data_params['methname_len'])
    #             apiseqs = self.pad(self._eval_sets['apiseqs'], self.data_params['apiseq_len'])
    #             tokens = self.pad(self._eval_sets['tokens'], self.data_params['tokens_len'])
    #             sbt = self.pad(self._eval_sets['sbt'], self.data_params['sbt_len'])
    #             n_results = K
    #             sims = model.predict([methnames, apiseqs, tokens, sbt, descs], batch_size=1000).flatten()
    #             negsims = np.negative(sims)
    #             predict_origin = np.argsort(negsims)  # predict = np.argpartition(negsims, kth=n_results-1)
    #             predict = predict_origin[:n_results]
    #             predict = [int(k) for k in predict]
    #             predict_origin = [int(k) for k in predict_origin]
    #             real = [num]
    #             succrate += SUCCRATE(real, predict_origin, n_results)
    #             acc += ACC(real, predict)
    #             mrr += MRR(real, predict)
    #             map += MAP(real, predict)
    #             ndcg += NDCG(real, predict)
    #         succrate = succrate / float(data_len)
    #         acc = acc / float(data_len)
    #         mrr = mrr / float(data_len)
    #         map = map / float(data_len)
    #         ndcg = ndcg / float(data_len)
    #         print('SuccRate={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(succrate, acc, mrr, map, ndcg))
    #         f2 = codecs.open(
    #             '/data/shuaijianhang/Vocab3Hybrid-CARLCS_Hiera_Attention/model_code/results/bootsrtap_results.txt', 'a',
    #             encoding='utf-8', errors='ignore')
    #         f2.write('SuccRate={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(succrate, acc, mrr, map, ndcg) + '\n')

    ##### Compute Representation #####
    def repr_code(self, model):
        methnames, apiseqs, tokens, others = self.load_use_data()
        methnames = self.pad(methnames, self.data_params['methname_len'])
        apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
        tokens = self.pad(tokens, self.data_params['tokens_len'])

        vecs = model.repr_code([methnames, apiseqs, tokens],
                               batch_size=1000)  # model.predict()方法可以设置batchsize来分批次分配tensor
        vecs = vecs.astype('float32')
        vecs = normalize(vecs)
        self.save_code_reprs(vecs)
        return vecs

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

    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
        # 1. compute similarity
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        # 2. choose top results
        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        chunk_codes = [self._code_base[i][k] for k in maxinds]
        chunk_sims = chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

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



def call_module():
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

    if args.mode == 'train':
        codesearcher.train(model)

    elif args.mode == 'eval':
        # evaluate for a particular epoch
        # load model
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.eval(model, 10)

    elif args.mode == 'repr_code':
        # load model
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        vecs = codesearcher.repr_code(model)

    elif args.mode == 'search':
        # search code based on a desc
        if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
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


        tl = list(zipped)
        return tl

        # for i in range(len(tl)):
        #     print('\ncode:\n', tl[i][0], '\nsim:\n', tl[i][1])


        # zipped = list(zipped)[:n_results]
        # results = '\n\n'.join(map(str, zipped))  # combine the result into a returning string
        # print(results)

    K.clear_session()
