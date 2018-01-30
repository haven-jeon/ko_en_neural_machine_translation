import argparse
import unicodedata
import string
import re
from konlpy.tag import Mecab
from stemming.porter2 import stem
from itertools import zip_longest, chain
from tqdm import tqdm
import time
import numpy as np
import os
from  mxnet import gluon
import mxnet as mx
from mxnet import nd as  F
import mxnet.autograd as autograd

#os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'

from  embedding_maker import *
from model import *
from mask_loss import SoftmaxCrossEntropyLossMask


mecab = Mecab()

SOS_token = "START"
EOS_token = "END"
embed_dim = 50




parser = argparse.ArgumentParser(description='Gluon Korean-English Translater')
parser.add_argument('--num-iters', type=int, default=5,
                    help='number of iterations to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--max-length', type=int, default=30,
                    help='max length of sentence (default: 30)')
parser.add_argument('--batch-size', type=int, default=60,
                    help='train batch size (default: 60)')
parser.add_argument('--hidden-size', type=int, default=384,
                    help='number of hidden units in encoder and decoder(default: 384)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='train on GPU with CUDA')
parser.add_argument('--test', action='store_true', default=False,
                    help='test layer by layer')
parser.add_argument('--train-sort', action='store_true', default=False,
                    help='need training data sort by length')
parser.add_argument('--embedding', action='store_true', default=False,
                    help='make embedding')
parser.add_argument('--train', action='store_true', default=False,
                    help='make embedding')
parser.add_argument('--gpu-count', type=int, default=1,
                    help='number of gpu (default: 1)')
parser.add_argument('--model-prefix', type=str, default="ko_en_mdl",
                    help='prefix of *.param file')
parser.add_argument('--init-model', type=str, default="",
                    help='model file to train start from')

opt = parser.parse_args()





train_corpus =('korean_parallel_corpora/korean-english-v1/korean-english-park.train.ko',
               'korean_parallel_corpora/korean-english-v1/korean-english-park.train.en') 

test_corpus =('korean_parallel_corpora/korean-english-v1/korean-english-park.test.ko', 
              'korean_parallel_corpora/korean-english-v1/korean-english-park.test.en')





class preprocessing:
    
    def __init__(self):
        punct = '"“”#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'
        self.table = str.maketrans({key: None for key in punct})
    
    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    @staticmethod
    def normalizeString(s):
        s = preprocessing.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def extract_stem(self, fname, l= 'ko'):
        spaced = []
        with open(fname) as f:
            if l == 'ko':
                spaced = [['START', ] + mecab.morphs(unicodedata.normalize('NFKC', i.strip()).translate(self.table)) + 
                          ['END', ] for i in f.readlines()]
            elif l == 'en':
                spaced = [['START', ] + [stem(j) for j in mecab.morphs(self.normalizeString(i)) ] + 
                          ['END', ] for i in f.readlines()]
            else:
                assert(False)
        return(spaced)

    def prepare_embedding(self, ko, en, filenm="embedding_train.txt"):
        ko_en = [ list(filter(None.__ne__, list(chain.from_iterable(list(zip_longest(i, j[1:-1]))))))  
                    for i, j in zip(ko, en) ]
        with open(filenm, "wt") as f:
            f.writelines([" ".join(i) + '\n' for i in ko_en])
        return(filenm)
    
    def train_embedding(self, data_files, model_file, embeddings_file, vocab_file, min_count, iter, size, workers, window, splitc=' '):
        create_embeddings(data_files, model_file=model_file, embeddings_file=embeddings_file, vocab_file=vocab_file, splitc=' ',
                          min_count=min_count, iter=iter, size=size, workers=workers, window=window)

#병렬 코퍼스이기 때문에 두 문장 모두 최장 길이 이하를 만족하는 학습셋만 취한다. 
def corpus_length_filter(ko, en, max_len=opt.max_length):
    tr_idx_ko = [idx  for idx, i in enumerate(ko) if len(i) <= max_len]
    tr_idx_en = [idx  for idx, i in enumerate(en) if len(i) <= max_len]

    tr = np.intersect1d(tr_idx_ko, tr_idx_en)
    tr = set(tr)
    ko_f = [i  for idx, i in enumerate(ko) if idx in tr] 
    en_f = [i  for idx, i in enumerate(en) if idx in tr] 
    return(ko_f, en_f)

def get_sorted_index(train_set):
    train_len= [len(i) for i in train_set]
    tr_sort_idx = np.array(train_len).argsort()
    return(tr_sort_idx)




def model_init(n_hidden,vocab_size, embed_dim, max_seq_length, embed_weights, ctx, end_idx, attention=True):
    #모형 인스턴스 생성 및 트래이너, loss 정의 
    #n_hidden, vocab_size, embed_dim, max_seq_length
    model = korean_english_translator(n_hidden, vocab_size, embed_dim, max_seq_length, end_idx, attention=True)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    model.embedding.weight.set_data(embed_weights)
         
    trainer = gluon.Trainer(model.collect_params(), 'rmsprop')
    loss = SoftmaxCrossEntropyLossMask(end_idx, axis = 2)
    return(model, loss, trainer)



def calculate_loss(model, data_iter, loss_obj, ctx):
    test_loss = []
    for i, (x_data, y_data, z_data) in enumerate(data_iter):
        x_data_l = gluon.utils.split_and_load(x_data, ctx,  even_split=False)
        y_data_l = gluon.utils.split_and_load(y_data, ctx,  even_split=False)
        z_data_l = gluon.utils.split_and_load(z_data, ctx,  even_split=False)
        with autograd.predict_mode():
            losses = [loss_obj(model(x, y, F.random.normal(0,1,(x.shape[0], n_hidden), ctx=x.context),F.arange(x.shape[0], ctx=x.context)), z) for x, y, z in zip(x_data_l, y_data_l, z_data_l)]
        curr_loss = np.mean([mx.nd.mean(i).asscalar() for i in losses])
        test_loss.append(curr_loss)
    return(np.mean(test_loss))



def train(epochs, tr_data_iterator, model, loss, trainer, ctx, start_epoch=1, mdl_desc="k2e_model", decay=False):
    ### 학습 코드 
    tot_test_loss = []
    tot_train_loss = []
    epochs = epochs + start_epoch
    for e in range(start_epoch, epochs):
        tic = time.time()
        # Decay learning rate.
        if e > 1 and decay:
            trainer.set_learning_rate(trainer.learning_rate * 0.7)
        train_loss = []
        batches = tqdm(tr_data_iterator, 'Batches')
        for i, (x_data, y_data, z_data) in enumerate(batches):
            x_data_l = gluon.utils.split_and_load(x_data, ctx, even_split=False)
            y_data_l = gluon.utils.split_and_load(y_data, ctx, even_split=False)
            z_data_l = gluon.utils.split_and_load(z_data, ctx, even_split=False)
            with autograd.record():
                losses = [loss(model(x, y, F.random.normal(0,1,(x.shape[0], n_hidden), ctx=x.context),F.arange(x.shape[0], ctx=x.context) ), 
                               z) for x, y, z in zip(x_data_l, y_data_l, z_data_l)]
            for l in losses:
                l.backward()
            trainer.step(x_data.shape[0])
            curr_loss = np.mean([mx.nd.mean(l).asscalar() for l in losses])
            train_loss.append(curr_loss)
            batches.set_description("loss {}".format(curr_loss))
            mx.nd.waitall()
        #caculate test loss
        test_loss = calculate_loss(model, te_data_iterator, loss_obj = loss, ctx=ctx) 
        print('[Epoch %d] time cost: %f'%(e, time.time()-tic))
        print("Epoch %s. Train Loss: %s, Test Loss : %s" % (e, np.mean(train_loss), test_loss))    
        tot_test_loss.append(test_loss)
        tot_train_loss.append(np.mean(train_loss))
        model.save_params("{}_{}.params".format(mdl_desc, e))
    return(tot_test_loss, tot_train_loss)






if opt.embedding == True:
    """임베딩 학습"""
    print("doing embedding training!")
    en_ko_pre = preprocessing()
    
    train_ko = en_ko_pre.extract_stem(train_corpus[0],l='ko')
    train_en = en_ko_pre.extract_stem(train_corpus[1],l='en')
    test_ko  = en_ko_pre.extract_stem(test_corpus[0] ,l='ko')
    test_en  = en_ko_pre.extract_stem(test_corpus[1] ,l='en')
    
    fn = en_ko_pre.prepare_embedding(train_ko + test_ko, train_en + test_en)
    en_ko_pre.train_embedding([fn,], "ko_en.mdl", 'ko_en.np', 'ko_en.dic', 
                              min_count=10, 
                              iter=50, 
                              size=50, 
                              workers=10, 
                              window=30)
    print("done embedding training!")


if opt.train:
    print("training : prepare data")
    
    en_ko_pre = preprocessing()
    
    #텍스트 tokenization 전처리 
    train_ko = en_ko_pre.extract_stem(train_corpus[0],l='ko')
    train_en = en_ko_pre.extract_stem(train_corpus[1],l='en')
    test_ko  = en_ko_pre.extract_stem(test_corpus[0] ,l='ko')
    test_en  = en_ko_pre.extract_stem(test_corpus[1] ,l='en')
    
    #max length 이하 필터
    train_ko_f, train_en_f = corpus_length_filter(train_ko, train_en, opt.max_length)
    test_ko_f, test_en_f = corpus_length_filter(test_ko, test_en, opt.max_length)
     
    #학습셋을 정렬해서 넣을 경우 
    tr_sort_idx = get_sorted_index(train_ko_f)
    
    #학습된 임베딩 사전 로딩     
    w2idx, idx2w = load_vocab("ko_en.dic") 
    
    #디코더 출력값 loss 계산을 위한 1 lag로 구성된 학습, 테스트셋 
    train_en_lag = [ i[1:] for i in train_en_f]
    test_en_lag = [ i[1:] for i in test_en_f]
    
    #encoding and padding 
    ko_train_x = encoding_and_padding(train_ko_f, w2idx, max_seq=opt.max_length)
    ko_test_x = encoding_and_padding(test_ko_f, w2idx, max_seq=opt.max_length)
    en_train_x = encoding_and_padding(train_en_f, w2idx, max_seq=opt.max_length)
    en_test_x = encoding_and_padding(test_en_f, w2idx, max_seq=opt.max_length)
    en_train_y = encoding_and_padding(train_en_lag, w2idx, max_seq=opt.max_length)
    en_test_y = encoding_and_padding(test_en_lag, w2idx, max_seq=opt.max_length)
    
    #Hyper parameters
    max_seq_length = opt.max_length 
    vocab_size = len(w2idx)
    n_hidden = opt.hidden_size
    embed_dim = embed_dim
    #embedding network에 넣을 행렬   
    embed_weights  = load_embedding("ko_en.np")
    #문장의 마지막을 인식할 인덱스 
    end_idx = w2idx['END']
    
    #학습을 위한 데이터 제너레이터 객체 생성 
    tr_set = gluon.data.ArrayDataset(ko_train_x[tr_sort_idx,], en_train_x[tr_sort_idx,], en_train_y[tr_sort_idx,])
    tr_data_iterator = gluon.data.DataLoader(tr_set, batch_size=opt.batch_size, shuffle=not opt.train_sort)
    te_set =gluon.data.ArrayDataset(ko_test_x, en_test_x, en_test_y)
    te_data_iterator = gluon.data.DataLoader(te_set, batch_size=30, shuffle=True)
    
    GPU_COUNT = opt.gpu_count
    ctx= [mx.gpu(i) for i in range(GPU_COUNT)]
    
    print("training : prepare model")
    #초기 모형 생성 및 loss정의 
    model, loss, trainer = model_init(n_hidden, vocab_size, embed_dim, max_seq_length, embed_weights, ctx, end_idx, attention=True)
    model.hybridize()
    if opt.init_model == '':
        print('train from null')
        tr_loss, te_loss = train(7, tr_data_iterator, model, loss, trainer, ctx=ctx, mdl_desc=opt.model_prefix, decay=False)
        trainer_sgd = gluon.Trainer(model.collect_params(), 'sgd', optimizer_params={'learning_rate':0.01, 'wd':1e-5})
        tr_loss, te_loss = train(3, tr_data_iterator, model, loss, trainer_sgd, start_epoch=8, ctx=ctx, mdl_desc=opt.model_prefix, decay=False)
    else:
        print("train start from '{}'".format(opt.init_model))
        model.load_params(opt.init_model, ctx=ctx)
        trainer_sgd = gluon.Trainer(model.collect_params(), 'sgd', optimizer_params={'learning_rate':0.1,}, kvstore='local')
        tr_loss, te_loss = train(5, tr_data_iterator, model, loss, trainer_sgd, ctx=ctx, mdl_desc=opt.model_prefix, decay=True)

if opt.test:
    """
    inference
    """
    embed_weights  = load_embedding("ko_en.np")
    vocab_size = embed_weights.shape[0]
    embed_dim = embed_weights.shape[1]
    max_seq_length = opt.max_length 
    ctx = mx.cpu(0)
    w2idx, idx2w = load_vocab("ko_en.dic") 
    end_idx = w2idx['END']
    
    model = korean_english_translator(opt.hidden_size, vocab_size, embed_dim, max_seq_length, end_idx, attention=True)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    
    model.load_params(opt.init_model, ctx=ctx)
    while 1:
        kor_sent = input("kor > ")
        print(kor_sent)
        eng_seq, _  = model.calulation(kor_sent, ko_dict=w2idx, en_dict=w2idx, en_rev_dict=idx2w, ctx=ctx)
        print("eng > {}".format(eng_seq))

    
        