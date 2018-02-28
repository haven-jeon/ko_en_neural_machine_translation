import numpy as np
import pandas as pd
from mxnet import nd as  F
import mxnet.autograd as autograd
import mxnet as mx
from  mxnet import gluon
from mxnet.gluon import nn, rnn
from konlpy.tag import Mecab
mecab = Mecab()


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    "from keras pad_seqnences()"
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def encoding_and_padding(corp_list, dic, max_seq=30):
    coding_seq = [ [dic.get(j, dic['__ETC__']) for j in i]  for i in corp_list ]
    return(pad_sequences(coding_seq, maxlen=max_seq, padding='post', truncating='post',value=dic['__PAD__']))



class korean_english_translator(gluon.HybridBlock):
    
    def __init__(self, n_hidden, vocab_size, embed_dim, max_seq_length, end_idx, attention=False, **kwargs):
        super(korean_english_translator,self).__init__(**kwargs)
        self.end_idx = end_idx
        #입력 시퀀스 길이
        self.in_seq_len = max_seq_length
        #출력 시퀀스 길이 
        self.out_seq_len = max_seq_length
        # GRU의 hidden 개수 
        self.n_hidden = n_hidden
        #고유문자개수 
        self.vocab_size = vocab_size
        #max_seq_length
        self.max_seq_length = max_seq_length
        #임베딩 차원수 
        self.embed_dim = embed_dim
        
        self.attention = attention
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=vocab_size, output_dim=embed_dim, dtype="float16")
            
            self.encoder= rnn.GRUCell(hidden_size=n_hidden)
            self.decoder = rnn.GRUCell(hidden_size=n_hidden)
            self.batchnorm = nn.BatchNorm(axis=2)
            #flatten을 false로 할 경우 마지막 차원에 fully connected가 적용된다. 
            self.dense = nn.Dense(self.vocab_size,flatten=False)
            if self.attention:
                self.dropout = nn.Dropout(0.3)
                self.attdense = nn.Dense(self.max_seq_length, flatten=False)
                self.attn_combine = nn.Dense( self.n_hidden, flatten=False)
            
    def hybrid_forward(self, F, inputs, outputs, initial_hidden_state, batch_size_seq):
        #문장 길이 2 == END tag index
        inputs = F.cast(inputs, dtype='float32')
        in_sent_last_idx = F.argmax(F.where(inputs == self.end_idx, F.ones_like(inputs), F.zeros_like(inputs)), axis=1)
        
        outputs = F.cast(outputs, dtype='float32')
        out_sent_last_idx = F.argmax(F.where(outputs == self.end_idx, F.ones_like(outputs), F.zeros_like(outputs)), axis=1)
        #encoder GRU
        embeddinged_in = F.cast(self.embedding(inputs), dtype='float32')
        
        next_h = initial_hidden_state
        for j in range(self.in_seq_len):
            p_outputs = F.slice_axis(embeddinged_in, axis=1, begin=j, end=j+1)
            p_outputs = F.reshape(p_outputs, (-1, self.embed_dim))
            enout, (next_h,) = self.encoder(p_outputs, [next_h,] )
            if j == 0:
                enouts = enout
                next_hs = next_h
            else:
                enouts = F.concat(enouts, enout, dim=1)
                next_hs = F.concat(next_hs, next_h, dim=1)
        #masking with 0 using length
        enouts = F.reshape(enouts, (-1, self.in_seq_len, self.n_hidden))
        enouts = F.transpose(enouts, (1,0,2))
        enouts = F.SequenceMask(enouts, sequence_length=in_sent_last_idx + 1, use_sequence_length=True)
        enouts = F.transpose(enouts, (1,0,2))
        
        next_hs = F.reshape(next_hs, (-1, self.n_hidden))
        #take가 0 dim만 지원하기 때문에.. 
        # N, 30, 300 -> N * 30, 300 , N = (0,1,2,3,4,5...)
        next_hs = next_hs.take(in_sent_last_idx  +  (batch_size_seq * self.max_seq_length))
        embeddinged_out = F.cast(self.embedding(outputs),dtype='float32')
        
        #decoder GRU with attention
        for i in range(self.out_seq_len):
            #out_seq_len 길이만큼 GRUCell을 unroll하면서 출력값을 적재한다. 
            p_outputs = F.slice_axis(embeddinged_out, axis=1, begin=i, end=i+1)
            p_outputs = F.reshape(p_outputs, (-1, self.embed_dim))
            # p_outputs = outputs[:,i,:]
            # 위와 같이 진행한 이유는 hybridize를 위함 
            if self.attention:
                p_outputs, _ = self.apply_attention(F=F, inputs=p_outputs, hidden=next_hs, encoder_outputs=enouts)
            deout, (next_hs,) = self.decoder(p_outputs, [next_hs,] )
            if i == 0:
                deouts = deout
            else:
                deouts = F.concat(deouts, deout, dim=1)
        #2dim -> 3dim 으로 reshape 
        deouts = F.reshape(deouts, (-1, self.out_seq_len, self.n_hidden))
        #0 padding 
        deouts = F.transpose(deouts, (1,0,2))
        deouts = F.SequenceMask(deouts, sequence_length=out_sent_last_idx + 1, use_sequence_length=True)
        deouts = F.transpose(deouts, (1,0,2))
        
        
        deouts = self.batchnorm(deouts)
        deouts_fc = self.dense(deouts)
        return(deouts_fc)
    
    def apply_attention(self, F, inputs, hidden, encoder_outputs):
        #inputs : decoder input의미
        concated = F.concat(inputs, hidden, dim=1)
        #(,max_seq_length) : max_seq_length 개별 시퀀스의 중요도  
        attn_weights = F.softmax(self.attdense(concated), axis=1)
        # (N,max_seq_length,n_hidden) x (N,max_seq_length) = (N, max_seq_length, n_hidden)
        #attn_weigths 가중치를 인코더 출력값에 곱해줌
        w_encoder_outputs = F.broadcast_mul(encoder_outputs, attn_weights.expand_dims(2))
        #(N, vocab_size * max_seq_length), (N, max_seq_length * n_hidden) = (N, ...)
        output = F.concat(inputs.flatten(), w_encoder_outputs.flatten(), dim=1)
        output = self.dropout(output)
        #(N, vocab_size)
        output = self.attn_combine(output)
        #attention weight은 시각화를 위해 뽑아둔다. 
        return(output, attn_weights)
    
    def calulation(self, input_str, ko_dict, en_dict, en_rev_dict, ctx):
        """
        inference 코드 
        """
        #앞뒤에 START,END 코드 추가 
        input_str = [['START', ] + mecab.morphs(input_str.strip()) + ['END', ],]
        X = encoding_and_padding(input_str, ko_dict, max_seq=self.max_seq_length)
        #string to embed 
        inputs = F.array(X, ctx=ctx)
        
        inputs = F.cast(inputs, dtype='float32')
        in_sent_last_idx = F.argmax(F.where(inputs == self.end_idx, F.ones_like(inputs), F.zeros_like(inputs)), axis=1)
        
        #encoder GRU
        embeddinged_in = F.cast(self.embedding(inputs), dtype='float32')
        next_h =  F.random.normal(0,1,(1, self.n_hidden), ctx=ctx)
        for j in range(self.in_seq_len):
            p_outputs = F.slice_axis(embeddinged_in, axis=1, begin=j, end=j+1)
            p_outputs = F.reshape(p_outputs, (-1, self.embed_dim))
            enout, (next_h,) = self.encoder(p_outputs, [next_h,] )
            if j == 0:
                enouts = enout
                next_hs = next_h
            else:
                enouts = F.concat(enouts, enout, dim=1)
                next_hs = F.concat(next_hs, next_h, dim=1)
        #masking with 0 using length
        enouts = F.reshape(enouts, (-1, self.in_seq_len, self.n_hidden))
        enouts = F.transpose(enouts, (1,0,2))
        enouts = F.SequenceMask(enouts, sequence_length=in_sent_last_idx + 1, use_sequence_length=True)
        enouts = F.transpose(enouts, (1,0,2))
        
        next_hs = F.reshape(next_hs, (-1, self.n_hidden))
        #take가 0 dim만 지원하기 때문에.. 
        # N, 30, 300 -> N * 30, 300 , N = (0,1,2,3,4,5...)
        next_hs = next_hs.take(in_sent_last_idx)
        
        #디코더의 초기 입력값으로 넣을 'START'를 임베딩한다.
        Y_init = F.array([[en_dict['START'],],], ctx=ctx)
        Y_init = F.cast(self.embedding(Y_init),dtype='float32')
        deout = Y_init[:,0,:]
        
        #출력 시퀀스 길이만큼 순회 
        for i in range(self.out_seq_len):
            if self.attention:
                #print(deout.shape)
                deout, att_weight = self.apply_attention(F=F, inputs=deout, hidden=next_hs, encoder_outputs=enouts)
                if i == 0:
                    att_weights = att_weight
                else:
                    att_weights = F.concat(att_weights,att_weight,dim=0)
            deout, (next_hs, ) = self.decoder(deout, [next_hs, ])
            #batchnorm을 적용하기 위해 차원 증가/원복 
            deout = F.expand_dims(deout,axis=1)
            deout = self.batchnorm(deout)
            #reduce dim
            deout = deout[:,0,:]
            #'START'의 다음 시퀀스 출력값도출 
            deout_sm = self.dense(deout)
            #print(deout_sm.shape)
            deout = F.one_hot(F.argmax(F.softmax(deout_sm, axis=1), axis=1), depth=self.vocab_size)
            #print(deout.shape)
            #decoder에 들어갈 수 있는 형태로 변환(임베딩 적용 및 차원 맞춤)
            deout = F.argmax(deout, axis=1)
            deout = F.expand_dims(deout, axis=0)
            deout = F.cast(self.embedding(deout)[:,0,:],dtype='float32')
            gen_char  = en_rev_dict[F.argmax(deout_sm, axis=1).asnumpy()[0].astype('int')]
            if gen_char == '__PAD__' or gen_char == 'END':
                break
            else: 
                if i == 0:
                    ret_seq = [gen_char, ]
                else:
                    ret_seq += [gen_char, ]
        return(" ".join(ret_seq), att_weights)







