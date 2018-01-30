from mxnet.gluon.loss import Loss,_apply_weighting, _reshape_like


class SoftmaxCrossEntropyLossMask(Loss):
    def __init__(self, end_idx, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLossMask, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.end_idx = end_idx

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        #각 label 문장의 마지막 문자('END') 인덱스 정보 추출 
        label = F.cast(label, dtype='float32')
        label_sent_length = F.argmax(F.where(label == self.end_idx, F.ones_like(label), F.zeros_like(label)), axis=1)
        
        
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        #(N, 30, val)
        #길이를 초과하는 영역에 대해서 0로 loss 마스킹을 수행함 
        loss = F.transpose(loss, (1,0, 2))
        loss = F.SequenceMask(loss, sequence_length=label_sent_length + 1, use_sequence_length=True)
        loss = F.transpose(loss, (1,0, 2))
        return F.sum(loss, axis=self._batch_axis, exclude=True)/(label_sent_length + 1)

    
    