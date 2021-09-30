# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# ================ CAN config ================
logger.info("=============== CAN config ===============")
weight_emb_w = [[16, 8], [8,4]] 
weight_emb_b = [0, 0]
logger.info("weight_emb_w:{}, weight_emb_b:{}".format(weight_emb_w, weight_emb_b))
orders = 3
order_indep = False # True
WEIGHT_EMB_DIM = (sum([w[0]*w[1] for w in weight_emb_w]) + sum(weight_emb_b)) #* orders
INDEP_NUM = 1
if order_indep:
    INDEP_NUM *= orders

logger.info("orders:{}".format(orders))
logger.info("WEIGHT_EMB_DIM:{}".format(WEIGHT_EMB_DIM))

CALC_MODE = "can"
logger.info("=============== CAN config ===============")
# ================ CAN config ================


class CANLayer(nn.Layer):
    def __init__(self, n_uid, n_mid, n_cate, n_carte,seq_len,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,dnn_layer_size,
                 use_negsampling = False, use_softmax=True, use_coaction=True, use_cartes=True):
        super(CANLayer, self).__init__()
        self.use_coaction = use_coaction
        self.use_cartes = use_cartes
        self.use_negsampling = use_negsampling
        self.use_dice = False

        logger.info("use_dice:{}".format(self.use_dice))
        logger.info("use_cartes:{}".format(self.use_cartes))
        logger.info("use_coaction:{}".format(self.use_coaction))
        logger.info("use_negsampling:{}".format(self.use_negsampling))

        self.fcn_net = FCN_Net(n_uid, n_mid, n_cate, n_carte,seq_len,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,dnn_layer_size,
                                use_negsampling=self.use_negsampling, use_softmax=True, use_coaction=self.use_coaction, use_cartes=self.use_cartes, use_dice=self.use_dice)

    def forward(self, inputs):

        if self.use_negsampling:
            outputs, aux_loss = self.fcn_net(inputs)
            return outputs, aux_loss
        else:
            outputs = self.fcn_net(inputs)
            return outputs


class Gen_Coaction(nn.Layer):
    def __init__(self, dim, mode="can",keep_fake_carte_seq=None):
        super(Gen_Coaction, self).__init__()
        self.dim = dim
        self.mode = mode
        # self.mask = mask
        self.keep_fake_carte_seq = keep_fake_carte_seq
        self.orders = orders
        

    def forward(self, ad, his_items,mask=None):
        weight, bias = [], []  # order_indep = True 时，产生bug
        idx = 0
        weight_orders = []
        bias_orders = []
        for i in range(self.orders):
            # weight, bias = [], []  # 修改上面
            # idx = 0  # 修改上面
            for w, b in zip(weight_emb_w, weight_emb_b):
                # print(w[0]*w[1])
                weight.append(paddle.reshape(ad[:, idx:idx+w[0]*w[1]], [-1, w[0], w[1]]))
                idx += w[0] * w[1]
                if b == 0:
                    bias.append(None)
                else:
                    bias.append(paddle.reshape(ad[:, idx:idx+b], [-1, 1, b]))
                    idx += b
            weight_orders.append(weight)
            bias_orders.append(bias)
            if not order_indep:
                break
    
        if self.mode == "can":
            out_seq = []
            hh = []
            for i in range(self.orders):
                hh.append(his_items**(i+1))
            #hh = [sum(hh)]
            for i, h in enumerate(hh):
                if order_indep:
                    weight, bias = weight_orders[i], bias_orders[i]
                else:
                    weight, bias = weight_orders[0], bias_orders[0]
                for j, (w, b) in enumerate(zip(weight, bias)):
                    h = paddle.matmul(h,w)
                    if b is not None:
                        h = h + b
                    if j != len(weight)-1:
                        h = paddle.tanh(h)
                    out_seq.append(h)
            out_seq = paddle.concat(out_seq, 2)
            if mask is not None:
                mask = paddle.unsqueeze(mask, axis=-1) 
                out_seq = out_seq * mask
        out = paddle.sum(out_seq, 1)
        if self.keep_fake_carte_seq and self.mode=="emb":
            return out, out_seq
        return out, None


class EmbeddingLayer(nn.Layer):
    def __init__(self,n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                 use_negsampling = False, use_softmax=True, use_coaction=True, use_cartes=True):
        super(EmbeddingLayer, self).__init__()

        self.use_negsampling = use_negsampling
        self.use_coaction = use_coaction
        self.use_cartes = use_cartes
        self.n_carte = n_carte
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HIDDEN_SIZE = HIDDEN_SIZE
        

        self.init_value_ = 0.01

        self.gen_coaction = Gen_Coaction(dim=EMBEDDING_DIM, mode="can", keep_fake_carte_seq=None)

        self.uid_batch_embedded = paddle.nn.Embedding(
            n_uid,
            EMBEDDING_DIM,
            sparse=False,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(EMBEDDING_DIM))),
                    # std=self.init_value_),
                # regularizer=paddle.regularizer.L2Decay(1e-6)
                ))
       
       
        self.mid_batch_embedded = paddle.nn.Embedding(
            n_mid,
            EMBEDDING_DIM,
            sparse=False,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(EMBEDDING_DIM))),
                # regularizer=paddle.regularizer.L2Decay(1e-6)
            ))
           
       
        self.mid_his_batch_embedded = paddle.nn.Embedding(
            n_mid,
            EMBEDDING_DIM,
            sparse=False,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(EMBEDDING_DIM))),
                # regularizer=paddle.regularizer.L2Decay(1e-6)
                ))

        if self.use_negsampling:
            self.noclk_mid_his_batch_embedded = paddle.nn.Embedding(
                n_mid,
                EMBEDDING_DIM,
                sparse=False,
                padding_idx=0,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0,
                        std=self.init_value_ /
                        math.sqrt(float(EMBEDDING_DIM))),
                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                    ))

        self.cate_batch_embedded = paddle.nn.Embedding(
                n_cate,
                EMBEDDING_DIM,
                sparse=False,
                padding_idx=0,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0,
                        std=self.init_value_ /
                    math.sqrt(float(EMBEDDING_DIM))),
                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                    ))

        self.cate_his_batch_embedded = paddle.nn.Embedding(
                n_cate,
                EMBEDDING_DIM,
                sparse=False,
                padding_idx=0,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0,
                        std=self.init_value_ /
                    math.sqrt(float(EMBEDDING_DIM))),
                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                    ))

        if self.use_negsampling:
            self.noclk_cate_his_batch_embedded = paddle.nn.Embedding(
                n_cate,
                EMBEDDING_DIM,
                sparse=False,
                padding_idx=0,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0,
                        std=self.init_value_ /
                        math.sqrt(float(EMBEDDING_DIM))),
                    regularizer=paddle.regularizer.L2Decay(1e-6)))

        if self.use_cartes:
            self.carte_batch_embedded = []
            for i, num in enumerate(self.n_carte):
                # print("carte num:", num)
                carte_embedding =  paddle.nn.Embedding(
                                    num,
                                    EMBEDDING_DIM,
                                    sparse=False,
                                    padding_idx=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.TruncatedNormal(
                                            mean=0.0,
                                            std=self.init_value_ /
                                            math.sqrt(float(EMBEDDING_DIM))),
                                    regularizer=paddle.regularizer.L2Decay(1e-6)))
                
                self.carte_batch_embedded.append(carte_embedding)

        ###  co-action ###
        if self.use_coaction:
            self.mlp_batch_embedded = []
            item_mlp_embeddings = paddle.nn.Embedding(
                                    n_mid,
                                    INDEP_NUM * WEIGHT_EMB_DIM,
                                    sparse=False,
                                    padding_idx=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.TruncatedNormal(
                                            mean=0.0,
                                            std=self.init_value_ /
                                                math.sqrt(float(INDEP_NUM * WEIGHT_EMB_DIM))),
                                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                                    ))
            cate_mlp_embeddings = paddle.nn.Embedding(
                                    n_cate,
                                    INDEP_NUM * WEIGHT_EMB_DIM,
                                    sparse=False,
                                    padding_idx=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.TruncatedNormal(
                                            mean=0.0,
                                            std=self.init_value_ /
                                            math.sqrt(float(INDEP_NUM * WEIGHT_EMB_DIM))),
                                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                                    ))
            self.mlp_batch_embedded.append(item_mlp_embeddings)
            self.mlp_batch_embedded.append(cate_mlp_embeddings)

            self.input_batch_embedded = []
            item_input_embeddings = paddle.nn.Embedding(
                                    n_mid,
                                    weight_emb_w[0][0] * INDEP_NUM,
                                    sparse=False,
                                    padding_idx=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.TruncatedNormal(
                                            mean=0.0,
                                            std=self.init_value_ /
                                            math.sqrt(float(weight_emb_w[0][0] * INDEP_NUM))),
                                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                                    ))
            cate_input_embeddings = paddle.nn.Embedding(
                                    n_cate,
                                    weight_emb_w[0][0] * INDEP_NUM,
                                    sparse=False,
                                    padding_idx=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=paddle.nn.initializer.TruncatedNormal(
                                            mean=0.0,
                                            std=self.init_value_ /
                                            math.sqrt(float(weight_emb_w[0][0] * INDEP_NUM))),
                                    # regularizer=paddle.regularizer.L2Decay(1e-6)
                                    ))

            self.input_batch_embedded.append(item_input_embeddings)
            self.input_batch_embedded.append(cate_input_embeddings)
        

    def forward(self, inps):
        uid_batch_ph = inps[0]
        mid_batch_ph = inps[1]
        cate_batch_ph = inps[2]
        mid_his_batch_ph = inps[3]
        cate_his_batch_ph = inps[4]
        mask = inps[5]
        noclk_mid_batch_ph = inps[6]
        noclk_cate_batch_ph = inps[7]
        carte_batch_ph = inps[8]

        # print('mask',mask)

        uid_batch_embedded = self.uid_batch_embedded(uid_batch_ph)

        mid_batch_embedded = self.mid_batch_embedded(mid_batch_ph)
        cate_batch_embedded = self.cate_batch_embedded(cate_batch_ph)
        item_eb = paddle.concat([mid_batch_embedded,cate_batch_embedded],axis=1)

        mid_his_batch_embedded = self.mid_his_batch_embedded(mid_his_batch_ph)
        cate_his_batch_embedded = self.cate_his_batch_embedded(cate_his_batch_ph)
        item_his_eb = paddle.concat([mid_his_batch_embedded,cate_his_batch_embedded],axis=2)
        # print('item_his_eb',item_his_eb.shape) # [512, 100, 36]
        item_his_eb_sum = paddle.sum(item_his_eb,axis=1)

        if self.use_negsampling:
            noclk_mid_his_batch_embedded = self.noclk_mid_his_batch_embedded(noclk_mid_batch_ph)
            noclk_cate_his_batch_embedded = self.noclk_cate_his_batch_embedded(noclk_cate_batch_ph)
            noclk_item_his_emb = paddle.concat([noclk_mid_his_batch_embedded[:, :, 0, :],noclk_cate_his_batch_embedded[:, :, 0, :]],axis=-1)
            noclk_item_his_emb = paddle.reshape(noclk_item_his_emb,[-1,noclk_mid_his_batch_embedded.shape[1],2*self.EMBEDDING_DIM])

            noclk_his_emb = paddle.concat([noclk_mid_his_batch_embedded,noclk_cate_his_batch_embedded],axis=-1)
            # print('noclk_his_emb',noclk_his_emb.shape) # [512, 100, 5, 36]
            noclk_his_emb_sum_1 = paddle.sum(noclk_his_emb,axis=2)
            # print('noclk_his_emb_sum_1',noclk_his_emb_sum_1.shape) # [512, 100, 36]
            noclk_his_emb_sum = paddle.sum(noclk_his_emb_sum_1,axis=1)

        cross = []
    
        # print('use_cartes:',self.use_cartes)
        if self.use_cartes:
            for i,embed in enumerate(self.carte_batch_embedded):
                emb = embed(carte_batch_ph[:,i,:])
                if mask is not None:
                    mask_ = paddle.unsqueeze(mask, axis=-1)
                    emb = emb * mask_
                carte_emb_sum = paddle.sum(emb, axis=1) 
                # print('carte_emb_sum',carte_emb_sum)
                cross.append(carte_emb_sum)
        
        if self.use_coaction:
            mlp_batch_embedded, input_batch_embedded = [],[]

            mlp_batch_embedded.append(self.mlp_batch_embedded[0](mid_batch_ph))
            mlp_batch_embedded.append(self.mlp_batch_embedded[1](cate_batch_ph))

            input_batch_embedded.append(self.input_batch_embedded[0](mid_his_batch_ph))
            input_batch_embedded.append(self.input_batch_embedded[1](cate_his_batch_ph))
                
            tmp_sum, tmp_seq = [], []
            if INDEP_NUM == 2:
                for i, mlp_batch in enumerate(mlp_batch_embedded):
                    for j, input_batch in enumerate(input_batch_embedded):
                        coaction_sum, coaction_seq = self.gen_coaction(mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j+1)], input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i+1)],mask)
                        # coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j+1)], input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i+1)],  EMBEDDING_DIM, mode=CALC_MODE,mask=self.mask) 
                        tmp_sum.append(coaction_sum)
                        tmp_seq.append(coaction_seq)
            else:
                for i, (mlp_batch, input_batch) in enumerate(zip(mlp_batch_embedded, input_batch_embedded)):
                    # print(mlp_batch.shape, input_batch.shape) [2,160] [2,100,16]
                    coaction_sum, coaction_seq = self.gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM], input_batch[:, :, : weight_emb_w[0][0]],mask)
                    # coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM], input_batch[:, :, : weight_emb_w[0][0]],  EMBEDDING_DIM, mode=CALC_MODE, mask=self.mask) 
                    tmp_sum.append(coaction_sum)
                    tmp_seq.append(coaction_seq)
            
            coaction_sum_ = paddle.concat(tmp_sum, axis=1)
            cross.append(coaction_sum_)
        # print('uid_batch_embedded:',uid_batch_embedded)
        if self.use_negsampling:
            return uid_batch_embedded, item_eb, item_his_eb_sum, cross, mask, item_his_eb, noclk_his_emb_sum_1
        else:
            return uid_batch_embedded, item_eb, item_his_eb_sum, cross, mask


class FCN_Net(nn.Layer):
    def __init__(self, n_uid, n_mid, n_cate, n_carte,seq_len,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,dnn_layer_size,
                 use_negsampling = False, use_softmax=True, use_coaction=True, use_cartes=True, use_dice=False):
        super(FCN_Net, self).__init__()

        self.use_dice = use_dice
        self.use_cartes = use_cartes
        self.use_softmax = use_softmax
        self.use_coaction = use_coaction
        self.dnn_layer_size = dnn_layer_size
        self.use_negsampling = use_negsampling

        self.seq_len = seq_len

        unit_list = [200,80]
       
        self.embed = EmbeddingLayer(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                 use_negsampling = use_negsampling, use_softmax=True, use_coaction=self.use_coaction, use_cartes=self.use_cartes)


        self.drop_out = paddle.nn.Dropout(p=0.5)

        # self.act1 =  paddle.nn.PReLU(1, 0.25)
        self.act1 = Prelu()
        
        self.act2 = Dice(axis=-1,epsilon=0.000000001)

        if self.use_coaction and not self.use_cartes:
            self.bnLayer = nn.BatchNorm(EMBEDDING_DIM*9) 
            self.dnn1 = nn.Linear(EMBEDDING_DIM*9,unit_list[0])
            # self.bnLayer = nn.BatchNorm(EMBEDDING_DIM*9) 
            # self.dnn1 = nn.Linear(EMBEDDING_DIM*9,unit_list[0])

        elif self.use_coaction and self.use_cartes:
            self.bnLayer = nn.BatchNorm(EMBEDDING_DIM*11) 
            self.dnn1 = nn.Linear(EMBEDDING_DIM*11,unit_list[0]) 
        else:
            self.bnLayer = nn.BatchNorm(EMBEDDING_DIM*5)
            self.dnn1 = nn.Linear(EMBEDDING_DIM*5,unit_list[0])

        self.bn2 = nn.BatchNorm(unit_list[0])
        self.dnn2 = nn.Linear(unit_list[0],unit_list[1])

        # self.dnn3 = nn.Linear(unit_list[1],2 if self.use_softmax else 1)
        self.bn3 = nn.BatchNorm(unit_list[1])
        self.dnn3 = nn.Linear(unit_list[1],1)

        self.cal_aux_loss = Calcute_aux_loss(seq_len, EMBEDDING_DIM, HIDDEN_SIZE)
       
    def forward(self, inputs):
        """
        inputs: list-> []
        outputs: [batch_size,dim]
        """
        # uid_batch_embedded:[batch_size,EMBEDDING_DIM] 
        # item_eb:[batch_size,EMBEDDING_DIM*2]
        # item_his_eb_sum:[batch_size,EMBEDDING_DIM*2]
        # cross: Embedding List if use_coaction else None
        if self.use_negsampling:
            uid_batch_embedded, item_eb, item_his_eb_sum, cross,mask, item_his_eb, noclk_his_emb = self.embed(inputs)
            dnn_input = paddle.concat([uid_batch_embedded, item_eb, item_his_eb_sum]+cross,axis=1)  # [2,162] [2,138]
            dnn_input = self.bnLayer(dnn_input)
            dnn_out1 = self.dnn1(dnn_input)
            if self.use_dice:
                dnn_out1 = self.act2(dnn_out1)
            else:
                dnn_out1 = self.act1(dnn_out1)
            dnn_out2 = self.dnn2(dnn_out1)
            if self.use_dice:
                dnn_out2 = self.act2(dnn_out2)
            else:
                dnn_out2 = self.act1(dnn_out2)
            dnn_out3 = self.dnn3(dnn_out2)
            outputs = dnn_out3
        
            aux_loss = self.cal_aux_loss(item_his_eb, noclk_his_emb, mask)

            return outputs, aux_loss

        else:
            uid_batch_embedded, item_eb, item_his_eb_sum, cross, mask = self.embed(inputs)
            # print(uid_batch_embedded.shape, item_eb.shape, item_his_eb_sum.shape)
            # print(len(cross))
            # uid_batch_embedded = self.mlp_user(uid_batch_embedded_)
            # item_eb = self.mlp_item(item_eb_)

            dnn_input = paddle.concat([uid_batch_embedded, item_eb, item_his_eb_sum]+cross,axis=1)  # [2,162] [2,138]
            # print(dnn_input.shape)

            dnn_input = self.bnLayer(dnn_input)
            
            dnn_out1 = self.dnn1(dnn_input)
            if self.use_dice:
                dnn_out1 = self.act2(dnn_out1)
            else:
                dnn_out1 = self.act1(dnn_out1)
            # dnn_out1 = self.drop_out(dnn_out1)
            # dnn_out1 = self.bn2(dnn_out1)
            dnn_out2 = self.dnn2(dnn_out1)
            if self.use_dice:
                dnn_out2 = self.act2(dnn_out2)
            else:
                dnn_out2 = self.act1(dnn_out2)
            # dnn_out2 = self.drop_out(dnn_out2)
            # dnn_out2 = self.bn3(dnn_out2)
            dnn_out3 = self.dnn3(dnn_out2)
            
            outputs = dnn_out3

            return outputs


class Auxiliary_loss(paddle.nn.Layer):
    def __init__(self,EMBEDDING_DIM):
        super(Auxiliary_loss, self).__init__()
    
        self.auxiliary_net = Auxiliary_net(EMBEDDING_DIM)

    def forward(self, h_states, click_seq, noclick_seq, mask):
        click_input_ = paddle.concat([h_states, click_seq], axis=-1)
        noclick_input_ = paddle.concat([h_states, noclick_seq], axis=-1)
        # print('click_input_',click_input_.shape,noclick_input_.shape)  # [512,99,72]

        _, max_seq_length, embedding_size = click_seq.shape
        embedding_size = embedding_size*2
        click_input_ = paddle.reshape(click_input_,[-1,embedding_size]) # [batch_size*max_seq_length,embedding_size]
        noclick_input_ = paddle.reshape(noclick_input_,[-1,embedding_size])
        # print('click_input_',click_input_.shape,noclick_input_.shape)

        click_prop_ = self.auxiliary_net(click_input_)  # [batch_size*max_seq_length,1]
        noclick_prop_ = self.auxiliary_net(noclick_input_)
        # print('click_prop_',click_prop_.shape,noclick_prop_.shape)

        # mask_select
        click_prop_ = paddle.reshape(click_prop_,[-1,max_seq_length]) # [batch_size, max_seq_length]
        noclick_prop_ = paddle.reshape(noclick_prop_,[-1,max_seq_length])
        # print('click_prop_',click_prop_.shape,noclick_prop_.shape)

        mask = (mask[:,1:] == 1)
        # print('mask_',mask)

        click_prop_ = paddle.masked_select(click_prop_, mask)
        noclick_prop_ = paddle.masked_select(noclick_prop_, mask)
        click_prop_ = paddle.reshape(click_prop_,[-1,1])
        noclick_prop_ = paddle.reshape(noclick_prop_,[-1,1])
        # print('click_prop_mask',click_prop_.shape)

        click_target = paddle.ones(shape=click_prop_.shape, dtype='float32')
        noclick_target = paddle.zeros(shape=noclick_prop_.shape, dtype='float32')
        # print('click_target',click_target.shape,noclick_target.shape)

        loss = F.binary_cross_entropy(
                paddle.concat([click_prop_, noclick_prop_], axis=0),
                paddle.concat([click_target, noclick_target], axis=0)
                )

        return loss


class Auxiliary_net(paddle.nn.Layer):
    def __init__(self,EMBEDDING_DIM):
        super(Auxiliary_net, self).__init__()
        self.bn1 = nn.BatchNorm(EMBEDDING_DIM*4)
        self.dnn1 = nn.Linear(EMBEDDING_DIM*4,100)
        self.dnn2 = nn.Linear(100,50)
        self.dnn3 = nn.Linear(50,1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        print('inputs',inputs.shape)
        inputs = self.bn1(inputs)
        dnn_out1 = self.dnn1(inputs)
        dnn_out1 = self.act(dnn_out1)
        dnn_out2 = self.dnn2(dnn_out1)
        dnn_out2 = self.act(dnn_out2)
        dnn_out3 = self.dnn3(dnn_out2)
        dnn_out3 = self.act(dnn_out3)
        outputs = dnn_out3
        return outputs


class Calcute_aux_loss(paddle.nn.Layer):
    def __init__(self,seq_len, EMBEDDING_DIM, HIDDEN_SIZE):
        super(Calcute_aux_loss, self).__init__()
        
        self.seq_len = seq_len
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HIDDEN_SIZE = HIDDEN_SIZE
        cell = paddle.nn.GRUCell(EMBEDDING_DIM*2,HIDDEN_SIZE)
        self.rnn  = paddle.nn.RNN(cell)

        self.auxiliary_loss = Auxiliary_loss(EMBEDDING_DIM)

        
    def forward(self, item_his_eb, noclk_item_his_eb,mask):
        # print(item_his_eb.shape, noclk_item_his_eb.shape)
        # print(self.seq_len)
        # rnn_outputs, _ = self.rnn(item_his_eb, sequence_length=self.seq_len)
        
        prev_h = paddle.randn((item_his_eb.shape[0],self.EMBEDDING_DIM*2 ))
        rnn_outputs, _ = self.rnn(item_his_eb, prev_h)
        # print('rnn_outputs',rnn_outputs.shape)
        # print(rnn_outputs[:, :-1, :].shape,item_his_eb[:, 1:, :].shape)

        aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], item_his_eb[:, 1:, :],noclk_item_his_eb[:, 1:, :],mask)

        return aux_loss



class Prelu(nn.Layer):
    def __init__(self):
        super(Prelu,self).__init__()
        
    def forward(self,x):
        alpha = paddle.create_parameter(default_initializer=paddle.nn.initializer.Constant(value=0.1),
                                                shape=[x.shape[-1]],
                                                dtype='float32')

        res = paddle.maximum(paddle.to_tensor([0.0],dtype=x.dtype),x) + alpha * paddle.minimum(paddle.to_tensor([0.0],dtype=x.dtype),x)
        return res


class Dice(nn.Layer):
    def __init__(self,axis,epsilon):  
        super(Dice,self).__init__()
        self.axis = axis
        self.epsilon = epsilon
        

    def forward(self,_x):
        alphas = paddle.create_parameter(default_initializer=paddle.nn.initializer.Constant(value=0.0),
                                                shape=[_x.shape[-1]],
                                                dtype='float32')
        
        input_shape = list(_x.shape)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # case: train mode (uses stats of the current batch)
        mean = paddle.mean(_x, axis=reduction_axes)
        brodcast_mean = paddle.reshape(mean, broadcast_shape)
        std = paddle.mean(paddle.square(_x - brodcast_mean) + self.epsilon, axis=reduction_axes)
        std = paddle.sqrt(std)
        brodcast_std = paddle.reshape(std, broadcast_shape)
        x_normed = (_x - brodcast_mean) / (brodcast_std + self.epsilon)
        x_p = F.sigmoid(x_normed)

        return alphas * (1.0 - x_p) * _x + x_p * _x




