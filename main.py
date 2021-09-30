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

import numpy as np
import paddle

import time
import random
import sys
from utils.data_iterator import DataIterator,MyDataset
from utils.preData import prepare_data
from utils.save_load import *
from utils import envs
import model

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import argparse
import copy

import logging



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

set_seed(12345)

best_auc = 0.0

class DygraphModel():
    # define model
    def create_model(self, config):
        batch_size = config.get("runner.batch_size")
        train_file = config.get("runner.train_file")
        test_file = config.get("runner.test_file")
        uid_voc = config.get("runner.uid_voc")
        mid_voc = config.get("runner.mid_voc")
        cat_voc = config.get("runner.cat_voc")
        item_carte_voc = config.get("runner.item_carte_voc")
        cate_carte_voc = config.get("runner.cate_carte_voc")

        uid_voc_test = config.get("runner.uid_voc_test")
        mid_voc_test = config.get("runner.mid_voc_test")
        cat_voc_test = config.get("runner.cat_voc_test")
        item_carte_voc_test = config.get("runner.item_carte_voc_test")
        cate_carte_voc_test = config.get("runner.cate_carte_voc_test")

        item_info = config.get("runner.item_info")
        reviews_info = config.get("runner.reviews_info")

        label_type = config.get("hyper_parameters.label_type")
        EMBEDDING_DIM = config.get("hyper_parameters.EMBEDDING_DIM")
        HIDDEN_SIZE = config.get("hyper_parameters.HIDDEN_SIZE")
        ATTENTION_SIZE = config.get("hyper_parameters.ATTENTION_SIZE")
        maxlen = config.get("hyper_parameters.maxlen")
        dnn_layer_size = config.get("hyper_parameters.dnn_layer_size")
        use_negsampling = config.get("hyper_parameters.use_negsampling")
        maxlen = config.get("hyper_parameters.maxlen")
        
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc,item_carte_voc,cate_carte_voc,item_info,reviews_info, batch_size, maxlen, shuffle_each_epoch=False, label_type=label_type)
        test_data = DataIterator(test_file, uid_voc_test, mid_voc_test, cat_voc_test, item_carte_voc_test,cate_carte_voc_test,item_info,reviews_info, batch_size, maxlen, label_type=label_type)
        n_uid, n_mid, n_cate, n_carte = train_data.get_n()

        n_uid = paddle.to_tensor(n_uid,dtype='int32')
        n_mid = paddle.to_tensor(n_mid,dtype='int32')
        n_cate = paddle.to_tensor(n_cate,dtype='int32')
        n_carte = paddle.to_tensor(n_carte,dtype='int32')

        EMBEDDING_DIM = paddle.to_tensor(EMBEDDING_DIM,dtype='int32')
        HIDDEN_SIZE = paddle.to_tensor(HIDDEN_SIZE,dtype='int32')
        can_model = model.CANLayer(n_uid, n_mid, n_cate, n_carte, maxlen, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,dnn_layer_size,
                 use_negsampling = use_negsampling, use_softmax=True, use_coaction=True, use_cartes=False)

        return can_model,train_data,test_data

    # define loss function by predicts and label
    def create_loss(self, pred, label):
    
        cost = paddle.nn.functional.binary_cross_entropy(pred,label=paddle.cast(
                label, dtype="float32"))
       
        # cost = paddle.nn.functional.log_loss(
        #     input=pred, label=paddle.cast(
        #         label, dtype="float32"))

        avg_cost = paddle.mean(x=cost)

        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name
        

def get_all_inters_from_yaml(file, filters):
    _envs = envs.load_yaml(file)
    all_flattens = {}

    def fatten_env_namespace(namespace_nests, local_envs):
        for k, v in local_envs.items():
            if isinstance(v, dict):
                nests = copy.deepcopy(namespace_nests)
                nests.append(k)
                fatten_env_namespace(nests, v)
            elif (k == "dataset" or k == "phase" or
                  k == "runner") and isinstance(v, list):
                for i in v:
                    if i.get("name") is None:
                        raise ValueError("name must be in dataset list. ", v)
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    nests.append(i["name"])
                    fatten_env_namespace(nests, i)
            else:
                global_k = ".".join(namespace_nests + [k])
                all_flattens[global_k] = v

    fatten_env_namespace([], _envs)
    ret = {}
    for k, v in all_flattens.items():
        for f in filters:
            if k.startswith(f):
                ret[k] = v
    return ret


def get_abs_model(model):
    if model.startswith("paddlerec."):
        dir = envs.paddlerec_adapter(model)
        path = os.path.join(dir, "config.yaml")
    else:
        if not os.path.isfile(model):
            raise IOError("model config: {} invalid".format(model))
        path = model
    return path

def load_yaml(yaml_file, other_part=None):
    part_list = ["workspace", "runner", "hyper_parameters"]
    if other_part:
        part_list += other_part
    running_config = get_all_inters_from_yaml(yaml_file, part_list)
    return running_config

def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument('--mode',type=str, default='train')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y
    return auc


def test(test_data, model,model_path,epoch_id,print_step):

    load_model(model_path, model, prefix='rec')

    model.eval()

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    step = 0
    stored_arr = []
    best_auc = 0.
    total_auc = []
    for src, tgt in test_data:
        step += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, carte = prepare_data(src, tgt, return_neg=True)

        target = paddle.to_tensor(target)
            
        inputs = []
        for inp in [uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats, carte]:
            inputs.append(paddle.to_tensor(inp, dtype='int64'))

        outputs = model(inputs)

        prob = F.sigmoid(outputs)
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
        # total_auc.append(test_auc)

        # if (step % print_step) == 0:
        #     logger.info("epoch: {}, ".format(epoch_id) +"batch_id:{},".format(step)+ "test_auc:{:.6f}".format(test_auc))

    if best_auc < test_auc:
        best_auc = test_auc
    # average_auc = np.sum(total_auc) / step
    logger.info("Test AUC:{:.6f}".format(test_auc))
    return best_auc


def train(train_data, model,optimizer, print_step, args):

    load_model_class = DygraphModel()
    config = load_yaml(args.config_yaml)

    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            if type(config.get(key)) is int:
                value = int(value)
            if type(config.get(key)) is bool:
                value = (True if value.lower() == "true" else False)
            config[key] = value

    epochs = config.get("runner.epochs", None)
    
    batch_size = config.get("runner.batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    
    use_gpu = config.get("runner.use_gpu") 
    print('use_gpu:',use_gpu)
    
    paddle.set_device('gpu' if use_gpu else 'cpu')
    # print(paddle.get_device())

    model.train()

    metric_list, metric_list_name = load_model_class.create_metrics()

    epochs = config.get("runner.epochs")
    maxlen = config.get("hyper_parameters.maxlen")
    use_negsampling = config.get("hyper_parameters.use_negsampling")
    print('maxlen',maxlen)

    start_time = time.time()
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        train_run_cost = 0.
        train_start = time.time()
        stored_arr = []
        step = 0
        for src, tgt in train_data:
            step += 1

            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, carte = prepare_data(src, tgt, maxlen, return_neg=True)

            target = paddle.to_tensor(target,stop_gradient=False,dtype='int64')
            target_ = paddle.to_tensor(target,stop_gradient=False,dtype='float32')
            # print('target',target)
            
            inputs = []
            for inp in [uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats, carte]:
                inputs.append(paddle.to_tensor(inp, dtype='int64'))

        # ===========================================================
            if use_negsampling:
                outputs, aux_loss = model(inputs)
                prediction = F.sigmoid(outputs)                
                loss = DygraphModel().create_loss(pred=prediction, label=target)
                loss += aux_loss
            else:
                outputs  = model(inputs)
                prediction = F.sigmoid(outputs)              
                loss = DygraphModel().create_loss(pred=prediction, label=target)


            # prob = F.sigmoid(outputs)
            # prob_1 = prob[:, 0].tolist()
            # target_1 = target[:, 0].tolist()
            # for p ,t in zip(prob_1, target_1):
            #     stored_arr.append([p, t])
            # test_auc = calc_auc(stored_arr)
        # ============================================
            
            # 计算 loss
            # Cross-entropy loss and optimizer initialization
            # if use_negsampling:
            #     outputs, aux_loss = model(inputs)
            #     y_hat = paddle.nn.Sigmoid()(outputs)
            #     ctr_loss = - paddle.mean(paddle.concat([paddle.log(y_hat + 0.00000001) * target, paddle.log(1 - y_hat + 0.00000001) * (1-target)], axis=1))
            #     loss = ctr_loss
            #     loss += aux_loss
            # else:
            #     outputs = model(inputs)
            #     y_hat = paddle.nn.Sigmoid()(outputs)
            #     ctr_loss = - paddle.mean(paddle.concat([paddle.log(y_hat + 0.00000001) * target, paddle.log(1 - y_hat + 0.00000001) * (1-target)], axis=1))
            #     loss = ctr_loss
         # ============================================

            prob = F.sigmoid(outputs)
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            for p ,t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            test_auc = calc_auc(stored_arr)

            # Accuracy metric
            y_hat = paddle.nn.Sigmoid()(outputs)
            # print('=============')
            # print(paddle.round(y_hat))
            accuracy = paddle.mean(paddle.cast(paddle.equal(paddle.round(y_hat), target_), 'float32')).numpy()[0] 
            accuracy_sum += accuracy          
            
            l2_emb = 0.01
            # for param in model.fcn_net.embed.mid_batch_embedded.parameters():
            #     # loss += args.l2_emb * paddle.norm(param)
            #     loss += l2_emb * paddle.norm(param)
            # for param in model.fcn_net.embed.uid_batch_embedded.parameters():
            #     # loss += args.l2_emb * paddle.norm(param)
            #     loss += l2_emb * paddle.norm(param)
            
            loss.backward()
            loss_sum += loss.numpy()[0]
            # loss_sum += loss
            
            optimizer.minimize(loss)
            optimizer.step()
            
            optimizer.clear_grad()

            train_run_cost += time.time() - train_start
            # total_samples += batch_size

            metric_str = ""
            for metric_id in range(len(metric_list_name)):
                metric_str += (metric_list_name[metric_id] +
                        ":{:.6f}, ".format(metric_list[metric_id].accumulate())
                    )

            test_auc_str = "train_auc:{:.6f},".format(test_auc)
            # print_loss = loss.numpy()[0]
            loss_str = "loss:{:.6f}".format(loss_sum/print_step)
            # loss_str = "loss:".format(loss_sum/print_step)

            if (step % print_step) == 0:
                logger.info(
                        "epoch: {}, ".format(
                            epoch) + "batch_id:{},".format(step)+test_auc_str + 'train_acc:{:.4f},'.format(accuracy_sum/print_step)+loss_str)
                loss_sum = 0.
                accuracy_sum = 0.

        save_model(model, optimizer, model_save_path, epoch, prefix='rec')


    

def main(args):
    load_model_class = DygraphModel()
    config = load_yaml(args.config_yaml)

    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            if type(config.get(key)) is int:
                value = int(value)
            if type(config.get(key)) is bool:
                value = (True if value.lower() == "true" else False)
            config[key] = value

     # tools.vars
    train_file = config.get("runner.train_file", True)
    test_file = config.get("runner.test_file", False)
    uid_voc = config.get("runner.uid_voc", False)
    mid_voc = config.get("runner.mid_voc", None)
    cat_voc = config.get("runner.cat_voc", None)
    item_carte_voc = config.get("runner.item_carte_voc", None)
    cate_carte_voc = config.get("runner.cate_carte_voc", None)
    epochs = config.get("runner.epochs", None)
    print_step = config.get("runner.print_step", None)
    batch_size = config.get("runner.batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")

    learning_rate = config.get("hyper_parameters.optimizer.learning_rate", "model_output")

    logger.info("**************common.configs**********")
    logger.info(
        "train_file: {}, test_file: {}, uid_voc: {}, mid_voc: {}, cat_voc: {},item_carte_voc: {},cate_carte_voc: {}, learning_rate:{},epochs: {}, batch_size: {}, model_save_path: {}".
        format(train_file,test_file,uid_voc,mid_voc,cat_voc,item_carte_voc,cate_carte_voc,learning_rate,epochs,batch_size,model_save_path))
    logger.info("**************common.configs**********")

    
    model, train_data, test_data = load_model_class.create_model(config)
    optimizer = load_model_class.create_optimizer(model,config)

    if args.mode == "train":
        train(train_data, model,optimizer, print_step, args)
    else:
        start_epoch = 0
        end_epoch = 1
    
        for epoch_id in range(start_epoch, end_epoch):
            logger.info("load model epoch {}".format(epoch_id))
            model_path = os.path.join(model_save_path, str(epoch_id))

            test_auc = test(test_data, model, model_path,epoch_id, print_step)
            logger.info("epoch: {}, ".format(epoch_id) + "Best_auc:{:.6f},".format(test_auc)+"Done!")

    



if __name__ == '__main__':
    
    args = parse_args()
    
    main(args)


