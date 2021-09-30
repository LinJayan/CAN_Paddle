import numpy
import json

import pickle as pkl
import random

import gzip

from utils import shuffle
from tqdm import tqdm

# import shuffle

# # py3，str编码问题.来自DIEN中的Issue！！！！
# data_iterator.py
# 改动1：
# def load_dict(filename):
# try:
# with open(filename, 'rb') as f:
# return unicode_to_utf8(json.load(f))
# except:
# with open(filename, 'rb') as f:
# # return unicode_to_utf8(pkl.load(f))
# return pkl.load(f)
# 改动2：
# f_meta = open("../data/item-info", "r", encoding='utf-8')
# 改动3：
# f_review = open("../data/reviews-info", "r", encoding='utf-8')

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def dict_unicode_to_utf8(d):
    return dict(((key[0].encode("UTF-8"), key[1].encode("UTF-8")), value) for (key,value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            # return unicode_to_utf8(json.load(f))
            return pkl.load(f)
    except:
        with open(filename, 'rb') as f:
            #return unicode_to_utf8(pkl.load(f))
            return pkl.load(f)


# def load_dict(filename):
#     try:
#         with open(filename, 'rb') as f:
#             return unicode_to_utf8(json.load(f))
#     except:
#         try:
#             with open(filename, 'rb') as f:
#                 return pkl.load(f)
#         except:
#             with open(filename, 'rb') as f:
#                 return dict_unicode_to_utf8(pkl.load(f))

# =======================================
#  源代码为python2,使用python3会导致不能数据迭代生成，原因是python3的str编码与python2有区别
# def load_dict(filename):
#     try:
#         with open(filename, 'rb') as f:
#             return unicode_to_utf8(json.load(f))
#     except:
#         try:
#             with open(filename, 'rb') as f:
#                 return unicode_to_utf8(pkl.load(f))
#         except:
#             with open(filename, 'rb') as f:
#                 return dict_unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
    
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:

    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_carte_voc,
                 cate_carte_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 label_type=1):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
            # self.source = open(source, 'rb')
        self.source_dicts = []
        #for source_dict in [uid_voc, mid_voc, cat_voc, cat_voc, cat_voc]:# 'item_carte_voc.pkl', 'cate_carte_voc.pkl']:
        
        for source_dict in [uid_voc, mid_voc, cat_voc, item_carte_voc, cate_carte_voc]:
            self.source_dicts.append(load_dict(source_dict))

        f_meta = open(item_info, "r" ,encoding='utf-8')
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map ={}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        f_review = open(reviews_info, "r", encoding='utf-8')
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])
        self.n_carte = [len(self.source_dicts[3]), len(self.source_dicts[4])]
        print("n_uid=%d, n_mid=%d, n_cat=%d" % (self.n_uid, self.n_mid, self.n_cat))

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False
        self.label_type = label_type

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat, self.n_carte

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source= shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                    
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0

        

                tmp = []
                item_carte = []
                for fea in ss[4].split(""):

                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                    i_c = self.source_dicts[3][(ss[2], fea)] if (ss[2], fea) in self.source_dicts[3] else 0
                    item_carte.append(i_c)
                mid_list = tmp

                tmp1 = []
                cate_carte = []
                for fea in ss[5].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                    c_c = self.source_dicts[4][(ss[3], fea)] if (ss[3], fea) in self.source_dicts[4] else 0
                    cate_carte.append(c_c)
                cat_list = tmp1

                # read from source file and map to word index

                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                noclk_mid_list = []
                noclk_cat_list = []
               
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random)-1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                       
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                carte_list = [item_carte, cate_carte]
                source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list, carte_list])
                if self.label_type == 1:
                    target.append([float(ss[0])])
                else:
                    target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break

            
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.__next__()

        return source, target



from paddle.io import Dataset


class MyDataset(Dataset):
    def __init__(self, data_x, data_y, mode = 'train'):
        super(MyDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
           return self.data_x[idx]
        else:
           return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)


# import numpy
# import json
# import pickle as pkl
# import random

# import gzip

# from utils import shuffle

# def unicode_to_utf8(d):
#     return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

# def load_dict(filename):
#     try:
#         with open(filename, 'rb') as f:
#             return unicode_to_utf8(json.load(f))
#     except:
#         with open(filename, 'rb') as f:

#             # return unicode_to_utf8(pkl.load(f))
#             return pkl.load(f)


# def fopen(filename, mode='r'):
#     if filename.endswith('.gz'):
#         return gzip.open(filename, mode)
#     return open(filename, mode)


# class DataIterator:

#     def __init__(self, source,
#                  uid_voc,
#                  mid_voc,
#                  cat_voc,
#                  batch_size=128,
#                  maxlen=100,
#                  skip_empty=False,
#                  shuffle_each_epoch=False,
#                  sort_by_length=True,
#                  max_batch_size=20,
#                  minlen=None):
#         if shuffle_each_epoch:
#             self.source_orig = source
#             self.source = shuffle.main(self.source_orig, temporary=True)
#         else:
#             self.source = fopen(source, 'r')
#         self.source_dicts = []
#         for source_dict in [uid_voc, mid_voc, cat_voc]:
#             self.source_dicts.append(load_dict(source_dict))

#         f_meta = open("item-info", "r")
#         meta_map = {}
#         for line in f_meta:
#             arr = line.strip().split("\t")
#             if arr[0] not in meta_map:
#                 meta_map[arr[0]] = arr[1]
#         self.meta_id_map ={}
#         for key in meta_map:
#             val = meta_map[key]
#             if key in self.source_dicts[1]:
#                 mid_idx = self.source_dicts[1][key]
#             else:
#                 mid_idx = 0
#             if val in self.source_dicts[2]:
#                 cat_idx = self.source_dicts[2][val]
#             else:
#                 cat_idx = 0
#             self.meta_id_map[mid_idx] = cat_idx

#         f_review = open("reviews-info", "r")
#         self.mid_list_for_random = []
#         for line in f_review:
#             arr = line.strip().split("\t")
#             tmp_idx = 0
#             if arr[1] in self.source_dicts[1]:
#                 tmp_idx = self.source_dicts[1][arr[1]]
#             self.mid_list_for_random.append(tmp_idx)

#         self.batch_size = batch_size
#         self.maxlen = maxlen
#         self.minlen = minlen
#         self.skip_empty = skip_empty

#         self.n_uid = len(self.source_dicts[0])
#         self.n_mid = len(self.source_dicts[1])
#         self.n_cat = len(self.source_dicts[2])

#         self.shuffle = shuffle_each_epoch
#         self.sort_by_length = sort_by_length

#         self.source_buffer = []
#         self.k = batch_size * max_batch_size

#         self.end_of_data = False

#     def get_n(self):
#         return self.n_uid, self.n_mid, self.n_cat

#     def __iter__(self):
#         return self

#     def reset(self):
#         if self.shuffle:
#             self.source= shuffle.main(self.source_orig, temporary=True)
#         else:
#             self.source.seek(0)

#     def next(self):
#         if self.end_of_data:
#             self.end_of_data = False
#             self.reset()
#             raise StopIteration

#         source = []
#         target = []

#         if len(self.source_buffer) == 0:
#             for k_ in xrange(self.k):
#                 ss = self.source.readline()
#                 if ss == "":
#                     break
#                 self.source_buffer.append(ss.strip("\n").split("\t"))

#             # sort by  history behavior length
#             if self.sort_by_length:
#                 his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
#                 tidx = his_length.argsort()

#                 _sbuf = [self.source_buffer[i] for i in tidx]
#                 self.source_buffer = _sbuf
#             else:
#                 self.source_buffer.reverse()

#         if len(self.source_buffer) == 0:
#             self.end_of_data = False
#             self.reset()
#             raise StopIteration

#         try:

#             # actual work here
#             while True:

#                 # read from source file and map to word index
#                 try:
#                     ss = self.source_buffer.pop()
#                 except IndexError:
#                     break

#                 uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
#                 mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
#                 cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
#                 tmp = []
#                 for fea in ss[4].split(""):
#                     m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
#                     tmp.append(m)
#                 mid_list = tmp

#                 tmp1 = []
#                 for fea in ss[5].split(""):
#                     c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
#                     tmp1.append(c)
#                 cat_list = tmp1

#                 # read from source file and map to word index

#                 #if len(mid_list) > self.maxlen:
#                 #    continue
#                 if self.minlen != None:
#                     if len(mid_list) >= self.minlen:
#                         continue
#                 if self.skip_empty and (not mid_list):
#                     continue

#                 noclk_mid_list = []
#                 noclk_cat_list = []
#                 for pos_mid in mid_list:
#                     noclk_tmp_mid = []
#                     noclk_tmp_cat = []
#                     noclk_index = 0
#                     while True:
#                         noclk_mid_indx = random.randint(0, len(self.mid_list_for_random)-1)
#                         noclk_mid = self.mid_list_for_random[noclk_mid_indx]
#                         if noclk_mid == pos_mid:
#                             continue
#                         noclk_tmp_mid.append(noclk_mid)
#                         noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
#                         noclk_index += 1
#                         if noclk_index >= 5:
#                             break
#                     noclk_mid_list.append(noclk_tmp_mid)
#                     noclk_cat_list.append(noclk_tmp_cat)
#                 source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list])
#                 target.append([float(ss[0]), 1-float(ss[0])])

#                 if len(source) >= self.batch_size or len(target) >= self.batch_size:
#                     break
#         except IOError:
#             self.end_of_data = True

#         # all sentence pairs in maxibatch filtered out because of length
#         if len(source) == 0 or len(target) == 0:
#             source, target = self.next()

#         return source, target
