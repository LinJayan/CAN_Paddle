import pickle

# 需要手动修改路径，产生训练和测试的数据
f_train = open("../dataset/data/local_train_frac", "r")
uid_dict = {}
mid_dict = {}
cat_dict = {}
item_carte_dict = {}
cate_carte_dict = {}

iddd = 0
for line in f_train:
    arr = line.strip("\n").split("\t")
    clk = arr[0]
    uid = arr[1]
    mid = arr[2]
    cat = arr[3]
    mid_list = arr[4]
    cat_list = arr[5]
    if uid not in uid_dict:
        uid_dict[uid] = 0
    uid_dict[uid] += 1
    if mid not in mid_dict:
        mid_dict[mid] = 0
    mid_dict[mid] += 1
    if cat not in cat_dict:
        cat_dict[cat] = 0
    cat_dict[cat] += 1
    if len(mid_list) == 0:
        continue
    for m in mid_list.split(""):
        if m not in mid_dict:
            mid_dict[m] = 0
        mid_dict[m] += 1
        if (mid, m) not in item_carte_dict:
            item_carte_dict[(mid, m)] = 0
        item_carte_dict[(mid, m)] += 1
    #print iddd
    iddd+=1
    for c in cat_list.split(""):
        if c not in cat_dict:
            cat_dict[c] = 0
        cat_dict[c] += 1
        if (cat, c) not in cate_carte_dict:
            cate_carte_dict[(cat, c)] = 0
        cate_carte_dict[(cat, c)] += 1

sorted_uid_dict = sorted(iter(uid_dict.items()), key=lambda x:x[1], reverse=True)
sorted_mid_dict = sorted(iter(mid_dict.items()), key=lambda x:x[1], reverse=True)
sorted_cat_dict = sorted(iter(cat_dict.items()), key=lambda x:x[1], reverse=True)
sorted_item_carte_dict = sorted(iter(item_carte_dict.items()), key=lambda x:x[1], reverse=True)
sorted_cate_carte_dict = sorted(iter(cate_carte_dict.items()), key=lambda x:x[1], reverse=True)

uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1

mid_voc = {}
mid_voc["default_mid"] = 0
index = 1
for key, value in sorted_mid_dict:
    mid_voc[key] = index
    index += 1

cat_voc = {}
cat_voc["default_cat"] = 0
index = 1
for key, value in sorted_cat_dict:
    cat_voc[key] = index
    index += 1

item_carte_voc = {}
item_carte_voc["default_item_carte"] = 0
index = 1
for key, value in sorted_item_carte_dict:
    item_carte_voc[key] = index
    index += 1

cate_carte_voc = {}
cate_carte_voc["default_cate_carte"] = 0
index = 1
for key, value in sorted_cate_carte_dict:
    cate_carte_voc[key] = index
    index += 1

# print(uid_voc) # ,mid_voc,cat_voc,item_carte_voc,cate_carte_voc
pickle.dump(uid_voc, open("../dataset/data/uid_voc.pkl", "wb"))
pickle.dump(mid_voc, open("../dataset/data/mid_voc.pkl", "wb"))
pickle.dump(cat_voc, open("../dataset/data/cat_voc.pkl", "wb"))
pickle.dump(item_carte_voc, open("../dataset/data/item_carte_voc.pkl", "wb"))
pickle.dump(cate_carte_voc, open("../dataset/data/cate_carte_voc.pkl", "wb"))



