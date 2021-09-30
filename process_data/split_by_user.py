# 采样训练数据作为训练和测试集
# import random

# fi1 = open("../dataset/local_train", "r")

# ftrain = open("../dataset/data/local_train_frac", "w")


# while True:
#     rand_int = random.randint(1, 30)
#     noclk_line = fi1.readline().strip()
#     clk_line = fi1.readline().strip()
#     if noclk_line == "" or clk_line == "":
#         break
#     if rand_int == 6:
#         print(noclk_line, file=ftrain)
#         print(clk_line, file=ftrain)
#     else:
#         continue


import random

# fi = open("local_test", "r")
fi = open("../dataset/data/local_train_frac", "r")
ftrain = open("../dataset/data/local_train_splitByUser", "w")
ftest = open("../dataset/data/local_test_splitByUser", "w")

while True:
    rand_int = random.randint(1, 10)
    noclk_line = fi.readline().strip()
    clk_line = fi.readline().strip()
    if noclk_line == "" or clk_line == "":
        break
    if rand_int == 6:
        print(noclk_line, file=ftest)
        print(clk_line, file=ftest)
   
    else:
        print(noclk_line, file=ftrain)
        print(clk_line, file=ftrain)
        
        


