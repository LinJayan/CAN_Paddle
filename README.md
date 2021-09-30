#### 基于Paddle复现CAN模型
##### **论文名称**<a href="https://arxiv.org/pdf/2011.05625.pdf">CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction</a>

### **一、简介**

    受深度学习成功的启发，最近的工业点击通过率(CTR)预测模型已经实现了从传统的浅层方法向深度方法的过渡。深度神经网络(DNNs)以其能够自动从原始特征中学习非线性交互而闻名，然而，非线性特征交互是以一种隐式的方式学习的。非线性相互作用可能难以捕获，并明确地建模原始特征的协同作用有利于CTR预测。协同作用是指特征对最终预测的集体效应。在本文中，我们认为目前的CTR模型并没有充分探索特征协同作用的潜力。我们进行了实验，结果表明，特征协同作用的影响被严重低估了。基于我们的观察，我们提出了特征协同网络(CAN)来探索特征协同网络的潜力。该模型能够有效、有效地捕获特征的协同作用，提高了模型的性能，同时降低了存储和计算消耗。

模型结构图如下所示，左图Co-Action Unit是论文的主要创新之处，可作为独立的模块应用，右边的图是Co-Action Unit的一般范式应用。
![](https://ai-studio-static-online.cdn.bcebos.com/08b5f22d797b42ce9dc624774214593d30c227cb1c924f2c93b5e712707a2d9d)


### **二、复现精度**

**复现要求**：Amazon数据集 AUC>=0.7690

**本次复现结果**：AUC = 0.754。

![](https://ai-studio-static-online.cdn.bcebos.com/96b464b27d5b4c298379080a52f81cfd29065d72fb7441d2aaf4bfb03392ad18)



### **三、数据集**

Amazon数据集：

(1)<a href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz">reviews_Books</a>

(2)<a href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz">meta_Books</a>

本次复现使用数据，沿用了<a href="https://github.com/mouna99/dien">DIEN</a>中提供处理好的数据：local_train_splitByUser，local_test_splitByUser，uid_voc.pkl，mid_voc.pkl，cat_voc.pkl，item_carte_voc.pkl，cate_carte_voc.pkl，其中item_carte_voc.pkl，cate_carte_voc.pkl两个数据使用process_data/generate_voc.py处理得到。



### **四、环境依赖**
CPU、GPU均可，相应设置。

PaddlePaddle >= 2.1.2

Python >= 3.7

### **五、快速开始**

 ============================== Step 1,git clone 代码库 ==============================
 
git clone https://github.com/LinJayan/CAN_Paddle.git

============================== Step 2 download and prepare data ==============================

cd dataset && ../prepare_data.sh

============================== Step 3, train model ==============================

启动训练 (需注意当前是否是 GPU 环境）

python -u main.py -m config.yaml --mode train

============================== Step 4, test ==============================

模型预测 (需注意当前是否是 GPU 环境）

python -u main.py -m config.yaml --mode test

### **六、代码结构与详细说明**
```
├─CAN_Paddle
   ├─ process_data # 数据下载、处理脚本代码
        ├── process_data.py
        ├── local_aggretor.py
        ├── generate_voc.py
        ├── split_by_user.py
   ├─ utils # 工具包
        ├── __init__.py
        ├── save_load.py
        ├── env.py
        ├── preData.py
        ├── data_iterator.py
        ├── shuffle.py
   ├─ model.py # 模型代码
   ├─ prepare_data.sh # 数据预处理脚本
   ├─ main.py # 训练、测试入口主程序
   ├─ config.yaml # 配置文件
   ├─LICENSE #项目LICENSE
   ├─README.md #文档
```

