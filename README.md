# AIS 异常轨迹探测
本项目是一个使用Pytorch实现的，基于Variational Recurrent Neural Network（VRNN）的无监督机器学习框架，
目标为提取高质量AIS数据中的正常航行特征并借此预测异常轨迹。
## 运行环境
详见requirement.txt\
此为充分条件，部分包未使用。\
主要使用包：（Python3.6）\
torch==1.7.1+cpu\
matplotlib==3.1.1\
scikit-learn==0.24.0\
scipy==1.5.4

HINT：开发使用Pytorch-CPU，大数据训练请使用GPU版本。
## 程序框架
1. data：\
    1）previewImage：存放预览图，按区域ID存放，图片名为生成时间\
    2）saves：存放模型参数，若需要按区域ID命名请更改config.py中参数\
    3）trainingData：存放预处理后的数据，txt格式，\
        数据意义为[MMSI,纬度，经度，速度，航向，船头所指方向，时间戳],\
        示例见202012.txt
2. config.py：\
    存储了几乎所有你可能需要更改的参数,路径，x_dim请勿更改，除非你想改变输入维度！\
    若不指明，部分函数会选择直接调用本文件的路径/参数作为默认。
3. DataPreprocessing.py：\
    对trainingData中读入的数据进行数据清洗和筛选，请注意数据质量。
4. TrackingDetectionModel.py: 主要类，包含全部功能函数：\
    1）无预训练模型开始训练并自动保存模型：train_from_scratch\
    2）先输入预训练模型再继续训练以及保存：train_after_backbone\
    3）对某条轨迹（Tensor）进行异常检测，请注意应当和训练模型所持维度保持一致：abnormal_detection\
    4）向模型输入高质量数据：generate_dataloader\
5.main.py：使用例子\
6.vrnnPytorch: 变分循环神经网络Pytorch实现，参考项目以及论文见注释。

Feb.2021
    