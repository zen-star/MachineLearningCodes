# 2018云从人头计数说明

*Author: 郑仕达*

*Team: GGSDT*

*Last Modified: 2019.02.14 12:43*


## 概述：
本竞赛的目标是实现给定图像中人头位置的框选和人头计数，为目标检测任务。
考察数据集，人头尺度多样、场景多样、有很多模糊和遮挡的人头，比较有难度。
经典的通用目标检测模型（如：Fast/Faster-RCNN、SSD、FPN、Yolo系列等），虽然一定程度上解决了目标检测中的一些通用问题，但是在针对本课题中多尺度、模糊、遮挡条件下的人头检测，仍然存在很大的提升空间。
考虑到人头检测和人脸检测具有一定的相似性，而且人脸检测在WIDERFACE数据竞赛上和论文有了非常多的成熟经验和模型积累，如SSH、Face RCNN、S3FD、FAN、PyramidBox等。
本次任务采用百度2018年下半年提出的PyramidBox人脸检测模型，根据人头检测的实际情况进行了适当的调整。
训练以ImageNet官方的预训练模型为骨架，在两块Titan XP（实际训练和测试只需要1块即可）上训练yuncong的数据集，得到了人头检测模型。


## 数据集特点：
yuncong_data数据集中包含了——
Mall: 固定视角的商场单一摄像头的拍摄图片
Part_A/B: 以室外为主的多尺度密集人头场景
our: 地铁站场景人头数多且模糊、商场场景背景复杂
UCSD: 校园室外场景，框尺寸相同，且比例偏大，人头普遍偏小
另外，存在几个问题——
1. 标注框相对于人头的大小在不同文件夹下不同，有些只框了头的一部分，有些框了整个上半身。
2. 存在无人、标注框质量很差、标注框超出图像等问题的图片，需要进行数据清洗。
针对以上对数据的了解，在data/widerface.py中，进行了相关清洗和处理工作。具体见“代码说明”和程序文件中的注释。


## 模型介绍：
PyramidBox是百度的一篇人脸检测论文提出的模型，截止2018.08在WIDERFACE数据集上排名第一，fddb上continuous score排名第一，discrete score排名第二。目前看来是一个非常出色的人脸检测模型。
这里会简要介绍一下PyramidBox的思想，不展开分析。如果有兴趣的话，可以参考arxiv上的论文原文(https://arxiv.org/pdf/1803.07737.pdf)。
1. PyramidBox是1-stage single shot的
2. 提出PyramidAnchors，设计contextual anchor通过半监督的方法，来有监督地学习高层feature map的上下文信息；具体地，结合了人头、人体等上下文信息，以检出小尺度、模糊、遮挡人脸；（本人头检测任务中，将人脸-人头-人体三级PyramidAnchors修改成了人头-人体两级PyramidAnchors）
3. 提出LFPN(Low-level Feature Pyramid Network)，联合合适的高层feature map语义信息（不使用高层，而是中层feature map特征）+低层feature map特征，让PyramidBox可以single shot检测所有尺度人脸；
4. 提出CPM(Context-sensitive Predict Module)，借鉴SSH+DSSD，使用wider+deeper的网络结构，融合人脸附近的上下文信息，并提出max-in-out层，提升检测分支中人脸检测+分类的准确率；
5. 提出DAS(Data Augmentation Sampling),增强不同尺度上的训练样本，特别是提升训练集中小尺度人脸的多样性
注1：以上2-4均为PyramidBox融合上下文信息检测人脸的关键部分，除了2中调整了PyramidAnchors，其余均直接使用到人头检测中。
注2：原文的骨架为VGG16，我们实现了VGG16和RESNET50骨架的PyramidBox，以对比性能


代码说明：
环境：Python 3.5、Pytorch 0.3.1、OpenCV 3、visdom


训练预测流程：
0. 文件目录结构：
训练数据： ./yuncong_data/
测试数据： ./yuncong_test_set/
测试文件列表： ./merged_list_test.txt
模型代码文件： ./GGSDT_PyramidBox/
终端执行代码路径： ./
1. 数据前处理：
Part_B_train.txt中，Part_B/train_data/IMG_188.jpg的末尾补上一个' 8'，或
去掉最后这个标注框。
将每个GT的.txt合成一个.txt，这里命名为nice_zsd.txt 
sh ./GGSDT_PyramidBox/extra/
