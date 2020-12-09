<font size=6>**项目目录**</font>
<br><br>
&emsp;&emsp;;&emsp;&emsp;&emsp;<font size=5.5>**说明：项目主体分为三部分：**</font>
<br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/6e6d64a65daa42ae9a368caf0f8e501a53f000c5045f40c7a96b1284c15123d5)
<br><br>
&emsp;&emsp;;&emsp;&emsp;&emsp;<font size=5.5>**目录：**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>1、实践体验</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>2、TSN网络与数据集结构的介绍</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>2.1.TSN网络简要介绍</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>2.2.数据集的结构</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>2.3.自制数据集</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>3、体验TSN训练过程</font>
<br><br><br><br>

<font size=5>**1.实践体验(请fork后按照[演示视频](https://www.bilibili.com/video/BV1D54y1v7aE)的方法体验测试实践的快感！)**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=5>说明：由于用的数据量比较少，所以不同人的手的测试效果可能不大一样，欢迎大家自制数据集进行尝试。</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/6dbc29b43fe74754b9da61754ffbb803896a26e10f10421486d93a118d4b4040)
<br><br>


```python
# 将测试数据的数据集进行转化
!pip install wget
!python GetTestdata.py
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Requirement already satisfied: wget in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (3.2)
    ['test', '.ipynb_checkpoints']
    {'test': 0}
    {'test': 0}
    testJpg test
    4



```python
# 预测该视频代表的手语信息
!python freeze_infer.py
```

    {'MODEL': {'name': 'TSN', 'format': 'pkl', 'num_classes': 10, 'seg_num': 3, 'seglen': 1, 'image_mean': [0.485, 0.456, 0.406], 'image_std': [0.229, 0.224, 0.225], 'num_layers': 50}, 'TRAIN': {'epoch': 45, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'use_gpu': True, 'num_gpus': 1, 'filelist': './dataset/train.txt', 'learning_rate': 0.01, 'learning_rate_decay': 0.1, 'l2_weight_decay': 0.0001, 'momentum': 0.9, 'total_videos': 80}, 'VALID': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 2, 'filelist': './dataset/val.txt'}, 'TEST': {'seg_num': 7, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'filelist': './dataset/test.txt'}, 'INFER': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 1, 'filelist': './test/test.txt'}}
    2020-08-06 23:37:47,236-INFO: [INFER] infer finished. average time: 0.44347667694091797
    ['a', ['walk'], [0.9852724671363831]]


<font size=5>**2.TSN网络与数据集结构的介绍**</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;<font size=4>**2.1.TSN网络简要介绍**</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=4>TSN（Temporal Segment Networks）是由空间流卷积网络和时间流卷积网络构成，其使用从整个视频中稀疏地采样一系列短片段的方式来代替稠密采样，这样既能捕获视频全局信息，也能去除冗余，降低计算量。稀疏采样得到的每个片段都将给出其本身对于行为类别的初步预测，从这些片段的“特点”(即通过用每帧特征平均融合后得到视频的整体特征)来得到视频级的分类预测结果。在学习过程中，通过迭代更新模型参数来优化视频级预测的损失值。网络结构如下图：</font>
<br>
&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/c73e53f670e449bcab37495bbcd165ac5ebdd5bb96584fe3b7ab3ea580f8e2e9)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;论文原文：[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)
<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;关于TSN网络结构的博客推荐1：[TSN(Temporal Segment Networks)算法笔记](https://blog.csdn.net/u014380165/article/details/79029309) 
<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;关于TSN网络结构的博客推荐2：[[行为识别论文详解]TSN(Temporal Segment Networks)](https://blog.csdn.net/zhuiqiuk/article/details/88377708?biz_id=102&utm_term=TSN%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-88377708&spm=1018.2118.3001.4187)

<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>**2.2.数据集的结构**</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>**2.2.1.label.npy的结构**</font>
<br><br>


```python
import numpy as np
label = np.load('label.npy', allow_pickle=True)
print(label)
```

    {'please': 0, 'walk': 1, 'come': 2}


&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>说明：label.npy的结构为一个字典，key为视频对应的标签，如'please'、'walk'等，而value为int值用以区分不同类。</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>**2.2.2.pkl的结构**</font>
<br><br>


```python
import pickle
f = open('dataset/train/come1.pkl', 'rb')
pkl_file = pickle.load(f)
f.close()

print(pkl_file)
```

    ('come1', 2, ['dataset/comeJpg/come1/come1_1.jpg', 'dataset/comeJpg/come1/come1_2.jpg', 'dataset/comeJpg/come1/come1_3.jpg', 'dataset/comeJpg/come1/come1_4.jpg', 'dataset/comeJpg/come1/come1_5.jpg', 'dataset/comeJpg/come1/come1_6.jpg', 'dataset/comeJpg/come1/come1_7.jpg', 'dataset/comeJpg/come1/come1_8.jpg', 'dataset/comeJpg/come1/come1_9.jpg', 'dataset/comeJpg/come1/come1_10.jpg', 'dataset/comeJpg/come1/come1_11.jpg', 'dataset/comeJpg/come1/come1_12.jpg', 'dataset/comeJpg/come1/come1_13.jpg', 'dataset/comeJpg/come1/come1_14.jpg', 'dataset/comeJpg/come1/come1_15.jpg', 'dataset/comeJpg/come1/come1_16.jpg', 'dataset/comeJpg/come1/come1_17.jpg', 'dataset/comeJpg/come1/come1_18.jpg', 'dataset/comeJpg/come1/come1_19.jpg', 'dataset/comeJpg/come1/come1_20.jpg', 'dataset/comeJpg/come1/come1_21.jpg', 'dataset/comeJpg/come1/come1_22.jpg', 'dataset/comeJpg/come1/come1_23.jpg', 'dataset/comeJpg/come1/come1_24.jpg', 'dataset/comeJpg/come1/come1_25.jpg', 'dataset/comeJpg/come1/come1_26.jpg', 'dataset/comeJpg/come1/come1_27.jpg', 'dataset/comeJpg/come1/come1_28.jpg', 'dataset/comeJpg/come1/come1_29.jpg', 'dataset/comeJpg/come1/come1_30.jpg', 'dataset/comeJpg/come1/come1_31.jpg', 'dataset/comeJpg/come1/come1_32.jpg', 'dataset/comeJpg/come1/come1_33.jpg', 'dataset/comeJpg/come1/come1_34.jpg', 'dataset/comeJpg/come1/come1_35.jpg', 'dataset/comeJpg/come1/come1_36.jpg', 'dataset/comeJpg/come1/come1_37.jpg', 'dataset/comeJpg/come1/come1_38.jpg'])


&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=3.5>说明：pkl的结构为：(文件名，该文件对应标签的int值，[视频帧1的地址，视频帧2的地址，视频帧3的地址.....])</font>
<br><br>

&emsp;&emsp;&emsp;&emsp;<font size=4>**2.3.自制数据集**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=4>说明：作者已经准备好了，如果是自制数据集，请解压自制数据集后运行一次GetDataset.py即可，不然会生成重复的文件QAQ</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/ee868dc3377e4dd398a226affdfcddf05be56d90785743d6945b91ba13b239cd)
<br><br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>数据集制作流程：</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/9f288b682207492eb6035327f9fbfb7f0f0e4c38af584cc4a98e07abb898246f)
<br><br>


```python
# 解压数据集
!unzip data/data48137/dataset.zip -d /home/aistudio/
```


```python
# 生成pkl等训练所需文件
!python GetDataset.py
```

    {'come': 0, 'walk': 1, 'please': 2}
    {'come': 0, 'walk': 1, 'please': 2}
    comeJpg come
    walkJpg walk
    pleaseJpg please
    54
    3
    3


<font size=5>**3、体验TSN训练过程**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<font size=5>流程图：</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/39e4bbeac6574c598989a66fb6a1defcd8e2e7fa5ec04a8aa148d38d0efb7344)
<br><br>


```python
# 开始训练
!pip install wget
!python train.py --model_name TSN \
                    --epoch 60 \
                    --save_dir 'checkpoints_models' \
                    --use_gpu True \
                    --pretrain data/ResNet50_pretrained
```


```python
# 固化模型
!python freeze.py --weights 'checkpoints_models'
```


```python
# 对test.txt中的样本进行推断
!python infer.py --weights 'checkpoints_models' --use_gpu True --save_dir 'infer'
```

<br><br>
<font size=5>**参考：**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>参考项目：[项目](https://aistudio.baidu.com/aistudio/projectdetail/112194)</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>参考飞桨官方demo：[飞桨官方demo](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/video)</font>
<br><br>

<font size=5>**项目总结**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=5>到此为止，本项目带您实现了用TSN网络开发了手语识别系统，您可以尝试自己录制视频并上传到本项目中进行尝试，希望本项目能给您带来启发，也希望本项目能对搭建起不太精通手语的人群与失聪人群交流的桥梁做出微小的贡献。</font>
<br><br>

<font size=5>**项目更新**</font>
<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=5>前面的项目实现了单一手语的视频的识别，基于前面的项目进一步开发了能实现对多手语的视频的识别的新项目。在新项目中，作者认为对于某一手语动作的持续时间约为2~3秒，在考虑该网络稳定性足够好的情况下，抽取出一部分视频，其所提供的特征信息也是足够的，所以作者做了一个1s的滑窗分段截取视频，用分段的视频所提供的特征进行预测，从而实现对多手语视频的手语分类并识别。经下面的实验证明，效果较好</font>
<br><br>


```python
# 清空文件
!rm -r test/*
!mkdir test/test
```


```python
# 制作1s的片段，用作特征以区分不同动作
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def clip_video(source_file, target_file, start_time, stop_time):

    source_video = VideoFileClip(source_file)
    video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
    video.write_videofile(target_file)

time = 1
for i in range(5):
    clip_video('test/test/test.mp4', 'test/test/test_'+str(i)+'.mp4', i * time, (i + 1) * time)

# 清空源文件
!rm test/test/test.mp4

# 制作数据集
!pip install wget
!python GetVideodata.py
```


```python
!python infer_video.py
```

    {'MODEL': {'name': 'TSN', 'format': 'pkl', 'num_classes': 10, 'seg_num': 3, 'seglen': 1, 'image_mean': [0.485, 0.456, 0.406], 'image_std': [0.229, 0.224, 0.225], 'num_layers': 50}, 'TRAIN': {'epoch': 45, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'use_gpu': True, 'num_gpus': 1, 'filelist': './dataset/train.txt', 'learning_rate': 0.01, 'learning_rate_decay': 0.1, 'l2_weight_decay': 0.0001, 'momentum': 0.9, 'total_videos': 80}, 'VALID': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 2, 'filelist': './dataset/val.txt'}, 'TEST': {'seg_num': 7, 'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 10, 'filelist': './dataset/test.txt'}, 'INFER': {'short_size': 240, 'target_size': 224, 'num_reader_threads': 1, 'buf_size': 1024, 'batch_size': 1, 'filelist': './test/test.txt'}}
    2020-08-06 23:38:25,440-INFO: [INFER] infer finished. average time: 0.396468448638916
    这里的手语表达的意思是：  please walk 


<br>
