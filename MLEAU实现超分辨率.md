# <center>Image Super-Resolution via Multi-Level Edge Embedding and Aggregated Attentive Upsampler Network</center>
SR(超分辨率)旨在恢复丢失的纹理和结构，并生成高分辨率图像内容。性能比较好的SR网络通常采用非常深的模型来获得空间精确的结果。但代价是长期上下文信息的丢失。通常缺乏保持空间细节和上下文信息之间的平衡。对于SR这种恢复应用。整个网络通常需要有效的保存低频信息和重构高频细节。本文通过收集上下文内容和恢复整个网络中的多频率信息来保持空间精确表示，通过学习一组丰富的特征，除了结合来自多个尺度上下文信息之外，还要同时保留SR的空间细节。

### SR三大方向
>- $deep\quad Learning-Based\quad Image \quad Super-Resolution$

>- $Attention-Based\quad Iamge\quad Super-Resolution$

>- $Upsampler\quad Techniques$

## 架构
一个初始化特征提取网络(IFEN),一个基于非局部和局部注意力的(NLLA)块和聚合注意力上采样器(AAU)块，具体来说就是使用IFEN网络通过卷积层将输入LR图像表示为一组特征图，其中PRELU作为激活函数，将提取的LR特征作为输入，提出的NLLA块通过更加关注细节保真度来专注于提取更多的信息特征，然后通过AAU将获得的深度特征嵌入到详细的放大特征当中。
<div align="center"><img src="image\MLEAU架构.PNG"></img></div>
为了平衡局部属性和非局部属性,建立了非线性局部分析块和基于局部残差注意的增强模块。这样使得即可收集局部感受野内的上下文信息，而且可以利用局部区域之外的信息，非局部操作对于具有重复细节的图像更有用，而局部操作适合于具有复杂纹理的图像，当它们一起使用时，可以相互补充并提高重建性能