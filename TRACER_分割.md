使用EFFicientNet作为主干编码器，并将现有的七个块合并为四个块，其中除了初始卷积块大小，输出分辨率发生了偏移，将每个编码器块的输出
表示为$E_i$并且掩蔽边缘注意模块最初应用于具有足够边界表示的第一个编码器块输出$E_1$，以利用增强的边缘信息并提高内存系哦啊了，在解码器端，
实现了联合和对象注意模块，它们聚合了多级特征并分别合并了编码器和解码器的输出。
在联合注意模块中，在$E_2$的尺度上继承了三个编码器块输出$E_2,E_3,E_4$,这部分是基于多核的感受野块获得的，因此我们通过联合注意力模块强调更清晰的
通道和空间信息，对象注意力模块提取具有互补边缘信息的不同对象，并利用该补充信息减少浅层编码器和解码器之间的差异。
物体注意力模块是由深度卷积块组成，最小化学习参数的数量以提高计算效率，最后，生成了四个深度监控图，分别是联合输出，两个对象注意力模块和深度监控图。


为了跟踪边缘信息，提出了掩蔽边缘注意模块，通过使用快速傅里叶变换来提取明确的边界，并增强了第一个编码器输出边界，现有使用边缘信息的方法，都不能在特征提取阶段利用显示边缘，因为这些方法需要更深编码器的输出来多的不同的边缘，因此本文的FFT仅仅使用第一个编码器中提取显示边缘，使用快速傅里叶变换和逆变换将第一个编码器表示分为高频和低频
$$\Large X_H=FFT^{-1}(f_r^H(FFT(X)))$$
其中$f_r^H$是一个高通滤波器消除指定半径r以外的所有频率。为了显示区分显式边缘，我们利用高通滤波器获得的高频具有足够的边界信息，当输入特征从频域转换到空域时，其中包含背景噪声，使用感受野运算RFB来消除噪声，从而生成显示边缘。

联合注意力模块用于聚合多级特征，并从通道和空间表示中检测更要的上下文，每个编码器输出$E_{i\in\{2,3,4\}}$分别聚合到32,64,28三个通道上，具体表示如下
$$\Large E'_2=E_2\otimes f(Up(E_3))\otimes f(Up(Up(E_4)))$$
$$\Large E''_3=f(cat[E_3\otimes f(Up(E_4)), f(Up(E_4))])$$
$$\Large E''_2=f(Up(E''_3))$$
我们通过$X=f(cat[E'_2,E''_2])\in \mathbb{R}^{(32+64+128)*H_2*W_2}$得到一个聚合表示，也就是$E_2$尺度，在聚合之后，这部分仍然是在通道和空间特征中相对重要的上下文信息，现有的研究已经将通道和空间注意模块独立应用于解码器和感受野块，尽管这两个空间相互依存，因此我们首先区分相对重要的通道上下文，并基于从通道上下文获得的互补置信分数强化空间信息。
<div align="center"><img src="image\fq_1.PNG"></div>

$\tilde{X}\in \mathbb{R}^{C\times1\times1}$表示通道池化，$F(\cdot)$表示$1\times1$卷积运算，通过使用自注意方法和softmax函数来获得上下文信息，通过sigmoid函数来区别重要通道。为了细化聚合表示，使用如下置信度通道权重$X_c=(X\otimes \alpha_c)+X$,后根据$\alpha_c$的分布和置信比$\gamma$保留置信通道
$$\Large \widetilde{X}_c=X_c\otimes mask\begin{cases} mask=1 & \alpha_c>F^{-1}(\gamma) \\ mask=0 & otherwise\end{cases}$$
这里的$F^{-1}(\gamma)$表示$\alpha_c$的$\gamma$的分位数，我们在分布$\alpha_c$下尾拍出来一个$\gamma$区域。然后，在空间上计算细化输入$\widetilde{X}_c$以区分显著对象并生成第一个解码器表示$\mathbb{R}^{1\times H_2\times W_2}$方程式表示如下：
$$\large D_0=\frac{exp(G_q(\tilde{X}_c)(G_k(\tilde{X}_c))^\top)}{\sum exp(G_q(\tilde{X}_c)(G_k(\tilde{X}_c))^\top)}G_v(\tilde{X}_c)+G_v(\tilde{X}_c)$$
这里$G(\cdot)$将输入特征投影到$\Large \tilde{X}_c\in \mathbb{R}^{1\times H_2 \times W_2}$使用具有$1\times1$核大小的卷积运算，将$DS_0$上采样到深度监控图。为了使用最小参数减少编码器和解码器表示之间的分布差异，我们组织了一个对象注意模块作为解码器，与现有研究相反，我们将D保持为单个通道以提高解码器效率，并且OAM从每个解码器表示$D_i\in \mathbb{R}^{1\times H\times W}$跟踪对象和互补边缘，wield细化显著对象，对象权重$\alpha_O$计算为$\alpha_O=\sigma(D_i)$,由于$\alpha_O$不能总是检测具有明确边缘区域的整个对象，因此生成一个互补的边缘权重来覆盖为检测的区域，对于D中每个像素$x_{ij}$,反转检测区域并消除与去噪比d相对应的背景噪声，以进行漏区检测，计算方式如下：
$$\Large \alpha_E=\begin{cases} 0&if(-\alpha(x_{ij})+1)>d \\
-\alpha(x_{ij})+1&otherwise
\end{cases} $$
合并编码器输出和解码器特征，为了减少差异，利用感受野运算RFB和上采样将$D_{i+1}$转化为$DS_{i+1}$

对于损失函数,我们结合了二元交叉熵(BCE)、IoU和L1损失函数来减少对象和背景之间的差异。尽管BCE和IoU被全局用于损失函数，但当所有像素被同等考虑时，这些函数会导致前景和背景之间的类别差异。与背景和显着对象中心的像素相比，与精细或显式边缘相邻的像素需要更多关注。 因此，我们提出了自适应像素强度（API）损失，它将像素强度 ω 应用于每个像素，如下所示：
$$\Large \omega_ij=(1-\lambda)\sum_{k\in K}|\frac{\sum_{h,w\in A_{i,j}y_{hw}^k} }{\sum_{h,w\in A_{ij} } }-y_{ij}|y_{ij}$$

