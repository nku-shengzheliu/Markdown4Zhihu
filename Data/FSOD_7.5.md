# CVPR2021 Few-shot object detection论文整理

最近在看Few-shot目标检测领域的文章，也是第一次接触这个领域，整理了一些CVPR2021的文章和大家分享~文章会持续更新

有错误的话还各位小伙伴请多多指正(￣ω￣(￣ω￣〃 (￣ω￣〃)ゝ

## Few-shot object detection(FSOD)简介

传统的目标检测有一个base class集合 $\mathcal{C}_{b}$，和base dataset $\mathcal{D}_{b}$。$\mathcal{D}_{b}$包含丰富的数据$\{(x_i,y_i)\}$，$x_i$表示image，$y_i$表示对应的标注，包括在集合$\mathcal{C}_{b}$中的类别信息和bounding box坐标信息。

而对于Few-shot目标检测（FSOD）任务，除了上述两种集合之外，还有novel class集合$\mathcal{C}_{n}$和novel dataset $\mathcal{D}_{n}$。并且$\mathcal{C}_{b} \cap \mathcal{C}_{n}=\emptyset$。对于$k$-shot检测，每个在$\mathcal{C}_{n}$中的类有只有$k$个带标注信息的objects。Few-shot检测器先在$\mathcal{D}_{b}$中进行学习，然后在$k$很小的情况下快速地在$\mathcal{D}_{n}$上进行泛化，以此在类别集合为$\mathcal{C}_{b} \cup \mathcal{C}_{n}$的测试集上达到良好的性能。

通常Few-shot检测器有两个训练阶段，第一个阶段和传统的目标检测器类似，在$\mathcal{D}_{b}$上进行训练。第二个阶段在$\mathcal{D}_{b}$和$\mathcal{D}_{n}$的并集上进行微调，为了防止$\mathcal{D}_{b}$中样本数量带来的影响，通常在微调前对$\mathcal{D}_{b}$采样一个子集，使得$\mathcal{D}_{b}$和$\mathcal{D}_{n}$并集中每一类样本的数量更均衡。由于第二个阶段内class数量的增多，往往在检测器的box classification和localization部分插入更多的class-specific参数，基于novel objects来训练它们。

## 1. Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection

> paper：https://openaccess.thecvf.com/content/CVPR2021/html/Li_Few-Shot_Object_Detection_via_Classification_Refinement_and_Distractor_Retreatment_CVPR_2021_paper.html
>
> code：None

### 1.1 Motivation

 数据稀缺(data scarcity)是FSOD中的难点。检测器的性能受到如下两方面数据数量的影响：

* explicit shot：指在novel class中可用的标签目标。
* implicit shot：指初始化检测器的backbone时，已经包含在其中的关于novel class的先验知识。

这些数据数量的影响有多大？作者通过图1给出了性能比较：

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705164326468.png" alt="image-20210705164326468" style="zoom:50%;" />

其中实线的变化反映了explicit shot对性能的影响。虚线则是移除backbone中的implicit shot信息后的性能。可以看到本文的模型（红线）在这两种信息变化后仍然有较好的表现。

作者认为**只利用视觉信息**是不够的。不管视觉信息是否可用，base class和novel class之间的语义关系信息是不变的。

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705164213204.png" alt="image-20210705164213204" style="zoom:50%;" />

例如上图，如果有先验知识：“自行车”这个新类和“摩托车”类似，并它能够与“人”互动，可以携带“瓶子”，那么作者认为利用这些先验知识进行学习比仅仅利用几张图片进行学习更有效。本文提出的**Semantic Relation Reasoning Few-Shot Detector(SRR-FSD)**模型就探索了如何利用这些**语义关系**。

本文的contribution：

* 第一个在FSOD里探索语义关系推理的工作。
* 提出的SRR-FSD模型在novel class数据非常有限时性能也很稳定，即题目所说的shot-stable。
* 提出了一种新的FSOD设置，即图1虚线对应的在分类数据集中移除novel class，不给backbone隐含的先验知识。我理解是一种更加严苛的条件。在这种设置下模型仍然是shot-stable的。

### 1.2 Model

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705195137169.png" alt="image-20210705195137169" style="zoom:67%;" />

模型整体pipeline如上图，基于Faster R-CNN。改进点主要在网络顶层的classification subnet。经过Faster R-CNN提取的视觉特征经过全连接层后，为$\mathbf{v} \in \mathcal{R}^{d}$。在原始的检测器中，这个特征向量和参数矩阵$\mathbf{W} \in \mathcal{R}^{N \times d}$相乘，得到该proposal对应$N$个类别（包括background类）的概率，整个过程如下式：
$$
\mathbf{p}=\operatorname{softmax}(\mathbf{W} \mathbf{v}+\mathbf{b})
$$
为了使用来自语言模态的语义关系信息，先使用$\mathbf{P} \in \mathcal{R}^{d_e \times d}$将$\mathbf{v}$映射到预先定义的$d_e$维的**语义空间**，目标是把proposal的视觉特征$\mathbf v$进行映射后得到的$d_e$维的向量和它的类别（比如cat, dog, people...）的word embedding一致。这个word embedding就是上面提到的语义空间，使用$\mathbf{W_e} \in \mathcal{R}^{N \times d_e}$表示，直观上理解就是$N$个类别，每个类别的embedding是$d_e$维，比如使用Glove词向量$d_e$就可以取50,300等，不同的词向量之间计算相似度能够反映两个词之间在语义上的距离，举个例子，比如man和woman这两个词的词向量的相似度就应该比man和dog更近（舔狗另说）。上面的公式现在可以表示为：
$$
\mathbf{p}=\operatorname{softmax}(\mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
$$
其实就是把原来的参数矩阵$\mathbf{W} \in \mathcal{R}^{N \times d}$变成两部分了，$\mathbf{W_e}$在训练的时候是固定的，它是如何得到的后面讲，模型训练时只学习参数$\mathbf{P}$。

上面的过程其实没有引入关系信息，类别和类别之间还没有交互。为了利用起来关系，作者使用了一个知识图(knowledge graph)来建模relationship，使用$\mathbf G$来表示，它是$N \times N$的邻接矩阵，表示类别之间的关联强度。最终公式表示为：
$$
\mathbf{p}=\operatorname{softmax}(\mathbf{G} \mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
$$
那么$\mathbf{G}$又是如何实现的？本文使用了一种dynamic relation graph方法，见下图：

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705205304138.png" alt="image-20210705205304138" style="zoom:67%;" />

作者称借鉴了self-attention的思想，原始的$\mathbf{W_e}$经过三个linear layers $f,g,h$进行转换，我理解是得到了query, key和value对应的向量，经过处理后的$W_e$融合了每个embedding的信息。对于初始的word embedding $W_e$，文中提到使用300维的word2Vec词向量，在relation reasoning module里将维度降低为32。

### 1.3 Experiments

SRR-FSD在VOC和COCO上进行实验。

对于VOC数据集，15个类作为base class，5个类作为novel class，有3种不同的划分方式，结果如下：

![image-20210705211707795](E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705211707795.png)

![image-20210705212125383](E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705212125383.png)

对于COCO数据集，使用和VOC数据集有重叠的20个类作为novel class，剩下的60个类作为base class。结果如下：

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705212246322.png" alt="image-20210705212246322" style="zoom:50%;" />

消融实验在VOC数据集上探究了semantic space projection(SSP)，relation reasoning(RR)，Decoupled Fine-tuning(DF)的影响：

![image-20210705212442578](E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705212442578.png)

以及三种不同的word embedding增强方式：

<img src="E:\1_CV论文整理\Tools\Markdown4Zhihu\Data\FSOD_7.5\image-20210705212522830.png" alt="image-20210705212522830" style="zoom:50%;" />

























