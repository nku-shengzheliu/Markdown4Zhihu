# CVPR2021 Few-shot object detection论文整理

最近在看Few-shot目标检测领域的文章，也是第一次接触这个领域，整理了一些CVPR2021的文章和大家分享~文章会持续更新

有错误的话还各位小伙伴请多多指正(￣ω￣(￣ω￣〃 (￣ω￣〃)ゝ

## Few-shot object detection(FSOD)简介

传统的目标检测有一个base class集合  <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{b}" alt="\mathcal{C}_{b}" class="ee_img tr_noresize" eeimg="1"> ，和base dataset  <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 。 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 包含丰富的数据 <img src="https://www.zhihu.com/equation?tex=\{(x_i,y_i)\}" alt="\{(x_i,y_i)\}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1"> 表示image， <img src="https://www.zhihu.com/equation?tex=y_i" alt="y_i" class="ee_img tr_noresize" eeimg="1"> 表示对应的标注，包括在集合 <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{b}" alt="\mathcal{C}_{b}" class="ee_img tr_noresize" eeimg="1"> 中的类别信息和bounding box坐标信息。

而对于Few-shot目标检测（FSOD）任务，除了上述两种集合之外，还有novel class集合 <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{n}" alt="\mathcal{C}_{n}" class="ee_img tr_noresize" eeimg="1"> 和novel dataset  <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{n}" alt="\mathcal{D}_{n}" class="ee_img tr_noresize" eeimg="1"> 。并且 <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{b} \cap \mathcal{C}_{n}=\emptyset" alt="\mathcal{C}_{b} \cap \mathcal{C}_{n}=\emptyset" class="ee_img tr_noresize" eeimg="1"> 。对于 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> -shot检测，每个在 <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{n}" alt="\mathcal{C}_{n}" class="ee_img tr_noresize" eeimg="1"> 中的类有只有 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 个带标注信息的objects。Few-shot检测器先在 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 中进行学习，然后在 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 很小的情况下快速地在 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{n}" alt="\mathcal{D}_{n}" class="ee_img tr_noresize" eeimg="1"> 上进行泛化，以此在类别集合为 <img src="https://www.zhihu.com/equation?tex=\mathcal{C}_{b} \cup \mathcal{C}_{n}" alt="\mathcal{C}_{b} \cup \mathcal{C}_{n}" class="ee_img tr_noresize" eeimg="1"> 的测试集上达到良好的性能。

通常Few-shot检测器有两个训练阶段，第一个阶段和传统的目标检测器类似，在 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 上进行训练。第二个阶段在 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{n}" alt="\mathcal{D}_{n}" class="ee_img tr_noresize" eeimg="1"> 的并集上进行微调，为了防止 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 中样本数量带来的影响，通常在微调前对 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 采样一个子集，使得 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{b}" alt="\mathcal{D}_{b}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{n}" alt="\mathcal{D}_{n}" class="ee_img tr_noresize" eeimg="1"> 并集中每一类样本的数量更均衡。由于第二个阶段内class数量的增多，往往在检测器的box classification和localization部分插入更多的class-specific参数，基于novel objects来训练它们。

## 1. Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection

> paper：https://openaccess.thecvf.com/content/CVPR2021/html/Li_Few-Shot_Object_Detection_via_Classification_Refinement_and_Distractor_Retreatment_CVPR_2021_paper.html
>
> code：None

### 1.1 Motivation

 数据稀缺(data scarcity)是FSOD中的难点。检测器的性能受到如下两方面数据数量的影响：

* explicit shot：指在novel class中可用的标签目标。
* implicit shot：指初始化检测器的backbone时，已经包含在其中的关于novel class的先验知识。

这些数据数量的影响有多大？作者通过图1给出了性能比较：

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705164326468.png" alt="image-20210705164326468" style="zoom:50%;" />

其中实线的变化反映了explicit shot对性能的影响。虚线则是移除backbone中的implicit shot信息后的性能。可以看到本文的模型（红线）在这两种信息变化后仍然有较好的表现。

作者认为**只利用视觉信息**是不够的。不管视觉信息是否可用，base class和novel class之间的语义关系信息是不变的。

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705164213204.jpg" alt="image-20210705164213204" style="zoom: 50%;" />

例如上图，如果有先验知识：“自行车”这个新类和“摩托车”类似，并它能够与“人”互动，可以携带“瓶子”，那么作者认为利用这些先验知识进行学习比仅仅利用几张图片进行学习更有效。本文提出的**Semantic Relation Reasoning Few-Shot Detector(SRR-FSD)**模型就探索了如何利用这些**语义关系**。

本文的contribution：

* 第一个在FSOD里探索语义关系推理的工作。
* 提出的SRR-FSD模型在novel class数据非常有限时性能也很稳定，即题目所说的shot-stable。
* 提出了一种新的FSOD设置，即图1虚线对应的在分类数据集中移除novel class，不给backbone隐含的先验知识。我理解是一种更加严苛的条件。在这种设置下模型仍然是shot-stable的。

### 1.2 Model

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705195137169.png" alt="image-20210705195137169" style="zoom: 67%;" />

模型整体pipeline如上图，基于Faster R-CNN。改进点主要在网络顶层的classification subnet。经过Faster R-CNN提取的视觉特征经过全连接层后，为 <img src="https://www.zhihu.com/equation?tex=\mathbf{v} \in \mathcal{R}^{d}" alt="\mathbf{v} \in \mathcal{R}^{d}" class="ee_img tr_noresize" eeimg="1"> 。在原始的检测器中，这个特征向量和参数矩阵 <img src="https://www.zhihu.com/equation?tex=\mathbf{W} \in \mathcal{R}^{N \times d}" alt="\mathbf{W} \in \mathcal{R}^{N \times d}" class="ee_img tr_noresize" eeimg="1"> 相乘，得到该proposal对应 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 个类别（包括background类）的概率，整个过程如下式：

<img src="https://www.zhihu.com/equation?tex=\mathbf{p}=\operatorname{softmax}(\mathbf{W} \mathbf{v}+\mathbf{b})
" alt="\mathbf{p}=\operatorname{softmax}(\mathbf{W} \mathbf{v}+\mathbf{b})
" class="ee_img tr_noresize" eeimg="1">
为了使用来自语言模态的语义关系信息，先使用 <img src="https://www.zhihu.com/equation?tex=\mathbf{P} \in \mathcal{R}^{d_e \times d}" alt="\mathbf{P} \in \mathcal{R}^{d_e \times d}" class="ee_img tr_noresize" eeimg="1"> 将 <img src="https://www.zhihu.com/equation?tex=\mathbf{v}" alt="\mathbf{v}" class="ee_img tr_noresize" eeimg="1"> 映射到预先定义的 <img src="https://www.zhihu.com/equation?tex=d_e" alt="d_e" class="ee_img tr_noresize" eeimg="1"> 维的**语义空间**，目标是把proposal的视觉特征 <img src="https://www.zhihu.com/equation?tex=\mathbf v" alt="\mathbf v" class="ee_img tr_noresize" eeimg="1"> 进行映射后得到的 <img src="https://www.zhihu.com/equation?tex=d_e" alt="d_e" class="ee_img tr_noresize" eeimg="1"> 维的向量和它的类别（比如cat, dog, people...）的word embedding一致。这个word embedding就是上面提到的语义空间，使用 <img src="https://www.zhihu.com/equation?tex=\mathbf{W_e} \in \mathcal{R}^{N \times d_e}" alt="\mathbf{W_e} \in \mathcal{R}^{N \times d_e}" class="ee_img tr_noresize" eeimg="1"> 表示，直观上理解就是 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 个类别，每个类别的embedding是 <img src="https://www.zhihu.com/equation?tex=d_e" alt="d_e" class="ee_img tr_noresize" eeimg="1"> 维，比如使用Glove词向量 <img src="https://www.zhihu.com/equation?tex=d_e" alt="d_e" class="ee_img tr_noresize" eeimg="1"> 就可以取50,300等，不同的词向量之间计算相似度能够反映两个词之间在语义上的距离，举个例子，比如man和woman这两个词的词向量的相似度就应该比man和dog更近（舔狗另说）。上面的公式现在可以表示为：

<img src="https://www.zhihu.com/equation?tex=\mathbf{p}=\operatorname{softmax}(\mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
" alt="\mathbf{p}=\operatorname{softmax}(\mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
" class="ee_img tr_noresize" eeimg="1">
其实就是把原来的参数矩阵 <img src="https://www.zhihu.com/equation?tex=\mathbf{W} \in \mathcal{R}^{N \times d}" alt="\mathbf{W} \in \mathcal{R}^{N \times d}" class="ee_img tr_noresize" eeimg="1"> 变成两部分了， <img src="https://www.zhihu.com/equation?tex=\mathbf{W_e}" alt="\mathbf{W_e}" class="ee_img tr_noresize" eeimg="1"> 在训练的时候是固定的，它是如何得到的后面讲，模型训练时只学习参数 <img src="https://www.zhihu.com/equation?tex=\mathbf{P}" alt="\mathbf{P}" class="ee_img tr_noresize" eeimg="1"> 。

上面的过程其实没有引入关系信息，类别和类别之间还没有交互。为了利用起来关系，作者使用了一个知识图(knowledge graph)来建模relationship，使用 <img src="https://www.zhihu.com/equation?tex=\mathbf G" alt="\mathbf G" class="ee_img tr_noresize" eeimg="1"> 来表示，它是 <img src="https://www.zhihu.com/equation?tex=N \times N" alt="N \times N" class="ee_img tr_noresize" eeimg="1"> 的邻接矩阵，表示类别之间的关联强度。最终公式表示为：

<img src="https://www.zhihu.com/equation?tex=\mathbf{p}=\operatorname{softmax}(\mathbf{G} \mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
" alt="\mathbf{p}=\operatorname{softmax}(\mathbf{G} \mathbf{W_e} \mathbf{P} \mathbf{v}+\mathbf{b})
" class="ee_img tr_noresize" eeimg="1">
那么 <img src="https://www.zhihu.com/equation?tex=\mathbf{G}" alt="\mathbf{G}" class="ee_img tr_noresize" eeimg="1"> 又是如何实现的？本文使用了一种dynamic relation graph方法，见下图：

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705205304138.png" alt="image-20210705205304138" style="zoom:67%;" />

作者称借鉴了self-attention的思想，原始的 <img src="https://www.zhihu.com/equation?tex=\mathbf{W_e}" alt="\mathbf{W_e}" class="ee_img tr_noresize" eeimg="1"> 经过三个linear layers  <img src="https://www.zhihu.com/equation?tex=f,g,h" alt="f,g,h" class="ee_img tr_noresize" eeimg="1"> 进行转换，我理解是得到了query, key和value对应的向量，经过处理后的 <img src="https://www.zhihu.com/equation?tex=W_e" alt="W_e" class="ee_img tr_noresize" eeimg="1"> 融合了每个embedding的信息。对于初始的word embedding  <img src="https://www.zhihu.com/equation?tex=W_e" alt="W_e" class="ee_img tr_noresize" eeimg="1"> ，文中提到使用300维的word2Vec词向量，在relation reasoning module里将维度降低为32。

### 1.3 Experiments

SRR-FSD在VOC和COCO上进行实验。

对于VOC数据集，15个类作为base class，5个类作为novel class，有3种不同的划分方式，结果如下：

![image-20210705211707795](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705211707795.png)

![image-20210705212125383](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705212125383.png)

对于COCO数据集，使用和VOC数据集有重叠的20个类作为novel class，剩下的60个类作为base class。结果如下：

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705212246322.png" alt="image-20210705212246322" style="zoom:50%;" />

消融实验在VOC数据集上探究了semantic space projection(SSP)，relation reasoning(RR)，Decoupled Fine-tuning(DF)的影响：

![image-20210705212442578](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705212442578.png)

以及三种不同的word embedding增强方式：

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/image-20210705212522830.png" alt="image-20210705212522830" style="zoom:50%;" />



## 2. Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection

> paper: https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection_CVPR_2021_paper.html
>
> code: https://github.com/hzhupku/DCNet

### 2.1 Motivation

本文提到，FSOD的研究可以分为两类：

* 基于meta-learning框架为class-specific的预测进行feature reweighting。
* 两阶段的方法（应该就是指上面那篇文章的做法），只微调网络的最后几层。

但是本文认为它们都面临几点困难：

* 对support features和query features之间的关系探索得不够充分。其中在support特征上进行**全局池化**操作来调整query分支较为常见，但是容易丢失详细的局部上下文信息。具体来说，物体的外观变形和遮挡是常见的问题，例如下图：

<img src="https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/21.jpg" alt="21" style="zoom: 50%;" />

​		由于没有足够的判别信息，模型无法学习用来预测class和bbox的关键特征。

* 尺度变化问题，在few-shot的设置下，具有scale-aware的特征提取器倾向于过拟合，导致模型在base class和novel class上的性能下降。

为了解决第一个困难，本文使用了**dense relation distillation module**来充分探索support样本集合，直观上来看，query和support物体是否属于同一类是依据它们之间的特征相似度来评判的。当物体外观变形或者有遮挡时，局部的细节特征占主导地位。因此作者采用了密集关系蒸馏机制让query和support特征在像素级别上匹配。

为了解决第二个困难，本文设计了**context-aware feature aggregation module**来捕捉不同尺度下的关键信息。使用自适应聚合机制，能够为不同尺度的物体保留上下文信息，具体实现是三种不同的池化方案+attention。

文章的contribution：

* Dense relation distillation module
* Adaptive context-aware feature aggregation module
* Extensive experiments

### 2.2 Model

在 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> -shot任务下，每个样本的构建过程如下：

1. 采样一组support image-mask pairs，使用 <img src="https://www.zhihu.com/equation?tex=\{x_i,y_i\}^N_{i=1}" alt="\{x_i,y_i\}^N_{i=1}" class="ee_img tr_noresize" eeimg="1"> 表示，其中 <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1"> 对应第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 个class中的一张RGB图片， <img src="https://www.zhihu.com/equation?tex=y_i" alt="y_i" class="ee_img tr_noresize" eeimg="1"> 是相应的二值的mask（类似分割的mask，bbox对应的区域是前景，其他是背景）， <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 是训练集中的类别数。
2. 采样一个query image  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> ，以及它的类别 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 。

模型的输入就是一组support pairs加上一个query image，输出是对query image的预测结果。

本文提出的**DCNet**模型包含两个模块：Dense Relation Distillation (DRD) module 和 Context-aware Feature Aggregation (CFA) module。整体pipeline如下：

![22](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/22.jpg)

* Dense Relation Distillation Module
  * 首先使用encoder得到query和support image的value map和key map。key map用来度量query特征和support特征之间的相似度，有助于确定在何处检索相关的support values。value map包含细节信息。具体地，对于query特征，它的key和value为： <img src="https://www.zhihu.com/equation?tex=k_{q} \in \mathbb{R}^{C / 8 \times H \times W}" alt="k_{q} \in \mathbb{R}^{C / 8 \times H \times W}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=v_{q} \in \mathbb{R}^{C / 2 \times H \times W}" alt="v_{q} \in \mathbb{R}^{C / 2 \times H \times W}" class="ee_img tr_noresize" eeimg="1"> 。对于support特征，它的key和value为： <img src="https://www.zhihu.com/equation?tex=k_{s} \in \mathbb{R}^{N \times C / 8 \times H \times W}" alt="k_{s} \in \mathbb{R}^{N \times C / 8 \times H \times W}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=v_{s} \in \mathbb{R}^{N \times C / 2 \times H \times W}" alt="v_{s} \in \mathbb{R}^{N \times C / 2 \times H \times W}" class="ee_img tr_noresize" eeimg="1"> 。 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 是特征维度， <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 是target class的数量（也是support样本的数量）。
  * 如图2，在Dense Relation Distillation Module中使用query和support的key map进行pixel-wise的相似度计算，得到 <img src="https://www.zhihu.com/equation?tex=W_{ij}" alt="W_{ij}" class="ee_img tr_noresize" eeimg="1"> 表示第i个query和第j个support之间的相似度。将得到的 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> 和support的value map矩阵内积然后和query的value map拼接。就是红框部分所示的过程 <img src="https://www.zhihu.com/equation?tex=y=concat[v_q,W*v_s]" alt="y=concat[v_q,W*v_s]" class="ee_img tr_noresize" eeimg="1"> 。将N个输出结果进行相加得到最终的结果。
  * 作者在这部分最后还进行了解释。之前的方法对support features使用全局池化操作，得到的是class-wise的特征，然后再去和query features进行交互。这样做确实会丢失一些细节的信息。本文使用的做法得到的是pixel-level的，能够保留更多信息。这种蒸馏方法可以视为non-local self-attention机制的扩展，可以视为一种cross attention机制。
* Context-aware Feature Aggregation
  * 上面的DRD模块总的来说，目的是为了得到**更精细的**query feature map。然后把这个特征送给RPN网络，通过ROI Align操作得到proposals以及对应的特征。作者在最后的pooling操作使用了多尺度的特征，即图中对应的 <img src="https://www.zhihu.com/equation?tex=4\times4,8\times8,12\times12" alt="4\times4,8\times8,12\times12" class="ee_img tr_noresize" eeimg="1"> 。通过这种方式作者希望保留更全面的特征表示。
  * 那么怎么融合这三种不同尺度的特征呢？可以看下图的图3，GAP表示global average pooling。

![23](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/23.PNG)

* Learning Strategy

如上图图4，在meta-training阶段包含大量base class的标注数据，对feature extractor、dense relation distillation module和context-aware feature aggregation module进行联合训练。

在meta fine-tune阶段，在base和novel class上进行训练。在k-shot设置下，novel class和base class都设成只有k个样本。训练过程和meta training相同，但是模型收敛迭代的次数较少。

### 2.3 Experiment

PASCAL VOC数据集：

![24](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/24.PNG)

COCO数据集：

![25](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/25.PNG)

消融实验：

![26](https://raw.githubusercontent.com/nku-shengzheliu/Markdown4Zhihu/tree/master/DataFSOD_7.6/26.PNG)

## 3. Hallucination Improves Few-Shot Object Detection

> paper: https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Hallucination_Improves_Few-Shot_Object_Detection_CVPR_2021_paper.html
>
> code: https://github.com/pppplin/HallucFsDet







## 4. Generalized Few-Shot Object Detection Without Forgetting

> paper: https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.html
>
> code: None





## 5. UniT: Unified Knowledge Transfer for Any-Shot Object Detection and Segmentation

> paper:https://openaccess.thecvf.com/content/CVPR2021/html/Khandelwal_UniT_Unified_Knowledge_Transfer_for_Any-Shot_Object_Detection_and_Segmentation_CVPR_2021_paper.html
>
> code: https://github.com/ubc-vision/UniT































