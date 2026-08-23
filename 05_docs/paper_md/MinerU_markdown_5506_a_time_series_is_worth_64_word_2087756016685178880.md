# A TIME SERIES IS WORTH 64 WORDS: LONG-TERM FORECASTING WITH TRANSFORMERS

Yuqi Nie<sup>1∗</sup>, Nam H. Nguyen<sup>2</sup> <sup>∗</sup>, Phanwadee Sinthong<sup>2</sup>, Jayant Kalagnanam<sup>2</sup> 

<sup>1</sup>Princeton University <sup>2</sup>IBM Research 

ynie@princeton.edu, nnguyen@us.ibm.com, Gift.Sinthong@ibm.com, jayant@us.ibm.com 

## ABSTRACT

We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pretraining tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy. 

## 1 INTRODUCTION

Forecasting is one of the most important tasks in time series analysis. With the rapid growth of deep learning models, the number of research works has increased significantly on this topic (Bryan & Stefan, 2021; Torres et al., 2021; Lara-Ben´ıtez et al., 2021). Deep models have shown excellent performance not only on forecasting tasks, but also on representation learning where abstract representation can be extracted and transferred to various downstream tasks such as classification and anomaly detection to attain state-of-the-art performance. 

Among deep learning models, Transformer has achieved great success on various application fields such as natural language processing (NLP) (Kalyan et al., 2021), computer vision (CV) (Khan et al., 2021), speech (Karita et al., 2019), and more recently time series (Wen et al., 2022), benefiting from its attention mechanism which can automatically learn the connections between elements in a sequence, thus becomes ideal for sequential modeling tasks. Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021), and FEDformer (Zhou et al., 2022) are among the best variants of the Transformer model successfully applying to time series data. Unfortunately, regardless of the complicated design of Transformer-based models, it is shown in the recent paper (Zeng et al., 2022) that a very simple linear model can outperform all of the previous models on a variety of common benchmarks and it challenges the usefulness of Transformer for time series forecasting. In this paper, we attempt to answer this question by proposing a channel-independence patch time series Transformer (PatchTST) model that contains two key designs: 

• Patching. Time series forecasting aims to understand the correlation between data in each different time steps. However, a single time step does not have semantic meaning like a word in a sentence, thus extracting local semantic information is essential in analyzing their connections. Most of the previous works only use point-wise input tokens, or just a handcrafted information from series. In contrast, we enhance the locality and capture comprehensive semantic information that is not available in point-level by aggregating time steps into subseries-level patches. 

<table><tr><td>Models</td><td>L</td><td>N</td><td>patch</td><td>method</td><td>MSE</td></tr><tr><td rowspan="5">Channel-independent PatchTST</td><td>96</td><td>96</td><td></td><td rowspan="4">down-sampled</td><td>0.518</td></tr><tr><td>380</td><td>96</td><td></td><td>0.447</td></tr><tr><td>336</td><td>336</td><td></td><td>0.397</td></tr><tr><td>336</td><td>42</td><td>√</td><td>0.367</td></tr><tr><td>336</td><td>42</td><td>√</td><td>self-supervised</td><td>0.349</td></tr><tr><td rowspan="2">Channel-mixing FEDFormer DLinear</td><td>336</td><td>336</td><td></td><td></td><td>0.597</td></tr><tr><td>336</td><td>336</td><td></td><td></td><td>0.410</td></tr></table>

<table><tr><td colspan="4">Running time (s) with L = 336</td></tr><tr><td>Dataset</td><td>w. patch</td><td>w.o. patch</td><td>Gain</td></tr><tr><td>Traffic</td><td>464</td><td>10040</td><td>x 22</td></tr><tr><td>Electricity</td><td>300</td><td>5730</td><td>x 19</td></tr><tr><td>Weather</td><td>156</td><td>680</td><td>x 4</td></tr></table>


Table 1: A case study of multivariate time series forecasting on Traffic dataset. The prediction horizon is 96. Results with different look-back window L and number of input tokens N are reported. The best result is in bold and the second best is underlined. Down-sampled means sampling every 4 step and adding the last value. All the results are from supervised training except the best result which uses self-supervised learning.


• Channel-independence. A multivariate time series is a multi-channel signal, and each Transformer input token can be represented by data from either a single channel or multiple channels. Depending on the design of input tokens, different variants of the Transformer architecture have been proposed. Channel-mixing refers to the latter case where the input token takes the vector of all time series features and projects it to the embedding space to mix information. On the other hand, channel-independence means that each input token only contains information from a single channel. This was proven to work well with CNN (Zheng et al., 2014) and linear models (Zeng et al., 2022), but hasn’t been applied to Transformer-based models yet. 

We offer a snapshot of our key results in Table 1 by doing a case study on Traffic dataset, which consists of 862 time series. Our model has several advantages: 

1. Reduction on time and space complexity: The original Transformer has $O ( N ^ { 2 } )$ complexity on both time and space, where N is the number of input tokens. Without pre-processing, N will have the same value as input sequence length L, which becomes a primary bottleneck of computation time and memory in practice. By applying patching, we can reduce N by a factor of the stride: $N \approx L / S$ , thus reducing the complexity quadratically. Table 1 illustrates the usefulness of patching. By setting patch length $\bar { P = 1 6 }$ and stride $S = 8$ with $L = 3 3 6$ the training time is significantly reduced as much as 22 time on large datasets. 

2. Capability of learning from longer look-back window: Table 1 shows that by increasing lookback window L from 96 to 336, MSE can be reduced from 0.518 to 0.397. However, simply extending L comes at the cost of larger memory and computational usage. Since time series often carries heavily temporal redundant information, some previous work tried to ignore parts of data points by using downsampling or a carefully designed sparse connection of attention (Li et al., 2019) and the model still yields sufficient information to forecast well. We study the case when $L = 3 8 0$ and the time series is sampled every 4 steps with the last point added to sequence, resulting in the number of input tokens being $N = 9 6$ . The model achieves better MSE score (0.447) than using the data sequence containing the most recent 96 time steps (0.518), indicating that longer look-back window conveys more important information even with the same number of input tokens. This leads us to think of a question: is there a way to avoid throwing values while maintaining a long look-back window? Patching is a good answer to it. It can group local time steps that may contain similar values while at the same time enables the model to reduce the input token length for computational benefit. As evident in Table 1, MSE score is further reduced from 0.397 to 0.367 with patching when $L = 3 3 6$ 

3. Capability of representation learning: With the emergence of powerful self-supervised learning techniques, sophisticated models with multiple non-linear layers of abstraction are required to capture abstract representation of the data. Simple models like linear ones (Zeng et al., 2022) may not be preferred for that task due to its limited expressibility. With our PatchTST model, we not only confirm that Transformer is actually effective for time series forecasting, but also demonstrate the representation capability that can further enhance the forecasting performance. Our PatchTST has achieved the best MSE (0.349) in Table 1. 

We introduce our approach in more detail and conduct extensive experiments in the following sections to conclusively prove our claims. We not only demonstrate the model effectiveness with supervised forecasting results and ablation studies, but also achieves SOTA self-supervised representation learning and transfer learning performance. 

## 2 RELATED WORK

Patch in Transformer-based Models. Transformer (Vaswani et al., 2017) has demonstrated a significant potential on different data modalities. Among all applications, patching is an essential part when local semantic information is important. In NLP, BERT (Devlin et al., 2018) considers subword-based tokenization (Schuster & Nakajima, 2012) instead of performing character-based tokenization. In CV, Vision Transformer (ViT) (Dosovitskiy et al., 2021) is a milestone work that splits an image into 16×16 patches before feeding into the Transformer model. The following influential works such as BEiT (Bao et al., 2022) and masked autoencoders (He et al., 2021) are all using patches as input. Similarly, in speech researchers are using convolutions to extract information in sub-sequence levels from raw audio input (Baevski et al., 2020; Hsu et al., 2021). 

Transformer-based Long-term Time Series Forecasting. There is a large body of work that tries to apply Transformer models to forecast long-term time series in recent years. We here summarize some of them. LogTrans (Li et al., 2019) uses convolutional self-attention layers with LogSparse design to capture local information and reduce the space complexity. Informer (Zhou et al., 2021) proposes a ProbSparse self-attention with distilling techniques to extract the most important keys efficiently. Autoformer (Wu et al., 2021) borrows the ideas of decomposition and auto-correlation from traditional time series analysis methods. FEDformer (Zhou et al., 2022) uses Fourier enhanced structure to get a linear complexity. Pyraformer (Liu et al., 2022) applies pyramidal attention module with inter-scale and intra-scale connections which also get a linear complexity. 

Most of these models focus on designing novel mechanisms to reduce the complexity of original attention mechanism, thus achieving better performance on forecasting, especially when the prediction length is long. However, most of the models use point-wise attention, which ignores the importance of patches. LogTrans (Li et al., 2019) avoids a point-wise dot product between the key and query, but its value is still based on a single time step. Autoformer (Wu et al., 2021) uses autocorrelation to get patch level connections, but it is a handcrafted design which doesn’t include all the semantic information within a patch. Triformer (Cirstea et al., 2022) proposes patch attention, but the purpose is to reduce complexity by using a pseudo timestamp as the query within a patch, thus it neither treats a patch as a input unit, nor reveals the semantic importance behind it. 

Time Series Representation Learning. Besides supervised learning, self-supervised learning is also an important research topic since it has shown the potential to learn useful representations for downstream tasks. There are many non-Transformer-based models proposed in recent years to learn representations in time series (Franceschi et al., 2019; Tonekaboni et al., 2021; Yang & Hong, 2022; Yue et al., 2022). Meanwhile, Transformer is known to be an ideal candidate towards foundation models (Bommasani et al., 2021) and learning universal representations. However, although people have made attempts on Transformer-based models like time series Transformer (TST) (Zerveas et al., 2021) and TS-TCC (Eldele et al., 2021), the potential is still not fully realized yet. 

## 3 PROPOSED METHOD

## 3.1 MODEL STRUCTURE

We consider the following problem: given a collection of multivariate time series samples with lookback window $L : ( \pmb { x } _ { 1 } , . . . , \pmb { x } _ { L } )$ where each $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at time step t is a vector of dimension M, we would like to forecast T future values $( \pmb { x } _ { L + 1 } , . . . , \pmb { x } _ { L + T } )$ . Our PatchTST is illustrated in Figure 1 where the model makes use of the vanilla Transformer encoder as its core architecture. 

Forward Process. We denote a i-th univariate series of length L starting at time index 1 as $\pmb { x } _ { 1 : L } ^ { ( i ) } = ( x _ { 1 } ^ { ( i ) } , . . . , x _ { L } ^ { ( i ) } )$ where $i = 1 , . . . , M$ . The input $( \pmb { x } _ { 1 } , . . . , \pmb { x } _ { L } )$ is split to M univariate series $\pmb { x } ^ { ( i ) } \in \mathbb { R } ^ { 1 \times L }$ , where each of them is fed independently into the Transformer backbone according to our channel-independence setting. Then the Transformer backbone will provide prediction results $\pmb { \hat { x } } ^ { ( i ) } = ( \hat { x } _ { L + 1 } ^ { ( i ) } , . . . , \hat { x } _ { L + T } ^ { ( i ) } ) \in \mathbb { R } ^ { 1 \times \bar { T } }$ accordingly . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/89d82d4d458969fc05637e406ee359306837a93e134a90655b29ac20889cedbc.jpg)



(a) PatchTST Model Overview


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/0508a83f6fb5e2539d6378799ffb6838e47d84c515aad5639d842ea17ceccfb2.jpg)



(b) Transformer Backbone (Supervised)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/87f859fabd09809a0bd8432c17b046d6b8a8424721547fb95e5b25e4243c536d.jpg)



(c) Transformer Backbone (Self-supervised)



Figure 1: PatchTST architecture. (a) Multivariate time series data is divided into different channels. They share the same Transformer backbone, but the forward processes are independent. (b) Each channel univariate series is passed through instance normalization operator and segmented into patches. These patches are used as Transformer input tokens. (c) Masked self-supervised representation learning with PatchTST where patches are randomly selected and set to zero. The model will reconstruct the masked patches.


Patching. Each input univariate time series $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ is first divided into patches which can be either overlapped or non-overlapped. Denote the patch length as $P$ and the stride - the non overlapping region between two consecutive patches as $S _ { \ i }$ , then the patching process will generate the a sequence of patches $\pmb { x } _ { p } ^ { ( i ) } \in \mathbb { R } ^ { P \times N }$ where N is the number of patches, $\begin{array} { r } { N = \lfloor \frac { ( L - P ) } { S } \rfloor + 2 } \end{array}$ . Here, we pad S repeated numbers of the last value $x _ { L } ^ { ( i ) } \in \mathbb { R }$ to the end of the original sequence before patching. 

With the use of patches, the number of input tokens can reduce from L to approximately $L / S$ This implies the memory usage and computational complexity of the attention map are quadratically decreased by a factor of S. Thus constrained on the training time and GPU memory, patch design can allow the model to see the longer historical sequence, which can significantly improve the forecasting performance, as demonstrated in Table 1. 

Transformer Encoder. We use a vanilla Transformer encoder that maps the observed signals to the latent representations. The patches are mapped to the Transformer latent space of dimension D via a trainable linear projection $\dot { \mathbf { W } } _ { p } \in \mathbb { R } ^ { D \times P }$ , and a learnable additive position encoding $\mathbf { W } _ { \mathrm { p o s } } \in \mathbb { R } ^ { D \times N }$ is applied to monitor the temporal order of patches: $\begin{array} { r } { \pmb { x } _ { d } ^ { ( i ) } = \mathbf { W } _ { p } \pmb { x } _ { p } ^ { ( i ) } + \mathbf { W } _ { \mathrm { p o s } } , } \end{array}$ , where $\pmb { x } _ { d } ^ { ( i ) } \in \mathbb { R } ^ { D \times N }$ denote the input that will be fed into Transformer encoder in Figure 1. Then each head $\ddot { h } = 1 , . . . , H$ in multi-head attention will transform them into query matrices $Q _ { h } ^ { ( i ) } = ( \mathbf { \boldsymbol { x } } _ { d } ^ { ( i ) } ) ^ { T } \mathbf { W } _ { h } ^ { Q }$ , key matrices $K _ { h } ^ { ( i ) } = ( \mathbf { \boldsymbol { x } } _ { d } ^ { ( i ) } ) ^ { T } \mathbf { W } _ { h } ^ { K }$ and value matrices $V _ { h } ^ { ( i ) } = ( \mathbf { \boldsymbol { x } } _ { d } ^ { ( i ) } ) ^ { T } \mathbf { \mathbf { W } } _ { h } ^ { V }$ , where $\mathbf { W } _ { h } ^ { Q } , \mathbf { W } _ { h } ^ { K } \in \mathbb { R } ^ { D \times d _ { k } }$ and $\mathbf { W } _ { h } ^ { V } \in \mathbb { R } ^ { D \times D }$ . After that a scaled production is used for getting attention output $\mathbf { O } _ { h } ^ { ( i ) } \in \mathbb { R } ^ { D \times N } ;$ 

$$
(\mathbf {O} _ {h} ^ {(i)}) ^ {T} = \text { Attention } (Q _ {h} ^ {(i)}, K _ {h} ^ {(i)}, V _ {h} ^ {(i)}) = \text { Softmax } (\frac {Q _ {h} ^ {(i)} K _ {h} ^ {(i) T}}{\sqrt {d _ {k}}}) V _ {h} ^ {(i)}
$$

The multi-head attention block also includes BatchNorm <sup>1</sup> layers and a feed forward network with residual connections as shown in Figure 1. Afterwards it generates the representation denoted as $\pmb { z } ^ { ( i ) } \in \mathbb { R } ^ { D \times N }$ . Finally a flatten layer with linear head is used to obtain the prediction result $\hat { \pmb x } ^ { ( i ) } =$ $( \hat { x } _ { L + 1 } ^ { ( i ) } , . . . , \hat { x } _ { L + T } ^ { ( i ) } ) \in \bar { \mathbb { R } } ^ { 1 \times T }$ 

Loss Function. We choose to use the MSE loss to measure the discrepancy between the prediction and the ground truth. The loss in each channel is gathered and averaged over M time series to get the overall objective loss: $\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { \pmb { x } } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \hat { \pmb { x } } _ { L + 1 : L + T } ^ { ( i ) } - \pmb { x } _ { L + 1 : L + T } ^ { ( i ) } \| _ { 2 } ^ { 2 } } \end{array}$ 

Instance Normalization. This technique has recently been proposed to help mitigating the distribution shift effect between the training and testing data (Ulyanov et al., 2016; Kim et al., 2022). It simply normalizes each time series instance $\pmb { x } ^ { ( i ) }$ with zero mean and unit standard deviation. In essence, we normalize each $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ before patching and the mean and deviation are added back to the output prediction. 

## 3.2 REPRESENTATION LEARNING

Self-supervised representation learning has become a popular approach to extract high level abstract representation from unlabelled data. In this section, we apply PatchTST to obtain useful representation of the multivariate time series. We will show that the learnt representation can be effectively transferred to forecasting tasks. Among popular methods to learn representation via self-supervise pre-training, masked autoencoder has been applied successfully to NLP (Devlin et al., 2018) and CV (He et al., 2021) domains. This technique is conceptually simple: a portion of input sequence is intentionally removed at random and the model is trained to recover the missing contents. 

Masked encoder has been recently employed in time series and delivered notable performance on classification and regression tasks (Zerveas et al., 2021). The authors proposed to apply the multivariate time series to Transformer, where each input token is a vector x<sub>i</sub> consisting of time series values at time step i-th. Masking is placed randomly within each time series and across different series. However, there are two potential issues with this setting: First, masking is applied at the level of single time steps. The masked values at the current time step can be easily inferred by interpolating with the immediate proceeding or succeeding time values without high level understanding of the entire sequence, which deviates from our goal of learning important abstract representation of the whole signal. Zerveas et al. (2021) proposed complex randomization strategies to resolve the problem in which groups of time series with different sizes are randomly masked. 

Second, the design of the output layer for forecasting task can be troublesome. Given the representation vectors $\boldsymbol { z } _ { t } \in \mathbb { R } ^ { D }$ corresponding to all L time steps, mapping these vectors to the output containing M variables each with prediction horizon $T$ via a linear map requires a parameter matrix W of dimension $( L \cdot D ) \times ( M \cdot \bar { T } )$ . This matrix can be particularly oversized if either one or all of these four values are large. This may cause overfitting when the number of downstream training samples is scarce. 

Our proposed PatchTST can naturally overcome the aforementioned issues. As shown in Figure 1, we use the same Transformer encoder as the supervised settings. The prediction head is removed and a $. D \times P$ linear layer is attached. As opposed to supervised model where patches can be overlapped, we divide each input sequence into regular non-overlapping patches. It is for convenience to ensure observed patches do not contain information of the masked ones. We then select a subset of the patch indices uniformly at random and mask the patches according to these selected indices with zero values. The model is trained with MSE loss to reconstruct the masked patches. 

We emphasize that each time series will have its own latent representation that are cross-learned via a shared weight mechanism. This design can allow the pre-training data to contain different number of time series than the downstream data, which may not be feasible by other approaches. 

## 4 EXPERIMENTS

## 4.1 LONG-TERM TIME SERIES FORECASTING

Datasets. We evaluate the performance of our proposed PatchTST on 8 popular datasets, including Weather, Traffic, Electricity, ILI and 4 ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2). These datasets have been extensively utilized for benchmarking and publicly available on (Wu et al., 2021). The statistics of those datasets are summarized in Table 2. We would like to highlight several large datasets: Weather, Traffic, and Electricity. They have many more number of time series, thus the results would be more stable and less susceptible to overfitting than other smaller datasets. 

<table><tr><td>Datasets</td><td>Weather</td><td>Traffic</td><td>Electricity</td><td>ILI</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td></tr><tr><td>Features</td><td>21</td><td>862</td><td>321</td><td>7</td><td>7</td><td>7</td><td>7</td><td>7</td></tr><tr><td>Timesteps</td><td>52696</td><td>17544</td><td>26304</td><td>966</td><td>17420</td><td>17420</td><td>69680</td><td>69680</td></tr></table>


Table 2: Statistics of popular datasets for benchmark.


Baselines and Experimental Settings. We choose the SOTA Transformer-based models, includ ing FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), Informer (Zhou et al., 2021), Pyraformer (Liu et al., 2022), LogTrans (Li et al., 2019), and a recent non-Transformer-based model DLinear (Zeng et al., 2022) as our baselines. All of the models are following the same experimental setup with prediction length $T \in \{ 2 4 , 3 6 , 4 8 , 6 0 \}$ for ILI dataset and $T \in \{ 9 6 , 1 9 2 , 3 3 6$ , 720} for other datasets as in the original papers. We collect baseline results from Zeng et al. (2022) with the default look-back window $L = 9 6$ for Transformer-based models, and $L = 3 3 6$ for DLinear. But in order to avoid under-estimating the baselines, we also run FEDformer, Autoformer and Informer for six different look-back window $L \in \{ 2 4 , 4 8 , 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ , and always choose the best results to create strong baselines. More details about the baselines could be found in Appendix A.1.2. We calculate the MSE and MAE of multivariate time series forecasting as metrics. 

Model Variants. We propose two versions of PatchTST in Table 3. PatchTST/64 implies the number of input patches is 64, which uses the look-back window L = 512. PatchTST/42 means the number of input patches is 42, which has the default look-back window $L = 3 3 6$ . Both of them use patch length P = 16 and stride S = 8. Thus, we could use PatchTST/42 as a fair comparison to DLinear and other Transformer-based models, and PatchTST/64 to explore even better results on larger datasets. More experimental details are provided in Appendix A.1. 

Results. Table 3 shows the multivariate long-term forecasting results. Overall, our model outperform all baseline methods. Quantitatively, compared with the best results that Transformer-based models can offer, PatchTST/64 achieves an overall 21.0% reduction on MSE and 16.7% reduction on MAE, while PatchTST/42 attains an overall 20.2% reduction on MSE and 16.4% reduction on MAE. Compared with the DLinear model, PatchTST can still outperform it in general, especially on large datasets (Weather, Traffic, Electricity) and ILI dataset. We also experiment with univariate datasets where the results are provided in Appendix A.3. 

## 4.2 REPRESENTATION LEARNING

In this section, we conduct experiments with masked self-supervised learning where we set the patches to be non-overlapped. Otherwise stated, across all representation learning experiments the input sequence length is chosen to be 512 and patch size is set to 12, which results in 42 patches. We consider high masking ratio where 40% of the patches are masked with zero values. We first apply self-supervised pre-training on the datasets mentioned in Section 4.1 for 100 epochs. Once the pre-trained model on each dataset is available, we perform supervised training to evaluate the learned representation with two options: (a) linear probing and (b) end-to-end fine-tuning. With (a), we only train the model head for 20 epochs while freezing the rest of the network; With (b), we apply linear probing for 10 epochs to update the model head and then end-to-end fine-tuning the entire network for 20 epochs. It was proven that a two-step strategy with linear probing followed by fine-tuning can outperform only doing fine-tuning directly (Kumar et al., 2022). We select a few representative results on below, and a full benchmark can be found in Appendix A.5. 

<table><tr><td colspan="2">Models</td><td colspan="2">PatchTST/64</td><td colspan="2">PatchTST/42</td><td colspan="2">DLinear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Pyraformer</td><td colspan="2">LogTrans</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.149</td><td>0.198</td><td>0.152</td><td>0.199</td><td>0.176</td><td>0.237</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td><td>0.896</td><td>0.556</td><td>0.458</td><td>0.490</td></tr><tr><td>192</td><td>0.194</td><td>0.241</td><td>0.197</td><td>0.243</td><td>0.220</td><td>0.282</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td><td>0.622</td><td>0.624</td><td>0.658</td><td>0.589</td></tr><tr><td>336</td><td>0.245</td><td>0.282</td><td>0.249</td><td>0.283</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td><td>0.739</td><td>0.753</td><td>0.797</td><td>0.652</td></tr><tr><td>720</td><td>0.314</td><td>0.334</td><td>0.320</td><td>0.335</td><td>0.323</td><td>0.362</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td><td>1.004</td><td>0.934</td><td>0.869</td><td>0.675</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.360</td><td>0.249</td><td>0.367</td><td>0.251</td><td>0.410</td><td>0.282</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td><td>2.085</td><td>0.468</td><td>0.684</td><td>0.384</td></tr><tr><td>192</td><td>0.379</td><td>0.256</td><td>0.385</td><td>0.259</td><td>0.423</td><td>0.287</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td><td>0.867</td><td>0.467</td><td>0.685</td><td>0.390</td></tr><tr><td>336</td><td>0.392</td><td>0.264</td><td>0.398</td><td>0.265</td><td>0.436</td><td>0.296</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td><td>0.869</td><td>0.469</td><td>0.734</td><td>0.408</td></tr><tr><td>720</td><td>0.432</td><td>0.286</td><td>0.434</td><td>0.287</td><td>0.466</td><td>0.315</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td><td>0.881</td><td>0.473</td><td>0.717</td><td>0.396</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.129</td><td>0.222</td><td>0.130</td><td>0.222</td><td>0.140</td><td>0.237</td><td>0.186</td><td>0.302</td><td>0.196</td><td>0.313</td><td>0.304</td><td>0.393</td><td>0.386</td><td>0.449</td><td>0.258</td><td>0.357</td></tr><tr><td>192</td><td>0.147</td><td>0.240</td><td>0.148</td><td>0.240</td><td>0.153</td><td>0.249</td><td>0.197</td><td>0.311</td><td>0.211</td><td>0.324</td><td>0.327</td><td>0.417</td><td>0.386</td><td>0.443</td><td>0.266</td><td>0.368</td></tr><tr><td>336</td><td>0.163</td><td>0.259</td><td>0.167</td><td>0.261</td><td>0.169</td><td>0.267</td><td>0.213</td><td>0.328</td><td>0.214</td><td>0.327</td><td>0.333</td><td>0.422</td><td>0.378</td><td>0.443</td><td>0.280</td><td>0.380</td></tr><tr><td>720</td><td>0.197</td><td>0.290</td><td>0.202</td><td>0.291</td><td>0.203</td><td>0.301</td><td>0.233</td><td>0.344</td><td>0.236</td><td>0.342</td><td>0.351</td><td>0.427</td><td>0.376</td><td>0.445</td><td>0.283</td><td>0.376</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.319</td><td>0.754</td><td>1.522</td><td>0.814</td><td>2.215</td><td>1.081</td><td>2.624</td><td>1.095</td><td>2.906</td><td>1.182</td><td>4.657</td><td>1.449</td><td>1.420</td><td>2.012</td><td>4.480</td><td>1.444</td></tr><tr><td>36</td><td>1.579</td><td>0.870</td><td>1.430</td><td>0.834</td><td>1.963</td><td>0.963</td><td>2.516</td><td>1.021</td><td>2.585</td><td>1.038</td><td>4.650</td><td>1.463</td><td>7.394</td><td>2.031</td><td>4.799</td><td>1.467</td></tr><tr><td>48</td><td>1.553</td><td>0.815</td><td>1.673</td><td>0.854</td><td>2.130</td><td>1.024</td><td>2.505</td><td>1.041</td><td>3.024</td><td>1.145</td><td>5.004</td><td>1.542</td><td>7.551</td><td>2.057</td><td>4.800</td><td>1.468</td></tr><tr><td>60</td><td>1.470</td><td>0.788</td><td>1.529</td><td>0.862</td><td>2.368</td><td>1.096</td><td>2.742</td><td>1.122</td><td>2.761</td><td>1.114</td><td>5.071</td><td>1.543</td><td>7.662</td><td>2.100</td><td>5.278</td><td>1.560</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.370</td><td>0.400</td><td>0.375</td><td>0.399</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.415</td><td>0.435</td><td>0.446</td><td>0.941</td><td>0.769</td><td>0.664</td><td>0.612</td><td>0.878</td><td>0.740</td></tr><tr><td>192</td><td>0.413</td><td>0.429</td><td>0.414</td><td>0.421</td><td>0.405</td><td>0.416</td><td>0.423</td><td>0.446</td><td>0.456</td><td>0.457</td><td>1.007</td><td>0.786</td><td>0.790</td><td>0.681</td><td>1.037</td><td>0.824</td></tr><tr><td>336</td><td>0.422</td><td>0.440</td><td>0.431</td><td>0.436</td><td>0.439</td><td>0.443</td><td>0.444</td><td>0.462</td><td>0.486</td><td>0.487</td><td>1.038</td><td>0.784</td><td>0.891</td><td>0.738</td><td>1.238</td><td>0.932</td></tr><tr><td>720</td><td>0.447</td><td>0.468</td><td>0.449</td><td>0.466</td><td>0.472</td><td>0.490</td><td>0.469</td><td>0.492</td><td>0.515</td><td>0.517</td><td>1.144</td><td>0.857</td><td>0.963</td><td>0.782</td><td>1.135</td><td>0.852</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.274</td><td>0.337</td><td>0.274</td><td>0.336</td><td>0.289</td><td>0.353</td><td>0.332</td><td>0.374</td><td>0.332</td><td>0.368</td><td>1.549</td><td>0.952</td><td>0.645</td><td>0.597</td><td>2.116</td><td>1.197</td></tr><tr><td>192</td><td>0.341</td><td>0.382</td><td>0.339</td><td>0.379</td><td>0.383</td><td>0.418</td><td>0.407</td><td>0.446</td><td>0.426</td><td>0.434</td><td>3.792</td><td>1.542</td><td>0.788</td><td>0.683</td><td>4.315</td><td>1.635</td></tr><tr><td>336</td><td>0.329</td><td>0.384</td><td>0.331</td><td>0.380</td><td>0.448</td><td>0.465</td><td>0.400</td><td>0.447</td><td>0.477</td><td>0.479</td><td>4.215</td><td>1.642</td><td>0.907</td><td>0.747</td><td>1.124</td><td>1.604</td></tr><tr><td>720</td><td>0.379</td><td>0.422</td><td>0.379</td><td>0.422</td><td>0.605</td><td>0.551</td><td>0.412</td><td>0.469</td><td>0.453</td><td>0.490</td><td>3.656</td><td>1.619</td><td>0.963</td><td>0.783</td><td>3.188</td><td>1.540</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.293</td><td>0.346</td><td>0.290</td><td>0.342</td><td>0.299</td><td>0.343</td><td>0.326</td><td>0.390</td><td>0.510</td><td>0.492</td><td>0.626</td><td>0.560</td><td>0.543</td><td>0.510</td><td>0.600</td><td>0.546</td></tr><tr><td>192</td><td>0.333</td><td>0.370</td><td>0.332</td><td>0.369</td><td>0.335</td><td>0.365</td><td>0.365</td><td>0.415</td><td>0.514</td><td>0.495</td><td>0.725</td><td>0.619</td><td>0.557</td><td>0.537</td><td>0.837</td><td>0.700</td></tr><tr><td>336</td><td>0.369</td><td>0.392</td><td>0.366</td><td>0.392</td><td>0.369</td><td>0.386</td><td>0.392</td><td>0.425</td><td>0.510</td><td>0.492</td><td>1.005</td><td>0.741</td><td>0.754</td><td>0.655</td><td>1.124</td><td>0.832</td></tr><tr><td>720</td><td>0.416</td><td>0.420</td><td>0.420</td><td>0.424</td><td>0.425</td><td>0.421</td><td>0.446</td><td>0.458</td><td>0.527</td><td>0.493</td><td>1.133</td><td>0.845</td><td>0.908</td><td>0.724</td><td>1.153</td><td>0.820</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.166</td><td>0.256</td><td>0.165</td><td>0.255</td><td>0.167</td><td>0.260</td><td>0.180</td><td>0.271</td><td>0.205</td><td>0.293</td><td>0.355</td><td>0.462</td><td>0.435</td><td>0.507</td><td>0.768</td><td>0.642</td></tr><tr><td>192</td><td>0.223</td><td>0.296</td><td>0.220</td><td>0.292</td><td>0.224</td><td>0.303</td><td>0.252</td><td>0.318</td><td>0.278</td><td>0.336</td><td>0.595</td><td>0.586</td><td>0.730</td><td>0.673</td><td>0.989</td><td>0.757</td></tr><tr><td>336</td><td>0.274</td><td>0.329</td><td>0.278</td><td>0.329</td><td>0.281</td><td>0.342</td><td>0.324</td><td>0.364</td><td>0.343</td><td>0.379</td><td>1.270</td><td>0.871</td><td>1.201</td><td>0.845</td><td>1.334</td><td>0.872</td></tr><tr><td>720</td><td>0.362</td><td>0.385</td><td>0.367</td><td>0.385</td><td>0.397</td><td>0.421</td><td>0.410</td><td>0.420</td><td>0.414</td><td>0.419</td><td>3.001</td><td>1.267</td><td>3.625</td><td>1.451</td><td>3.048</td><td>1.328</td></tr></table>


Table 3: Multivariate long-term forecasting results with supervised PatchTST. We use prediction lengths $T \in \{ 2 4 , 3 6 , 4 8 , 6 0 \}$ for ILI dataset and T ∈ {96, 192, 336, 720} for the others. The best results are in bold and the second best are underlined.


<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="6">PatchTST</td><td rowspan="2" colspan="2">DLinear</td><td rowspan="2" colspan="2">FEDformer</td><td rowspan="2" colspan="2">Autoformer</td><td rowspan="2" colspan="2">Informer</td></tr><tr><td colspan="2">Fine-tuning</td><td colspan="2">Lin. Prob.</td><td colspan="2">Sup.</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.144</td><td>0.193</td><td>0.158</td><td>0.209</td><td>0.152</td><td>0.199</td><td>0.176</td><td>0.237</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td></tr><tr><td>192</td><td>0.190</td><td>0.236</td><td>0.203</td><td>0.249</td><td>0.197</td><td>0.243</td><td>0.220</td><td>0.282</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td></tr><tr><td>336</td><td>0.244</td><td>0.280</td><td>0.251</td><td>0.285</td><td>0.249</td><td>0.283</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td></tr><tr><td>720</td><td>0.320</td><td>0.335</td><td>0.321</td><td>0.336</td><td>0.320</td><td>0.335</td><td>0.323</td><td>0.362</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.352</td><td>0.244</td><td>0.399</td><td>0.294</td><td>0.367</td><td>0.251</td><td>0.410</td><td>0.282</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td></tr><tr><td>192</td><td>0.371</td><td>0.253</td><td>0.412</td><td>0.298</td><td>0.385</td><td>0.259</td><td>0.423</td><td>0.287</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td></tr><tr><td>336</td><td>0.381</td><td>0.257</td><td>0.425</td><td>0.306</td><td>0.398</td><td>0.265</td><td>0.436</td><td>0.296</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td></tr><tr><td>720</td><td>0.425</td><td>0.282</td><td>0.460</td><td>0.323</td><td>0.434</td><td>0.287</td><td>0.466</td><td>0.315</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.126</td><td>0.221</td><td>0.138</td><td>0.237</td><td>0.130</td><td>0.222</td><td>0.140</td><td>0.237</td><td>0.186</td><td>0.302</td><td>0.196</td><td>0.313</td><td>0.304</td><td>0.393</td></tr><tr><td>192</td><td>0.145</td><td>0.238</td><td>0.156</td><td>0.252</td><td>0.148</td><td>0.240</td><td>0.153</td><td>0.249</td><td>0.197</td><td>0.311</td><td>0.211</td><td>0.324</td><td>0.327</td><td>0.417</td></tr><tr><td>336</td><td>0.164</td><td>0.256</td><td>0.170</td><td>0.265</td><td>0.167</td><td>0.261</td><td>0.169</td><td>0.267</td><td>0.213</td><td>0.328</td><td>0.214</td><td>0.327</td><td>0.333</td><td>0.422</td></tr><tr><td>720</td><td>0.193</td><td>0.291</td><td>0.208</td><td>0.297</td><td>0.202</td><td>0.291</td><td>0.203</td><td>0.301</td><td>0.233</td><td>0.344</td><td>0.236</td><td>0.342</td><td>0.351</td><td>0.427</td></tr></table>


Table 4: Multivariate long-term forecasting results with self-supervised PatchTST. We use prediction lengths T ∈ {96, 192, 336, 720}. The best results are in bold and the second best are underlined.


Comparison with Supervised Methods. Table 4 compares the performance of PatchTST (with fine-tuning, linear probing, and supervising from scratch) with other supervised method. As shown in the table, on large datasets our pre-training procedure contributes a clear improvement compared to supervised training from scratch. By just fine-tuning the model head (linear probing), the forecasting performance is already comparable with training the entire network from scratch and better than DLinear. The best results are observed with end-to-end fine-tuning. Self-supervised PatchTST significantly outperforms other Transformer-based models on all the datasets. 

Transfer Learning. We test the capability of transfering the pre-trained model to downstream tasks. In particular, we pre-train the model on Electricity dataset and fine-tune on other datasets. We observe from Table 5 that overall the fine-tuning MSE is lightly worse than pre-training and finetuning on the same dataset, which is reasonable. The fine-tuning performance is also worse than supervised training in some cases. However, the forecasting performance is still better than other models. Note that as opposed to supervised PatchTST where the entire model is trained for each prediction horizon, here we only retrain the linear head or the entire model for much fewer epochs, which results in significant computational time reduction. 

<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="6">PatchTST</td><td rowspan="2" colspan="2">DLinear</td><td rowspan="2" colspan="2">FEDformer</td><td rowspan="2" colspan="2">Autoformer</td><td rowspan="2" colspan="2">Informer</td></tr><tr><td colspan="2">Fine-tuning</td><td colspan="2">Lin. Prob.</td><td colspan="2">Sup.</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.145</td><td>0.195</td><td>0.163</td><td>0.216</td><td>0.152</td><td>0.199</td><td>0.176</td><td>0.237</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td></tr><tr><td>192</td><td>0.193</td><td>0.243</td><td>0.205</td><td>0.252</td><td>0.197</td><td>0.243</td><td>0.220</td><td>0.282</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td></tr><tr><td>336</td><td>0.244</td><td>0.280</td><td>0.253</td><td>0.289</td><td>0.249</td><td>0.283</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td></tr><tr><td>720</td><td>0.321</td><td>0.337</td><td>0.320</td><td>0.336</td><td>0.320</td><td>0.335</td><td>0.323</td><td>0.362</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.388</td><td>0.273</td><td>0.400</td><td>0.288</td><td>0.367</td><td>0.251</td><td>0.410</td><td>0.282</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td></tr><tr><td>192</td><td>0.400</td><td>0.277</td><td>0.412</td><td>0.293</td><td>0.385</td><td>0.259</td><td>0.423</td><td>0.287</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td></tr><tr><td>336</td><td>0.408</td><td>0.280</td><td>0.425</td><td>0.307</td><td>0.398</td><td>0.265</td><td>0.436</td><td>0.296</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td></tr><tr><td>720</td><td>0.447</td><td>0.310</td><td>0.457</td><td>0.317</td><td>0.434</td><td>0.287</td><td>0.466</td><td>0.315</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td></tr></table>


Table 5: Transfer learning task: PatchTST is pre-trained on Electricity dataset and the model is transferred to other datasets. The best results are in bold and the second best are underlined.


<table><tr><td rowspan="2" colspan="2">Models</td><td rowspan="2">IMP.</td><td colspan="4">PatchTST</td><td rowspan="2" colspan="2">BTSF</td><td rowspan="2" colspan="2">TS2Vec</td><td rowspan="2" colspan="2">TNC</td><td rowspan="2" colspan="2">TS-TCC</td></tr><tr><td colspan="2">Transferred</td><td colspan="2">Self-supervised</td></tr><tr><td colspan="2">Metrics</td><td>MSE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>24</td><td>42.3%</td><td>0.312</td><td>0.362</td><td>0.322</td><td>0.369</td><td>0.541</td><td>0.519</td><td>0.599</td><td>0.534</td><td>0.632</td><td>0.596</td><td>0.653</td><td>0.610</td></tr><tr><td>48</td><td>44.7%</td><td>0.339</td><td>0.378</td><td>0.354</td><td>0.385</td><td>0.613</td><td>0.524</td><td>0.629</td><td>0.555</td><td>0.705</td><td>0.688</td><td>0.720</td><td>0.693</td></tr><tr><td>168</td><td>34.5%</td><td>0.424</td><td>0.437</td><td>0.419</td><td>0.424</td><td>0.640</td><td>0.532</td><td>0.755</td><td>0.636</td><td>1.097</td><td>0.993</td><td>1.129</td><td>1.044</td></tr><tr><td>336</td><td>48.5%</td><td>0.472</td><td>0.472</td><td>0.445</td><td>0.446</td><td>0.864</td><td>0.689</td><td>0.907</td><td>0.717</td><td>1.454</td><td>0.919</td><td>1.492</td><td>1.076</td></tr><tr><td>720</td><td>48.8%</td><td>0.508</td><td>0.507</td><td>0.487</td><td>0.478</td><td>0.993</td><td>0.712</td><td>1.048</td><td>0.790</td><td>1.604</td><td>1.118</td><td>1.603</td><td>1.206</td></tr></table>


Table 6: Representation learning methods comparison. Column name transferred implies pretraining PatchTST on Traffic dataset and transferring the representation to ETTh1, while selfsupervised implies both pre-training and linear probing on ETTh1. The best and second best results are in bold and underlined. IMP. denotes the improvement on best results of PatchTST compared to that of baselines, which is in the range of 34.5% to 48.8% on various prediction lengths.


Comparison with Other Self-supervised Methods. We compare our self-supervised model with BTSF (Yang & Hong, 2022), TS2Vec (Yue et al., 2022), TNC (Tonekaboni et al., 2021), and TS-TCC (Eldele et al., 2021) which are among the state-of-the-art contrastive learning representation methods for time series <sup>2</sup>. We test the forecasting performance on ETTh1 dataset, where we only apply linear probing after the learned representation is obtained (only fine-tune the last linear layer) to make the comparison fair. Results from Table 6 strongly indicates the superior performance of PatchTST, both from pre-training on its own ETTh1 data (self-supervised columns) or pre-training on Traffic (transferred columns). 

## 4.3 ABLATION STUDY

Patching and Channel-independence. We study the effects of patching and channel-independence in Table 7. We include FEDformer as the SOTA benchmark for Transformer-based model. By comparing results with and without the design of patching / channel-independence accordingly, one can observe that both of them are important factors in improving the forecasting performance. The moti vation of patching is natural; furthermore this technique improves the running time and memory consumption as shown in Table 1 due to shorter Transformer sequence input. Channel-independence, on the other hand, may not be as intuitive as patching is in terms of the technical advantages. Therefore, we provide an in-depth analysis on the key factors that make channel-independence more preferable in Appendix A.7. More ablation study results are available in Appendix A.4. 

Varying Look-back Window. In principle, a longer look-back window increases the receptive field, which will potentially improves the forecasting performance. However, as argued in (Zeng et al., 2022), this phenomenon hasn’t been observed in most of the Transformer-based models. We also demonstrate in Figure 2 that in most cases, these Transformer-based baselines have not benefited from longer look-back window L, which indicates their ineffectiveness in capturing temporal information. In contrast, our PatchTST consistently reduces the MSE scores as the receptive field increases, which confirms our model’s capability to learn from longer look-back window. 

<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="8">PatchTST</td><td rowspan="2" colspan="2">FEDformer</td></tr><tr><td colspan="2">P+CI</td><td colspan="2">CI</td><td colspan="2">P</td><td colspan="2">Original</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.152</td><td>0.199</td><td>0.164</td><td>0.213</td><td>0.168</td><td>0.223</td><td>0.177</td><td>0.236</td><td>0.238</td><td>0.314</td></tr><tr><td>192</td><td>0.197</td><td>0.243</td><td>0.205</td><td>0.250</td><td>0.213</td><td>0.262</td><td>0.221</td><td>0.270</td><td>0.275</td><td>0.329</td></tr><tr><td>336</td><td>0.249</td><td>0.283</td><td>0.255</td><td>0.289</td><td>0.266</td><td>0.300</td><td>0.271</td><td>0.306</td><td>0.339</td><td>0.377</td></tr><tr><td>720</td><td>0.320</td><td>0.335</td><td>0.327</td><td>0.343</td><td>0.351</td><td>0.359</td><td>0.340</td><td>0.353</td><td>0.389</td><td>0.409</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.367</td><td>0.251</td><td>0.397</td><td>0.271</td><td>0.595</td><td>0.376</td><td>-</td><td>-</td><td>0.576</td><td>0.359</td></tr><tr><td>192</td><td>0.385</td><td>0.259</td><td>0.411</td><td>0.276</td><td>0.612</td><td>0.387</td><td>-</td><td>-</td><td>0.610</td><td>0.380</td></tr><tr><td>336</td><td>0.398</td><td>0.265</td><td>0.423</td><td>0.282</td><td>0.651</td><td>0.391</td><td>-</td><td>-</td><td>0.608</td><td>0.375</td></tr><tr><td>720</td><td>0.434</td><td>0.287</td><td>0.457</td><td>0.309</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.621</td><td>0.375</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.130</td><td>0.222</td><td>0.136</td><td>0.231</td><td>0.196</td><td>0.307</td><td>0.205</td><td>0.318</td><td>0.186</td><td>0.302</td></tr><tr><td>192</td><td>0.148</td><td>0.240</td><td>0.164</td><td>0.263</td><td>0.215</td><td>0.323</td><td>-</td><td>-</td><td>0.197</td><td>0.311</td></tr><tr><td>336</td><td>0.167</td><td>0.261</td><td>0.168</td><td>0.262</td><td>0.228</td><td>0.338</td><td>-</td><td>-</td><td>0.213</td><td>0.328</td></tr><tr><td>720</td><td>0.202</td><td>0.291</td><td>0.219</td><td>0.312</td><td>0.244</td><td>0.345</td><td>-</td><td>-</td><td>0.233</td><td>0.344</td></tr></table>


Table 7: Ablation study of patching and channel-independence in PatchTST. 4 cases are included: (a) both patching and channel-independence are included in model (P+CI); (b) only channelindependence (CI); (c) only patching (P); (d) neither of them is included (Original TST model). PatchTST means supervised PatchTST/42. ’-’ in table means the model runs out of GPU memory (NVIDIA A40 48GB) even with batch size 1. The best results are in bold.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/a7bbafb2c678f1bdc81dc0bb4f1fb973a15d10460637aca57b5a8af3b4fd823e.jpg)



Figure 2: Forecasting performance (MSE) with varying look-back windows on 3 large datasets: Electricity, Traffic, and Weather. The look-back windows are selected to be $L \ =$ 24, 48, 96, 192, 336, 720, and the prediction horizons are $T \ = \ 9 6 , 7 2 0 .$ We use supervised PatchTST/42 and other open-source Transformer-based baselines for this experiment.


## 5 CONCLUSION AND FUTURE WORK

This paper proposes an effective design of Transformer-based models for time series forecasting tasks by introducing two key components: patching and channel-independent structure. Compared to the previous works, it could capture local semantic information and benefit from longer look-back windows. We not only show that our model outperforms other baselines in supervised learning, but also prove its promising capability in self-supervised representation learning and transfer learning. 

Our model exhibits the potential to be the based model for future work of Transformer-based forecasting and be a building block for time series foundation models. Patching is simple but proven to be an effective operator that can be transferred easily to other models. Channel-independence, on the other hand, can be further exploited to incorporate the correlation between different channels. It would be an important future step to model the cross-channel dependencies properly. 

## REFERENCES



Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449–12460, 2020. 





Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018. 





Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=p-BhZSz59o4. 





Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 





G. E. P. Box and Gwilym M. Jenkins. Time series analysis, forecasting and control. 1970. 





Lim Bryan and Zohren Stefan. Time-series forecasting with deep learning: a survey. Phil. Trans. R. Soc. A, 2021. doi: 10.1098/rsta.2020.0209. 





Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, et al. Spectral temporal graph neural network for multivariate timeseries forecasting. Advances in neural information processing systems, 33:17766–17778, 2020. 





Weiqi Chen, Wenwei Wang, Bingqing Peng, Qingsong Wen, Tian Zhou, and Liang Sun. Learning to rotate: Quaternion transformer for complicated periodical time series forecasting. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 146–156, 2022. 





Yuzhou Chen, Ignacio Segovia, and Yulia R Gel. Z-gcnets: time zigzags at graph convolutional networks for time series forecasting. In International Conference on Machine Learning, pp. 1684–1694. PMLR, 2021. 





Razvan-Gabriel Cirstea, Chenjuan Guo, Bin Yang, Tung Kieu, Xuanyi Dong, and Shirui Pan. Triformer: Triangular, variable-specific attentions for long sequence multivariate time series forecasting. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 1994–2001, 7 2022. doi: 10.24963/ijcai.2022/277. URL https: //doi.org/10.24963/ijcai.2022/277. 





Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018. URL http://arxiv.org/abs/1810.04805. 





Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=YicbFdNTTy. 





Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, and Cuntai Guan. Time-series representation learning via temporal and contextual contrasting. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 2352–2359, 2021. 





Eugene F Fama. Efficient capital markets: A review of theory and empirical work. The journal of Finance, 25(2):383–417, 1970. 





Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jaggi. Unsupervised scalable representation learning for multivariate time series. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/file/ 53c6de78244e9f528eb3e1cda69699bb-Paper.pdf. 





Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross B. Girshick. Masked au-´ toencoders are scalable vision learners. CoRR, abs/2111.06377, 2021. URL https://arxiv. org/abs/2111.06377. 





Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 





S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 1997. 





Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021. 





Katikapalli Subramanyam Kalyan, Ajit Rajasekharan, and Sivanesan Sangeetha. Ammus: A survey of transformer-based pretrained models in natural language processing. arXiv preprint arXiv:2108.05542, 2021. 





Shigeki Karita, Nanxin Chen, Tomoki Hayashi, Takaaki Hori, Hirofumi Inaguma, Ziyan Jiang, Masao Someki, Nelson Enrique Yalta Soplin, Ryuichi Yamamoto, Xiaofei Wang, et al. A comparative study on transformer vs rnn in speech applications. In IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 449–456. IEEE, 2019. 





Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM Computing Surveys (CSUR), 2021. 





Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=cGDAkQo1C0p. 





Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Finetuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=UYneFzXSJWh. 





Pedro Lara-Ben´ıtez, Manuel Carranza-Garc´ıa, and Jose C Riquelme. An experimental review on´ deep learning architectures for time series forecasting. International Journal of Neural Systems, 31(03):2130001, 2021. 





Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/file/ 6775a0635c302542da2c32aa19d86be0-Paper.pdf. 





Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations, 2022. 





Barbara Rossi. Exchange rate predictability. Journal of Economic Literature, 51(4):1063–1119, December 2013. doi: 10.1257/jel.51.4.1063. URL https://www.aeaweb.org/articles? id=10.1257/jel.51.4.1063. 





David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal ofForecasting, 36(3):1181– 1191, 2020. 





Mike Schuster and Kaisuke Nakajima. Japanese and korean voice search. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5149–5152, 2012. doi: 10.1109/ICASSP.2012.6289079. 





Sana Tonekaboni, Danny Eytan, and Anna Goldenberg. Unsupervised representation learning for time series with temporal neighborhood coding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=8qDwejCuCN. 





Jose F Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Mart´ ´ınez-Alvarez, and Alicia Troncoso.<sup>´</sup> Deep learning for time series forecasting: a survey. Big Data, 9(1):3–21, 2021. 





Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://proceedings.neurips. cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 





Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: A survey. arXiv preprint arXiv:2202.07125, 2022. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, 2021. 





Ling Yang and Shenda Hong. Unsupervised time-series representation learning with iterative bilinear temporal-spectral fusion. In ICML, 2022. 





Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pp. 8980–8987, 2022. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? arXiv preprint arXiv:2205.13504, 2022. 





George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty, and Carsten Eickhoff. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 2114–2124, 2021. 





Yi Zheng, Qi Liu, Enhong Chen, Yong Ge, and J Leon Zhao. Time series classification using multichannels deep convolutional neural networks. In International conference on web-age information management, pp. 298–310. Springer, 2014. 





Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In The Thirty-Fifth AAAI Conference on Artificial Intelligence, volume 35, pp. 11106–11115, 2021. 





Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International Conference on Machine Learning, 2022. 



## A APPENDIX

## A.1 EXPERIMENTAL DETAILS

## A.1.1 DATASETS

We use 8 popular multivariate datasets provided in (Wu et al., 2021) for forecasting and representation learning. Weather<sup>3</sup> dataset collects 21 meteorological indicators in Germany, such as humidity and air temperature. $T r a f f i c ^ { 4 }$ dataset records the road occupancy rates from different sensors on San Francisco freeways. Electricity<sup>5</sup> is a dataset that describes 321 customers’ hourly electricity consumption. $I L I ^ { 6 }$ dataset collects the number of patients and influenza-like illness ratio in a weekly frequency. $E T T ^ { 7 }$ (Electricity Transformer Temperature) datasets are collected from two different electric transformers labeled with 1 and 2, and each of them contains 2 different resolutions (15 minutes and 1 hour) denoted with m and h. Thus, in total we have 4 ETT datasets: ETTm1, ETTm2, ETTh1, and ETTh2. 

There is an additional $E x c h a n g e - r a t e ^ { 8 }$ dataset mentioned in the original paper, which is the daily exchange-rate of eight different countries. However, financial datasets generally have different properties compared to time series datasets in other fields, for example the predictability. It is well known that if a market is efficient, the best prediction for $x _ { t }$ will be just $x _ { t - 1 }$ (Fama, 1970). Rossi (2013) argues that the toughest benchmark for exchange-rate forecasting is a random walk without drift. Also, Zeng et al. (2022) shows that by simply repeating the last value in the look-back window, the MSE loss on exchange-rate dataset can outperform or be comparable to the best results. Therefore, we are prudent in containing it into our benchmark. 

## A.1.2 DETAILS OF BASELINE SETTINGS

The default look-back windows for different baseline models could be different. For Transformerbased models, the default look-back window is $L = 9 6 ;$ and for DLinear, the default look-back window is $L = 3 3 6$ . The reason of this difference is that Transformer-based baselines are easy to overfit when look-back window is long while DLinear tend to underfit. However, this can possibly lead to an under-estimation of the baselines. To address this issue, we re-run FEDformer, Autoformer and Informer by ourselves for six different look-back window $L \in \{ 2 4 , 4 8 , 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . And for each forecasting task (aka each different prediction length on each dataset), we choose the best one from those six results. Thus it could be a strong baseline. 

The ILI dataset is much smaller than the other datasets, so a different set of parameters is applied (the default look-back windows of Transformer-based models and DLinear are $L = 3 6$ and $L =$ 104 respectively; we run FEDformer, Autoformer and Informer for six different look-back window $L \in \{ 2 \bar { 4 } , 3 6 , 4 \bar { 8 } , 6 0 , 1 0 4 , 1 4 4 \}$ and choose the best results). 

## A.1.3 BASELINES FROM TRADITIONAL MODELS

Time series has been an ancient field of study, with many traditional models developed, for example the famous ARIMA model (Box & Jenkins, 1970). With the bloom of deep learning community, many new models were proposed for sequence modeling and time series forecasting before Transformer appears, such as LSTM (Hochreiter & Schmidhuber, 1997), TCN (Bai et al., 2018) and DeepAR (Salinas et al., 2020). However, they are demonstrated to be not as effective as Transformerbased models in long-term forecasting tasks (Zhou et al., 2021; Wu et al., 2021), thus we don’t include them in our baselines. 

## A.1.4 MODEL PARAMETERS

By default, PatchTST contains 3 encoder layers with head number $H = 1 6$ and dimension of latent space $D \ = \ 1 2 8 .$ The feed forward network in Transformer encoder block consists of 2 linear layers with GELU (Hendrycks & Gimpel, 2016) activation function: one projecting the hidden representation $D = 1 2 8$ to a new dimension $F = 2 5 6 .$ , and another layer that project it back to $\bar { D \ } = 1 2 8$ . For very small datasets (ILI, ETTh1, ETTh2), a reduced size of parameters is used $( H = 4 , D = 1 6$ and $F = 1 2 8 )$ to mitigate the possible overfitting. Dropout with probability 0.2 is applied in the encoders for all experiments. The code will be publicly available. 

## A.1.5 IMPLEMENTATION DETAILS

Although PatchTST processes channels in parallel which has to make multiple copies of the Transformer’s weights, the computation can be implemented efficiently and does not require any special operator. The batch of samples of x $\in \mathbb { R } ^ { M \times L }$ with size $B \times M \stackrel { \cdot } { \times } L$ is passed through the patching operator to generate a 4D tensor of size $B \times M \times P \times N$ which represents a batch of $\pmb { x } _ { p } ^ { ( i ) } \in \mathbb { R } ^ { P \times N }$ in M series. By reshaping the tensor to form a 3D one of size $( B \cdot { \bf \bar { \it M } } ) \times P \times N$ , this batch can be consumed by any standard Transformer implementation. 

We further argue that our proposed PatchTST contains additional benefits: The components in Transformer backbone module shown in Figure 1 can differ across different input series, for instance the embedding layers and head layers. Note that if the embedding layers are designed differently for each group of time series, the reshaping step will be applied after embedding. Besides, the number of variables in the multivariate time series during the training may not need to match the number of series for testing. This is especially beneficial for self-supervised pre-training where the pre-training data can have different number of variables from the fine-tuning data. 

## A.2 VISUALIZATION

We visualize the long-term forecasting results of supervised PatchTST/42 and other baselines in Figure 3. Here, we predict 192 steps ahead on Weather and Eletricity datasets and 60 steps ahead on ILI dataset. PatchTST provides the best forecasting both in terms of scale and bias. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/44a3766c7d9ab11d0006c302314cf3632ad4269cef4e7fb629681c793b733956.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/3d7628e848518561d0e94cd9a99e97566eb9899f1722d425556d373dd5676fab.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/de9e809e87a39025193bc8ce9daa7e814cfd4103fff97ecdab71f216879267f8.jpg)



Figure 3: Visualization of 192-step forecasting on {Weather, Traffic} datasets with the look-back window $L = 3 3 6$ and 60-step forecasting on ILI dataset with $L = 1 0 4$ . PatchTST (in red) can capture the trend and closest to the ground truth (in blue).


## A.3 UNIVARIATE FORECASTING

Table 8 summaries the results of univariate forecasting on ETT datasets. There is a target feature ”oil temperature” within those datasets, which is the univariate series that we are trying to forecast. We cite the baseline results from (Zeng et al., 2022). 

## A.4 MORE RESULTS ON ABLATION STUDY

## A.4.1 VARYING PATCH LENGTH

This experiment studies the effect of patch lengths to the forecasting performance. We fix the lookback window to be 336 and vary the patch lengths $P = [ 4 , 8 , 1 6 , 2 4 , \hat { 3 2 } , 4 0 ]$ . The stride length is set the same as patch length, meaning no overlapping between patches. The model is trained to predict 96 steps. One observation from Figure 4 is that MSE scores don’t vary significantly with different choices of P, which indicate the robustness of our model against the patch length hyperparameter. Overall, PatchTST benefits from increased patch length, not only in forecasting performance but also in the computational reduction. The ideal patch length may depend on the dataset, but P between {8, 16} seems to be general good numbers. 

<table><tr><td colspan="2">Models</td><td colspan="2">PatchTST/64</td><td colspan="2">PatchTST/42</td><td colspan="2">DLinear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">LogTrans</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.059</td><td>0.189</td><td>0.055</td><td>0.179</td><td>0.056</td><td>0.180</td><td>0.079</td><td>0.215</td><td>0.071</td><td>0.206</td><td>0.193</td><td>0.377</td><td>0.283</td><td>0.468</td></tr><tr><td>192</td><td>0.074</td><td>0.215</td><td>0.071</td><td>0.205</td><td>0.071</td><td>0.204</td><td>0.104</td><td>0.245</td><td>0.114</td><td>0.262</td><td>0.217</td><td>0.395</td><td>0.234</td><td>0.409</td></tr><tr><td>336</td><td>0.076</td><td>0.220</td><td>0.081</td><td>0.225</td><td>0.098</td><td>0.244</td><td>0.119</td><td>0.270</td><td>0.107</td><td>0.258</td><td>0.202</td><td>0.381</td><td>0.386</td><td>0.546</td></tr><tr><td>720</td><td>0.087</td><td>0.236</td><td>0.087</td><td>0.232</td><td>0.189</td><td>0.359</td><td>0.142</td><td>0.299</td><td>0.126</td><td>0.283</td><td>0.183</td><td>0.355</td><td>0.475</td><td>0.629</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.131</td><td>0.284</td><td>0.129</td><td>0.282</td><td>0.131</td><td>0.279</td><td>0.128</td><td>0.271</td><td>0.153</td><td>0.306</td><td>0.213</td><td>0.373</td><td>0.217</td><td>0.379</td></tr><tr><td>192</td><td>0.171</td><td>0.329</td><td>0.168</td><td>0.328</td><td>0.176</td><td>0.329</td><td>0.185</td><td>0.330</td><td>0.204</td><td>0.351</td><td>0.227</td><td>0.387</td><td>0.281</td><td>0.429</td></tr><tr><td>336</td><td>0.171</td><td>0.336</td><td>0.185</td><td>0.351</td><td>0.209</td><td>0.367</td><td>0.231</td><td>0.378</td><td>0.246</td><td>0.389</td><td>0.242</td><td>0.401</td><td>0.293</td><td>0.437</td></tr><tr><td>720</td><td>0.223</td><td>0.380</td><td>0.224</td><td>0.383</td><td>0.276</td><td>0.426</td><td>0.278</td><td>0.420</td><td>0.268</td><td>0.409</td><td>0.291</td><td>0.439</td><td>0.218</td><td>0.387</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.026</td><td>0.123</td><td>0.026</td><td>0.121</td><td>0.028</td><td>0.123</td><td>0.033</td><td>0.140</td><td>0.056</td><td>0.183</td><td>0.109</td><td>0.277</td><td>0.049</td><td>0.171</td></tr><tr><td>192</td><td>0.040</td><td>0.151</td><td>0.039</td><td>0.150</td><td>0.045</td><td>0.156</td><td>0.058</td><td>0.186</td><td>0.081</td><td>0.216</td><td>0.151</td><td>0.310</td><td>0.157</td><td>0.317</td></tr><tr><td>336</td><td>0.053</td><td>0.174</td><td>0.053</td><td>0.173</td><td>0.061</td><td>0.182</td><td>0.084</td><td>0.231</td><td>0.076</td><td>0.218</td><td>0.427</td><td>0.591</td><td>0.289</td><td>0.459</td></tr><tr><td>720</td><td>0.073</td><td>0.206</td><td>0.074</td><td>0.207</td><td>0.080</td><td>0.210</td><td>0.102</td><td>0.250</td><td>0.110</td><td>0.267</td><td>0.438</td><td>0.586</td><td>0.430</td><td>0.579</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.065</td><td>0.187</td><td>0.065</td><td>0.186</td><td>0.063</td><td>0.183</td><td>0.067</td><td>0.198</td><td>0.065</td><td>0.189</td><td>0.088</td><td>0.225</td><td>0.075</td><td>0.208</td></tr><tr><td>192</td><td>0.093</td><td>0.231</td><td>0.094</td><td>0.231</td><td>0.092</td><td>0.227</td><td>0.102</td><td>0.245</td><td>0.118</td><td>0.256</td><td>0.132</td><td>0.283</td><td>0.129</td><td>0.275</td></tr><tr><td>336</td><td>0.121</td><td>0.266</td><td>0.120</td><td>0.265</td><td>0.119</td><td>0.261</td><td>0.130</td><td>0.279</td><td>0.154</td><td>0.305</td><td>0.180</td><td>0.336</td><td>0.154</td><td>0.302</td></tr><tr><td>720</td><td>0.172</td><td>0.322</td><td>0.171</td><td>0.322</td><td>0.175</td><td>0.320</td><td>0.178</td><td>0.325</td><td>0.182</td><td>0.335</td><td>0.300</td><td>0.435</td><td>0.160</td><td>0.321</td></tr></table>


Table 8: Univariate long-term forecasting results with supervised PatchTST. ETT datasets are used with prediction lengths $T \in$ {96, 192, 336, 720}. The best results are in bold.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/9e31c9c5af89215ecd9780ee73c6c1849dfacf65d76b1446a5b3bb3c6ce5ebc4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/b9d56f3d6961e048cb23b282c2e9982f6c54105f97545c990477e15428ff9937.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/b71f43526d28241450b4dcf46a7b8e3088fb7d84b29881ea2cf758f3bda3dd72.jpg)



Figure 4: MSE scores with varying patch lengths $P = [ 2 , 4 , 8 , 1 2 , 1 6 , 2 4 , 3 2 , 4 0 ]$ where the lookback window is 336 and the prediction length is 96.


## A.4.2 VARYING LOOK-BACK WINDOW

Here we provide a full benchmark of quantitative results in Table 9 for varying look-back window in supervised PatchTST/42 regarding Figure 2 in the main text. Generally speaking, our model gains performance improvement with increasing look-back window, which show the effectiveness of our model in learning information from longer receptive field. 

## A.4.3 PATCHING AND CHANNEL-INDEPENDENCE

Implementation Details. For ablation study on patching and channel independence in section 4.3, we run different variants of PatchTST: 

• Both patching and channel-independence are included in model (P+CI): this is the full PatchTST model that we have proposed in paper. 

• Only channel-independence (CI): we simply set both patch length P and stride S to be 1 to avoid patching and only keep channel-independence. 

• Only patching (P): referring to the implementation in A.1.5, instead of reshaping the 4D tensor from $\mathbf { \bar { B } } \times \mathbf { M } \mathbf { \bar { \times } } P \times N \mathbf { \ t o \left( \right)} \bar { B \mathbf { \cdot } } M  \times P \times N$ , we reshape it to $B \times ( M \cdot P ) \times N$ for channel-mixing with patching. 

<table><tr><td colspan="2">L</td><td colspan="2">24(24)</td><td colspan="2">48(36)</td><td colspan="2">96(48)</td><td colspan="2">192(60)</td><td colspan="2">336(104)</td><td colspan="2">720(144)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.222</td><td>0.246</td><td>0.212</td><td>0.243</td><td>0.178</td><td>0.219</td><td>0.160</td><td>0.204</td><td>0.152</td><td>0.199</td><td>0.147</td><td>0.198</td></tr><tr><td>192</td><td>0.265</td><td>0.279</td><td>0.254</td><td>0.277</td><td>0.224</td><td>0.259</td><td>0.204</td><td>0.245</td><td>0.197</td><td>0.243</td><td>0.190</td><td>0.240</td></tr><tr><td>336</td><td>0.325</td><td>0.322</td><td>0.310</td><td>0.316</td><td>0.278</td><td>0.298</td><td>0.257</td><td>0.285</td><td>0.249</td><td>0.283</td><td>0.242</td><td>0.282</td></tr><tr><td>720</td><td>0.404</td><td>0.374</td><td>0.385</td><td>0.365</td><td>0.350</td><td>0.346</td><td>0.329</td><td>0.338</td><td>0.320</td><td>0.335</td><td>0.304</td><td>0.328</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.766</td><td>0.419</td><td>0.671</td><td>0.381</td><td>0.477</td><td>0.305</td><td>0.401</td><td>0.267</td><td>0.367</td><td>0.251</td><td>0.365</td><td>0.251</td></tr><tr><td>192</td><td>0.725</td><td>0.398</td><td>0.616</td><td>0.356</td><td>0.471</td><td>0.299</td><td>0.406</td><td>0.268</td><td>0.385</td><td>0.259</td><td>0.382</td><td>0.258</td></tr><tr><td>336</td><td>0.752</td><td>0.410</td><td>0.635</td><td>0.364</td><td>0.485</td><td>0.305</td><td>0.421</td><td>0.277</td><td>0.398</td><td>0.265</td><td>0.398</td><td>0.267</td></tr><tr><td>720</td><td>0.786</td><td>0.427</td><td>0.673</td><td>0.383</td><td>0.518</td><td>0.325</td><td>0.452</td><td>0.297</td><td>0.434</td><td>0.287</td><td>0.436</td><td>0.289</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.268</td><td>0.316</td><td>0.225</td><td>0.293</td><td>0.174</td><td>0.259</td><td>0.138</td><td>0.230</td><td>0.130</td><td>0.222</td><td>0.130</td><td>0.224</td></tr><tr><td>192</td><td>0.259</td><td>0.316</td><td>0.217</td><td>0.291</td><td>0.178</td><td>0.265</td><td>0.149</td><td>0.243</td><td>0.148</td><td>0.240</td><td>0.147</td><td>0.241</td></tr><tr><td>336</td><td>0.283</td><td>0.335</td><td>0.238</td><td>0.309</td><td>0.196</td><td>0.282</td><td>0.169</td><td>0.262</td><td>0.167</td><td>0.261</td><td>0.163</td><td>0.259</td></tr><tr><td>720</td><td>0.321</td><td>0.365</td><td>0.278</td><td>0.342</td><td>0.237</td><td>0.316</td><td>0.211</td><td>0.299</td><td>0.202</td><td>0.291</td><td>0.197</td><td>0.290</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>3.062</td><td>1.118</td><td>1.610</td><td>0.803</td><td>1.281</td><td>0.704</td><td>1.300</td><td>0.700</td><td>1.522</td><td>0.814</td><td>1.470</td><td>0.793</td></tr><tr><td>36</td><td>2.732</td><td>1.071</td><td>1.262</td><td>0.731</td><td>1.251</td><td>0.752</td><td>1.367</td><td>0.776</td><td>1.430</td><td>0.834</td><td>1.518</td><td>0.856</td></tr><tr><td>48</td><td>3.059</td><td>1.117</td><td>1.991</td><td>0.845</td><td>1.901</td><td>0.879</td><td>1.690</td><td>0.812</td><td>1.673</td><td>0.854</td><td>1.834</td><td>0.921</td></tr><tr><td>60</td><td>2.610</td><td>1.057</td><td>1.702</td><td>0.829</td><td>1.611</td><td>0.844</td><td>1.526</td><td>0.795</td><td>1.529</td><td>0.862</td><td>1.656</td><td>0.885</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.464</td><td>0.445</td><td>0.410</td><td>0.417</td><td>0.393</td><td>0.408</td><td>0.382</td><td>0.401</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.408</td></tr><tr><td>192</td><td>0.521</td><td>0.474</td><td>0.469</td><td>0.448</td><td>0.445</td><td>0.434</td><td>0.428</td><td>0.425</td><td>0.414</td><td>0.421</td><td>0.413</td><td>0.431</td></tr><tr><td>336</td><td>0.570</td><td>0.498</td><td>0.516</td><td>0.469</td><td>0.484</td><td>0.451</td><td>0.451</td><td>0.436</td><td>0.431</td><td>0.436</td><td>0.445</td><td>0.454</td></tr><tr><td>720</td><td>0.575</td><td>0.522</td><td>0.509</td><td>0.487</td><td>0.480</td><td>0.471</td><td>0.452</td><td>0.459</td><td>0.449</td><td>0.466</td><td>0.458</td><td>0.477</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.333</td><td>0.362</td><td>0.307</td><td>0.348</td><td>0.294</td><td>0.343</td><td>0.285</td><td>0.340</td><td>0.274</td><td>0.336</td><td>0.279</td><td>0.341</td></tr><tr><td>192</td><td>0.422</td><td>0.409</td><td>0.397</td><td>0.399</td><td>0.377</td><td>0.393</td><td>0.356</td><td>0.386</td><td>0.339</td><td>0.379</td><td>0.349</td><td>0.386</td></tr><tr><td>336</td><td>0.442</td><td>0.432</td><td>0.412</td><td>0.420</td><td>0.381</td><td>0.409</td><td>0.350</td><td>0.395</td><td>0.331</td><td>0.380</td><td>0.375</td><td>0.409</td></tr><tr><td>720</td><td>0.462</td><td>0.453</td><td>0.434</td><td>0.441</td><td>0.412</td><td>0.433</td><td>0.395</td><td>0.427</td><td>0.379</td><td>0.422</td><td>0.394</td><td>0.434</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.624</td><td>0.481</td><td>0.424</td><td>0.403</td><td>0.321</td><td>0.360</td><td>0.291</td><td>0.340</td><td>0.290</td><td>0.342</td><td>0.295</td><td>0.348</td></tr><tr><td>192</td><td>0.671</td><td>0.507</td><td>0.468</td><td>0.429</td><td>0.362</td><td>0.384</td><td>0.328</td><td>0.365</td><td>0.332</td><td>0.369</td><td>0.334</td><td>0.373</td></tr><tr><td>336</td><td>0.714</td><td>0.533</td><td>0.501</td><td>0.453</td><td>0.392</td><td>0.402</td><td>0.365</td><td>0.389</td><td>0.366</td><td>0.392</td><td>0.361</td><td>0.393</td></tr><tr><td>720</td><td>0.744</td><td>0.554</td><td>0.553</td><td>0.484</td><td>0.450</td><td>0.435</td><td>0.422</td><td>0.423</td><td>0.420</td><td>0.424</td><td>0.416</td><td>0.419</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.212</td><td>0.290</td><td>0.189</td><td>0.272</td><td>0.178</td><td>0.260</td><td>0.169</td><td>0.254</td><td>0.165</td><td>0.255</td><td>0.162</td><td>0.254</td></tr><tr><td>192</td><td>0.282</td><td>0.334</td><td>0.260</td><td>0.317</td><td>0.249</td><td>0.307</td><td>0.230</td><td>0.294</td><td>0.220</td><td>0.292</td><td>0.216</td><td>0.293</td></tr><tr><td>336</td><td>0.354</td><td>0.376</td><td>0.328</td><td>0.359</td><td>0.313</td><td>0.346</td><td>0.280</td><td>0.329</td><td>0.278</td><td>0.329</td><td>0.269</td><td>0.329</td></tr><tr><td>720</td><td>0.458</td><td>0.433</td><td>0.429</td><td>0.415</td><td>0.400</td><td>0.398</td><td>0.378</td><td>0.386</td><td>0.367</td><td>0.385</td><td>0.350</td><td>0.380</td></tr></table>


Table 9: Multivariate long-term forecasting results with varying look-back window L in supervised PatchTST/42.


• Neither patching nor channel-independence is included (Original), which is just the original TST model Zerveas et al. (2021). 

Note that the default number of maximum epochs for Electricity and Traffic datasets are reduced from 100 to 20 for ablation experiments due to the huge space and time complexity of the original time series Transformer and channel independent model (no patching). 

Full Benchmark of Ablation Study. The full benchmark is shown in Table 10, which is a completed version of Table 7 in the main text. We can observe that patching together with channelindependence achieves the best results from the table, especially on larger datasets (Weather, Traffic, and Electricity) where the models are less susceptible to overfitting and thus the results would be more convincing. As one can see the improvement is robust on both patching and channelindependence. It is interesting to see that the improvement on ILI dataset is significant as well. 

## A.4.4 INSTANCE NORMALIZATION

Normalization is a technology used in many time series model to improve the forecasting performance (Kim et al., 2022; Chen et al., 2022; Zeng et al., 2022). In this experiment, we perform analysis on the effect of the instance normalization in our model. We train two models PatchTST/64 and PatchTST/42 with and without using instance normalization and observe the evaluated scores. As indicated in Table 11, instance normalization improves the forecasting performance slightly on two models. However, even without instance normalization operator, PatchTST still outperforms other Transformer methods on most of the datasets. This is to confirm that the improvement mainly comes from patching and channel independence designs. 

## A.5 MORE RESULTS ON SELF-SUPERVISED REPRESENTATION LEARNING

## A.5.1 FULL BENCHMARK OF MULTIVARIATE FORECASTING

In this section we provide a full benchmark of multivariate forecasting results with self-supervised PatchTST in Table 12, which is an extended version of Table 4. 

<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="8">PatchTST</td><td rowspan="2" colspan="2">FEDformer</td></tr><tr><td colspan="2">P+CI</td><td colspan="2">CI</td><td colspan="2">P</td><td colspan="2">Original</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.152</td><td>0.199</td><td>0.164</td><td>0.213</td><td>0.168</td><td>0.223</td><td>0.177</td><td>0.236</td><td>0.238</td><td>0.314</td></tr><tr><td>192</td><td>0.197</td><td>0.243</td><td>0.205</td><td>0.250</td><td>0.213</td><td>0.262</td><td>0.221</td><td>0.270</td><td>0.275</td><td>0.329</td></tr><tr><td>336</td><td>0.249</td><td>0.283</td><td>0.255</td><td>0.289</td><td>0.266</td><td>0.300</td><td>0.271</td><td>0.306</td><td>0.339</td><td>0.377</td></tr><tr><td>720</td><td>0.320</td><td>0.335</td><td>0.327</td><td>0.343</td><td>0.351</td><td>0.359</td><td>0.340</td><td>0.353</td><td>0.389</td><td>0.409</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.367</td><td>0.251</td><td>0.397</td><td>0.271</td><td>0.595</td><td>0.376</td><td>-</td><td>-</td><td>0.576</td><td>0.359</td></tr><tr><td>192</td><td>0.385</td><td>0.259</td><td>0.411</td><td>0.276</td><td>0.612</td><td>0.387</td><td>-</td><td>-</td><td>0.610</td><td>0.380</td></tr><tr><td>336</td><td>0.398</td><td>0.265</td><td>0.423</td><td>0.282</td><td>0.651</td><td>0.391</td><td>-</td><td>-</td><td>0.608</td><td>0.375</td></tr><tr><td>720</td><td>0.434</td><td>0.287</td><td>0.457</td><td>0.309</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.621</td><td>0.375</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.130</td><td>0.222</td><td>0.136</td><td>0.231</td><td>0.196</td><td>0.307</td><td>0.205</td><td>0.318</td><td>0.186</td><td>0.302</td></tr><tr><td>192</td><td>0.148</td><td>0.240</td><td>0.164</td><td>0.263</td><td>0.215</td><td>0.323</td><td>-</td><td>-</td><td>0.197</td><td>0.311</td></tr><tr><td>336</td><td>0.167</td><td>0.261</td><td>0.168</td><td>0.262</td><td>0.228</td><td>0.338</td><td>-</td><td>-</td><td>0.213</td><td>0.328</td></tr><tr><td>720</td><td>0.202</td><td>0.291</td><td>0.219</td><td>0.312</td><td>0.244</td><td>0.345</td><td>-</td><td>-</td><td>0.233</td><td>0.344</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.522</td><td>0.814</td><td>2.111</td><td>1.048</td><td>2.157</td><td>0.964</td><td>2.737</td><td>1.081</td><td>2.624</td><td>1.095</td></tr><tr><td>36</td><td>1.430</td><td>0.834</td><td>2.000</td><td>1.002</td><td>2.564</td><td>1.058</td><td>2.126</td><td>0.935</td><td>2.516</td><td>1.021</td></tr><tr><td>48</td><td>1.673</td><td>0.854</td><td>2.167</td><td>1.029</td><td>2.348</td><td>1.022</td><td>2.178</td><td>0.971</td><td>2.505</td><td>1.041</td></tr><tr><td>60</td><td>1.529</td><td>0.862</td><td>2.075</td><td>1.021</td><td>2.486</td><td>1.065</td><td>2.354</td><td>1.026</td><td>2.742</td><td>1.122</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.375</td><td>0.399</td><td>0.365</td><td>0.395</td><td>0.416</td><td>0.438</td><td>0.455</td><td>0.459</td><td>0.376</td><td>0.415</td></tr><tr><td>192</td><td>0.414</td><td>0.421</td><td>0.403</td><td>0.415</td><td>0.459</td><td>0.464</td><td>0.503</td><td>0.486</td><td>0.423</td><td>0.446</td></tr><tr><td>336</td><td>0.431</td><td>0.436</td><td>0.430</td><td>0.433</td><td>0.484</td><td>0.480</td><td>0.514</td><td>0.503</td><td>0.444</td><td>0.462</td></tr><tr><td>720</td><td>0.449</td><td>0.466</td><td>0.449</td><td>0.454</td><td>0.500</td><td>0.494</td><td>0.531</td><td>0.520</td><td>0.469</td><td>0.492</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.274</td><td>0.336</td><td>0.277</td><td>0.337</td><td>0.334</td><td>0.388</td><td>0.348</td><td>0.394</td><td>0.332</td><td>0.374</td></tr><tr><td>192</td><td>0.339</td><td>0.379</td><td>0.343</td><td>0.384</td><td>0.381</td><td>0.418</td><td>0.395</td><td>0.424</td><td>0.407</td><td>0.446</td></tr><tr><td>336</td><td>0.331</td><td>0.380</td><td>0.333</td><td>0.383</td><td>0.361</td><td>0.414</td><td>0.369</td><td>0.419</td><td>0.400</td><td>0.447</td></tr><tr><td>720</td><td>0.379</td><td>0.422</td><td>0.379</td><td>0.420</td><td>0.423</td><td>0.448</td><td>0.433</td><td>0.458</td><td>0.412</td><td>0.469</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.290</td><td>0.342</td><td>0.300</td><td>0.354</td><td>0.326</td><td>0.368</td><td>0.324</td><td>0.370</td><td>0.326</td><td>0.390</td></tr><tr><td>192</td><td>0.332</td><td>0.369</td><td>0.333</td><td>0.374</td><td>0.391</td><td>0.405</td><td>0.373</td><td>0.398</td><td>0.365</td><td>0.415</td></tr><tr><td>336</td><td>0.366</td><td>0.392</td><td>0.369</td><td>0.397</td><td>0.427</td><td>0.425</td><td>0.415</td><td>0.421</td><td>0.392</td><td>0.425</td></tr><tr><td>720</td><td>0.420</td><td>0.424</td><td>0.413</td><td>0.423</td><td>0.481</td><td>0.457</td><td>0.480</td><td>0.459</td><td>0.446</td><td>0.458</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.165</td><td>0.255</td><td>0.166</td><td>0.259</td><td>0.195</td><td>0.274</td><td>0.208</td><td>0.289</td><td>0.180</td><td>0.271</td></tr><tr><td>192</td><td>0.220</td><td>0.292</td><td>0.223</td><td>0.295</td><td>0.259</td><td>0.314</td><td>0.265</td><td>0.328</td><td>0.252</td><td>0.318</td></tr><tr><td>336</td><td>0.278</td><td>0.329</td><td>0.279</td><td>0.330</td><td>0.297</td><td>0.345</td><td>0.323</td><td>0.365</td><td>0.324</td><td>0.364</td></tr><tr><td>720</td><td>0.367</td><td>0.385</td><td>0.370</td><td>0.387</td><td>0.400</td><td>0.404</td><td>0.469</td><td>0.444</td><td>0.410</td><td>0.420</td></tr></table>


Table 10: Ablation study of patching (P) and channel-independence (CI) in PatchTST/42. A full benchmark regarding Table 7. The best results are in bold. ’-’ in table means the model runs out of GPU memory (NVIDIA A40 48GB) even with batch size 1.


<table><tr><td colspan="2">Models</td><td colspan="2">PatchTST/64 (+in)</td><td colspan="2">PatchTST/64 (-in)</td><td colspan="2">PatchTST/42 (+in)</td><td colspan="2">PatchTST/42 (-in)</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.149</td><td>0.198</td><td>0.161</td><td>0.219</td><td>0.152</td><td>0.199</td><td>0.156</td><td>0.210</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td></tr><tr><td>192</td><td>0.194</td><td>0.241</td><td>0.201</td><td>0.254</td><td>0.197</td><td>0.243</td><td>0.199</td><td>0.250</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td></tr><tr><td>336</td><td>0.245</td><td>0.282</td><td>0.253</td><td>0.298</td><td>0.249</td><td>0.283</td><td>0.248</td><td>0.294</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td></tr><tr><td>720</td><td>0.314</td><td>0.334</td><td>0.323</td><td>0.357</td><td>0.320</td><td>0.335</td><td>0.313</td><td>0.342</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.360</td><td>0.249</td><td>0.413</td><td>0.295</td><td>0.367</td><td>0.251</td><td>0.425</td><td>0.299</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td></tr><tr><td>192</td><td>0.379</td><td>0.256</td><td>0.425</td><td>0.302</td><td>0.385</td><td>0.259</td><td>0.439</td><td>0.302</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td></tr><tr><td>336</td><td>0.392</td><td>0.264</td><td>0.435</td><td>0.307</td><td>0.398</td><td>0.265</td><td>0.456</td><td>0.316</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td></tr><tr><td>720</td><td>0.432</td><td>0.286</td><td>0.473</td><td>0.321</td><td>0.434</td><td>0.287</td><td>0.488</td><td>0.333</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.129</td><td>0.222</td><td>0.133</td><td>0.230</td><td>0.130</td><td>0.222</td><td>0.131</td><td>0.226</td><td>0.186</td><td>0.302</td><td>0.196</td><td>0.313</td><td>0.304</td><td>0.393</td></tr><tr><td>192</td><td>0.147</td><td>0.240</td><td>0.148</td><td>0.244</td><td>0.148</td><td>0.240</td><td>0.150</td><td>0.244</td><td>0.197</td><td>0.311</td><td>0.211</td><td>0.324</td><td>0.327</td><td>0.417</td></tr><tr><td>336</td><td>0.163</td><td>0.259</td><td>0.164</td><td>0.262</td><td>0.167</td><td>0.261</td><td>0.168</td><td>0.267</td><td>0.213</td><td>0.328</td><td>0.214</td><td>0.327</td><td>0.333</td><td>0.422</td></tr><tr><td>720</td><td>0.197</td><td>0.290</td><td>0.196</td><td>0.291</td><td>0.202</td><td>0.291</td><td>0.201</td><td>0.298</td><td>0.233</td><td>0.344</td><td>0.236</td><td>0.342</td><td>0.351</td><td>0.427</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.319</td><td>0.754</td><td>3.563</td><td>1.317</td><td>1.522</td><td>0.814</td><td>3.489</td><td>1.345</td><td>2.624</td><td>1.095</td><td>2.906</td><td>1.182</td><td>4.657</td><td>1.449</td></tr><tr><td>36</td><td>1.579</td><td>0.870</td><td>3.426</td><td>1.205</td><td>1.430</td><td>0.834</td><td>4.629</td><td>1.550</td><td>2.516</td><td>1.021</td><td>2.585</td><td>1.038</td><td>4.650</td><td>1.463</td></tr><tr><td>48</td><td>1.553</td><td>0.815</td><td>4.309</td><td>1.449</td><td>1.673</td><td>0.854</td><td>3.746</td><td>1.383</td><td>2.505</td><td>1.041</td><td>3.024</td><td>1.145</td><td>5.004</td><td>1.542</td></tr><tr><td>60</td><td>1.470</td><td>0.788</td><td>4.065</td><td>1.402</td><td>1.529</td><td>0.862</td><td>5.174</td><td>1.622</td><td>2.742</td><td>1.122</td><td>2.761</td><td>1.114</td><td>5.071</td><td>1.543</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.370</td><td>0.400</td><td>0.385</td><td>0.410</td><td>0.375</td><td>0.399</td><td>0.388</td><td>0.412</td><td>0.376</td><td>0.415</td><td>0.435</td><td>0.446</td><td>0.941</td><td>0.769</td></tr><tr><td>192</td><td>0.413</td><td>0.429</td><td>0.417</td><td>0.432</td><td>0.414</td><td>0.421</td><td>0.430</td><td>0.438</td><td>0.423</td><td>0.446</td><td>0.456</td><td>0.457</td><td>1.007</td><td>0.786</td></tr><tr><td>336</td><td>0.422</td><td>0.440</td><td>0.439</td><td>0.449</td><td>0.431</td><td>0.436</td><td>0.454</td><td>0.458</td><td>0.444</td><td>0.462</td><td>0.486</td><td>0.487</td><td>1.038</td><td>0.784</td></tr><tr><td>720</td><td>0.447</td><td>0.468</td><td>0.478</td><td>0.494</td><td>0.449</td><td>0.466</td><td>0.494</td><td>0.497</td><td>0.469</td><td>0.492</td><td>0.515</td><td>0.517</td><td>1.144</td><td>0.857</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.274</td><td>0.337</td><td>0.299</td><td>0.359</td><td>0.274</td><td>0.336</td><td>0.313</td><td>0.374</td><td>0.332</td><td>0.374</td><td>0.332</td><td>0.368</td><td>1.549</td><td>0.952</td></tr><tr><td>192</td><td>0.341</td><td>0.382</td><td>0.354</td><td>0.404</td><td>0.339</td><td>0.379</td><td>0.402</td><td>0.432</td><td>0.407</td><td>0.446</td><td>0.426</td><td>0.434</td><td>3.792</td><td>1.542</td></tr><tr><td>336</td><td>0.329</td><td>0.384</td><td>0.374</td><td>0.420</td><td>0.331</td><td>0.380</td><td>0.448</td><td>0.465</td><td>0.400</td><td>0.447</td><td>0.477</td><td>0.479</td><td>4.215</td><td>1.642</td></tr><tr><td>720</td><td>0.379</td><td>0.422</td><td>0.479</td><td>0.492</td><td>0.379</td><td>0.422</td><td>0.688</td><td>0.588</td><td>0.412</td><td>0.469</td><td>0.453</td><td>0.490</td><td>3.656</td><td>1.619</td></tr><tr><td rowspan="4">ETTml</td><td>96</td><td>0.293</td><td>0.346</td><td>0.308</td><td>0.358</td><td>0.290</td><td>0.342</td><td>0.308</td><td>0.358</td><td>0.326</td><td>0.390</td><td>0.510</td><td>0.492</td><td>0.626</td><td>0.560</td></tr><tr><td>192</td><td>0.333</td><td>0.370</td><td>0.335</td><td>0.375</td><td>0.332</td><td>0.369</td><td>0.356</td><td>0.390</td><td>0.365</td><td>0.415</td><td>0.514</td><td>0.495</td><td>0.725</td><td>0.619</td></tr><tr><td>336</td><td>0.369</td><td>0.392</td><td>0.362</td><td>0.392</td><td>0.366</td><td>0.392</td><td>0.389</td><td>0.411</td><td>0.392</td><td>0.425</td><td>0.510</td><td>0.492</td><td>1.005</td><td>0.741</td></tr><tr><td>720</td><td>0.416</td><td>0.420</td><td>0.432</td><td>0.429</td><td>0.420</td><td>0.424</td><td>0.430</td><td>0.439</td><td>0.446</td><td>0.458</td><td>0.527</td><td>0.493</td><td>1.133</td><td>0.845</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.166</td><td>0.256</td><td>0.172</td><td>0.258</td><td>0.165</td><td>0.255</td><td>0.167</td><td>0.257</td><td>0.180</td><td>0.271</td><td>0.205</td><td>0.293</td><td>0.355</td><td>0.462</td></tr><tr><td>192</td><td>0.223</td><td>0.296</td><td>0.245</td><td>0.306</td><td>0.220</td><td>0.292</td><td>0.226</td><td>0.303</td><td>0.252</td><td>0.318</td><td>0.278</td><td>0.336</td><td>0.595</td><td>0.586</td></tr><tr><td>336</td><td>0.274</td><td>0.329</td><td>0.306</td><td>0.346</td><td>0.278</td><td>0.329</td><td>0.301</td><td>0.348</td><td>0.324</td><td>0.364</td><td>0.343</td><td>0.379</td><td>1.270</td><td>0.871</td></tr><tr><td>720</td><td>0.362</td><td>0.385</td><td>0.391</td><td>0.404</td><td>0.367</td><td>0.385</td><td>0.392</td><td>0.407</td><td>0.410</td><td>0.420</td><td>0.414</td><td>0.419</td><td>3.001</td><td>1.267</td></tr></table>


Table 11: Multivariate long-term forecasting results of supervised PatchTST with instance normalization (+in) or without instance normalization (-in). The best results are in bold and the second best are underlined. Although the models perform slightly better with instance normalization, compared to other Transformer models, the proposed approach achieve significantly better forecasting on most of the datasets even without instance normalization.


## A.5.2 FULL BENCHMARK OF TRANSFER LEARNING

In this section we provide Table 13 which contains the results of pre-training on Electricity dataset then transferred to other 6 datasets. Except Traffic data, the number of time series employed in the pre-training is much larger than the number of series during fine-tuning. It is a full version with respect to Table 5 in the main text and more cogently proves the capability to do transfer learning using our PatchTST model. 

## A.6 ROBUSTNESS ANALYSIS

## A.6.1 RESULTS WITH DIFFERENT RANDOM SEEDS

The results reported in the main text and appendix above are run with the fixed random seed 2021. To examine the robustness of our results, we train the supervised PatchTST model with 5 different random seeds: 2019, 2020, 2021, 2022, 2023 and calculate the MSE and MAE scores with each selected seed. The mean and standard derivation of the results are reported in Table 14. It is clear that the variances are considerably small which indicates the robustness against choice of random seeds of our model. 

We also validate the self-supervised PatchTST model on different runs. We pre-train the model once and fine-tune the model 5 times with different random batch selections. The mean and standard derivation across different runs are also provided in Table 14. We also observe that the variance is insignificant especially on large datasets while higher variance can be seen on smaller datasets. 

<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="6">PatchTST</td><td rowspan="2" colspan="2">DLinear</td><td rowspan="2" colspan="2">FEDformer</td><td rowspan="2" colspan="2">Autoformer</td><td rowspan="2" colspan="2">Informer</td></tr><tr><td colspan="2">Fine-tuning</td><td colspan="2">Lin. Prob.</td><td colspan="2">Sup.</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.144</td><td>0.193</td><td>0.158</td><td>0.209</td><td>0.152</td><td>0.199</td><td>0.176</td><td>0.237</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td></tr><tr><td>192</td><td>0.190</td><td>0.236</td><td>0.203</td><td>0.249</td><td>0.197</td><td>0.243</td><td>0.220</td><td>0.282</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td></tr><tr><td>336</td><td>0.244</td><td>0.280</td><td>0.251</td><td>0.285</td><td>0.249</td><td>0.283</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td></tr><tr><td>720</td><td>0.320</td><td>0.335</td><td>0.321</td><td>0.336</td><td>0.320</td><td>0.335</td><td>0.323</td><td>0.362</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.352</td><td>0.244</td><td>0.399</td><td>0.294</td><td>0.367</td><td>0.251</td><td>0.410</td><td>0.282</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td></tr><tr><td>192</td><td>0.371</td><td>0.253</td><td>0.412</td><td>0.298</td><td>0.385</td><td>0.259</td><td>0.423</td><td>0.287</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td></tr><tr><td>336</td><td>0.381</td><td>0.257</td><td>0.425</td><td>0.306</td><td>0.398</td><td>0.265</td><td>0.436</td><td>0.296</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td></tr><tr><td>720</td><td>0.425</td><td>0.282</td><td>0.460</td><td>0.323</td><td>0.434</td><td>0.287</td><td>0.466</td><td>0.315</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.126</td><td>0.221</td><td>0.138</td><td>0.237</td><td>0.130</td><td>0.222</td><td>0.140</td><td>0.237</td><td>0.186</td><td>0.302</td><td>0.196</td><td>0.313</td><td>0.304</td><td>0.393</td></tr><tr><td>192</td><td>0.145</td><td>0.238</td><td>0.156</td><td>0.252</td><td>0.148</td><td>0.240</td><td>0.153</td><td>0.249</td><td>0.197</td><td>0.311</td><td>0.211</td><td>0.324</td><td>0.327</td><td>0.417</td></tr><tr><td>336</td><td>0.164</td><td>0.256</td><td>0.170</td><td>0.265</td><td>0.167</td><td>0.261</td><td>0.169</td><td>0.267</td><td>0.213</td><td>0.328</td><td>0.214</td><td>0.327</td><td>0.333</td><td>0.422</td></tr><tr><td>720</td><td>0.193</td><td>0.291</td><td>0.208</td><td>0.297</td><td>0.202</td><td>0.291</td><td>0.203</td><td>0.301</td><td>0.233</td><td>0.344</td><td>0.236</td><td>0.342</td><td>0.351</td><td>0.427</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.366</td><td>0.397</td><td>0.371</td><td>0.400</td><td>0.375</td><td>0.399</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.415</td><td>0.435</td><td>0.446</td><td>0.941</td><td>0.769</td></tr><tr><td>192</td><td>0.431</td><td>0.443</td><td>0.411</td><td>0.428</td><td>0.414</td><td>0.421</td><td>0.405</td><td>0.416</td><td>0.423</td><td>0.446</td><td>0.456</td><td>0.457</td><td>1.007</td><td>0.786</td></tr><tr><td>336</td><td>0.450</td><td>0.456</td><td>0.445</td><td>0.446</td><td>0.431</td><td>0.436</td><td>0.439</td><td>0.443</td><td>0.444</td><td>0.462</td><td>0.486</td><td>0.487</td><td>1.038</td><td>0.784</td></tr><tr><td>720</td><td>0.472</td><td>0.484</td><td>0.487</td><td>0.478</td><td>0.449</td><td>0.466</td><td>0.472</td><td>0.490</td><td>0.469</td><td>0.492</td><td>0.515</td><td>0.517</td><td>1.144</td><td>0.857</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.284</td><td>0.343</td><td>0.285</td><td>0.344</td><td>0.274</td><td>0.336</td><td>0.289</td><td>0.353</td><td>0.332</td><td>0.374</td><td>0.332</td><td>0.368</td><td>1.549</td><td>0.952</td></tr><tr><td>192</td><td>0.355</td><td>0.387</td><td>0.356</td><td>0.387</td><td>0.339</td><td>0.379</td><td>0.383</td><td>0.418</td><td>0.407</td><td>0.446</td><td>0.426</td><td>0.434</td><td>3.792</td><td>1.542</td></tr><tr><td>336</td><td>0.379</td><td>0.411</td><td>0.377</td><td>0.410</td><td>0.331</td><td>0.380</td><td>0.448</td><td>0.465</td><td>0.400</td><td>0.447</td><td>0.477</td><td>0.479</td><td>4.215</td><td>1.642</td></tr><tr><td>720</td><td>0.400</td><td>0.435</td><td>0.395</td><td>0.434</td><td>0.379</td><td>0.422</td><td>0.605</td><td>0.551</td><td>0.412</td><td>0.469</td><td>0.453</td><td>0.490</td><td>3.656</td><td>1.619</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.289</td><td>0.344</td><td>0.292</td><td>0.348</td><td>0.290</td><td>0.342</td><td>0.299</td><td>0.343</td><td>0.326</td><td>0.390</td><td>0.510</td><td>0.492</td><td>0.626</td><td>0.560</td></tr><tr><td>192</td><td>0.323</td><td>0.368</td><td>0.329</td><td>0.369</td><td>0.332</td><td>0.369</td><td>0.335</td><td>0.365</td><td>0.365</td><td>0.415</td><td>0.514</td><td>0.495</td><td>0.725</td><td>0.619</td></tr><tr><td>336</td><td>0.353</td><td>0.387</td><td>0.364</td><td>0.391</td><td>0.366</td><td>0.392</td><td>0.369</td><td>0.386</td><td>0.392</td><td>0.425</td><td>0.510</td><td>0.492</td><td>1.005</td><td>0.741</td></tr><tr><td>720</td><td>0.398</td><td>0.416</td><td>0.415</td><td>0.419</td><td>0.420</td><td>0.424</td><td>0.425</td><td>0.421</td><td>0.446</td><td>0.458</td><td>0.527</td><td>0.493</td><td>1.133</td><td>0.845</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.166</td><td>0.256</td><td>0.167</td><td>0.257</td><td>0.165</td><td>0.255</td><td>0.167</td><td>0.260</td><td>0.180</td><td>0.271</td><td>0.205</td><td>0.293</td><td>0.355</td><td>0.462</td></tr><tr><td>192</td><td>0.221</td><td>0.295</td><td>0.229</td><td>0.300</td><td>0.220</td><td>0.292</td><td>0.224</td><td>0.303</td><td>0.252</td><td>0.318</td><td>0.278</td><td>0.336</td><td>0.595</td><td>0.586</td></tr><tr><td>336</td><td>0.278</td><td>0.333</td><td>0.289</td><td>0.343</td><td>0.278</td><td>0.329</td><td>0.281</td><td>0.342</td><td>0.324</td><td>0.364</td><td>0.343</td><td>0.379</td><td>1.270</td><td>0.871</td></tr><tr><td>720</td><td>0.365</td><td>0.388</td><td>0.363</td><td>0.386</td><td>0.367</td><td>0.385</td><td>0.397</td><td>0.421</td><td>0.410</td><td>0.420</td><td>0.414</td><td>0.419</td><td>3.001</td><td>1.267</td></tr></table>


Table 12: Multivariate long-term forecasting results with self-supervised PatchTST. An full benchmark regarding Table 4. The best results are in bold.


<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="6">PatchTST</td><td rowspan="2" colspan="2">DLinear</td><td rowspan="2" colspan="2">FEDformer</td><td rowspan="2" colspan="2">Autoformer</td><td rowspan="2" colspan="2">Informer</td></tr><tr><td colspan="2">Fine-tuning</td><td colspan="2">Lin. Prob.</td><td colspan="2">Sup.</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.145</td><td>0.195</td><td>0.163</td><td>0.216</td><td>0.152</td><td>0.199</td><td>0.176</td><td>0.237</td><td>0.238</td><td>0.314</td><td>0.249</td><td>0.329</td><td>0.354</td><td>0.405</td></tr><tr><td>192</td><td>0.193</td><td>0.243</td><td>0.205</td><td>0.252</td><td>0.197</td><td>0.243</td><td>0.220</td><td>0.282</td><td>0.275</td><td>0.329</td><td>0.325</td><td>0.370</td><td>0.419</td><td>0.434</td></tr><tr><td>336</td><td>0.244</td><td>0.280</td><td>0.253</td><td>0.289</td><td>0.249</td><td>0.283</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.377</td><td>0.351</td><td>0.391</td><td>0.583</td><td>0.543</td></tr><tr><td>720</td><td>0.321</td><td>0.337</td><td>0.320</td><td>0.336</td><td>0.320</td><td>0.335</td><td>0.323</td><td>0.362</td><td>0.389</td><td>0.409</td><td>0.415</td><td>0.426</td><td>0.916</td><td>0.705</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.388</td><td>0.273</td><td>0.400</td><td>0.288</td><td>0.367</td><td>0.251</td><td>0.410</td><td>0.282</td><td>0.576</td><td>0.359</td><td>0.597</td><td>0.371</td><td>0.733</td><td>0.410</td></tr><tr><td>192</td><td>0.400</td><td>0.277</td><td>0.412</td><td>0.293</td><td>0.385</td><td>0.259</td><td>0.423</td><td>0.287</td><td>0.610</td><td>0.380</td><td>0.607</td><td>0.382</td><td>0.777</td><td>0.435</td></tr><tr><td>336</td><td>0.408</td><td>0.280</td><td>0.425</td><td>0.307</td><td>0.398</td><td>0.265</td><td>0.436</td><td>0.296</td><td>0.608</td><td>0.375</td><td>0.623</td><td>0.387</td><td>0.776</td><td>0.434</td></tr><tr><td>720</td><td>0.447</td><td>0.310</td><td>0.457</td><td>0.317</td><td>0.434</td><td>0.287</td><td>0.466</td><td>0.315</td><td>0.621</td><td>0.375</td><td>0.639</td><td>0.395</td><td>0.827</td><td>0.466</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.368</td><td>0.398</td><td>0.372</td><td>0.402</td><td>0.375</td><td>0.399</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.415</td><td>0.435</td><td>0.446</td><td>0.941</td><td>0.769</td></tr><tr><td>192</td><td>0.425</td><td>0.439</td><td>0.411</td><td>0.428</td><td>0.414</td><td>0.421</td><td>0.405</td><td>0.416</td><td>0.423</td><td>0.446</td><td>0.456</td><td>0.457</td><td>1.007</td><td>0.786</td></tr><tr><td>336</td><td>0.470</td><td>0.471</td><td>0.442</td><td>0.454</td><td>0.431</td><td>0.436</td><td>0.439</td><td>0.443</td><td>0.444</td><td>0.462</td><td>0.486</td><td>0.487</td><td>1.038</td><td>0.784</td></tr><tr><td>720</td><td>0.472</td><td>0.484</td><td>0.497</td><td>0.501</td><td>0.449</td><td>0.466</td><td>0.472</td><td>0.490</td><td>0.469</td><td>0.492</td><td>0.515</td><td>0.517</td><td>1.144</td><td>0.857</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.285</td><td>0.345</td><td>0.280</td><td>0.341</td><td>0.274</td><td>0.334</td><td>0.289</td><td>0.353</td><td>0.332</td><td>0.374</td><td>0.332</td><td>0.368</td><td>1.549</td><td>0.952</td></tr><tr><td>192</td><td>0.350</td><td>0.388</td><td>0.350</td><td>0.387</td><td>0.339</td><td>0.379</td><td>0.383</td><td>0.418</td><td>0.407</td><td>0.446</td><td>0.426</td><td>0.434</td><td>3.792</td><td>1.542</td></tr><tr><td>336</td><td>0.378</td><td>0.410</td><td>0.373</td><td>0.410</td><td>0.331</td><td>0.380</td><td>0.448</td><td>0.465</td><td>0.400</td><td>0.447</td><td>0.477</td><td>0.479</td><td>4.215</td><td>1.642</td></tr><tr><td>720</td><td>0.401</td><td>0.438</td><td>0.398</td><td>0.436</td><td>0.379</td><td>0.422</td><td>0.605</td><td>0.551</td><td>0.412</td><td>0.469</td><td>0.453</td><td>0.490</td><td>3.656</td><td>1.619</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.288</td><td>0.345</td><td>0.291</td><td>0.346</td><td>0.290</td><td>0.342</td><td>0.299</td><td>0.343</td><td>0.326</td><td>0.390</td><td>0.510</td><td>0.492</td><td>0.626</td><td>0.560</td></tr><tr><td>192</td><td>0.330</td><td>0.372</td><td>0.335</td><td>0.373</td><td>0.332</td><td>0.369</td><td>0.335</td><td>0.365</td><td>0.365</td><td>0.415</td><td>0.514</td><td>0.495</td><td>0.725</td><td>0.619</td></tr><tr><td>336</td><td>0.359</td><td>0.392</td><td>0.365</td><td>0.391</td><td>0.366</td><td>0.392</td><td>0.369</td><td>0.386</td><td>0.392</td><td>0.425</td><td>0.510</td><td>0.492</td><td>1.005</td><td>0.741</td></tr><tr><td>720</td><td>0.406</td><td>0.421</td><td>0.423</td><td>0.424</td><td>0.420</td><td>0.424</td><td>0.425</td><td>0.421</td><td>0.446</td><td>0.458</td><td>0.527</td><td>0.493</td><td>1.133</td><td>0.845</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.164</td><td>0.256</td><td>0.166</td><td>0.257</td><td>0.165</td><td>0.255</td><td>0.167</td><td>0.260</td><td>0.180</td><td>0.271</td><td>0.205</td><td>0.293</td><td>0.355</td><td>0.462</td></tr><tr><td>192</td><td>0.223</td><td>0.296</td><td>0.221</td><td>0.295</td><td>0.220</td><td>0.292</td><td>0.224</td><td>0.303</td><td>0.252</td><td>0.318</td><td>0.278</td><td>0.336</td><td>0.595</td><td>0.586</td></tr><tr><td>336</td><td>0.277</td><td>0.332</td><td>0.277</td><td>0.332</td><td>0.278</td><td>0.329</td><td>0.281</td><td>0.342</td><td>0.324</td><td>0.364</td><td>0.343</td><td>0.379</td><td>1.270</td><td>0.871</td></tr><tr><td>720</td><td>0.365</td><td>0.387</td><td>0.368</td><td>0.389</td><td>0.367</td><td>0.385</td><td>0.397</td><td>0.421</td><td>0.410</td><td>0.420</td><td>0.414</td><td>0.419</td><td>3.001</td><td>1.267</td></tr></table>


Table 13: Transfer learning task: PatchTST is pre-trained on Electricity dataset and the model is transferred to other datasets. A full benchmark regarding Table 13. The best results are in bold.


## A.6.2 RESULTS WITH DIFFERENT MODEL PARAMETERS

To see whether PatchTST is sensitive to the choice of Transformer settings, we perform another experiments with varying model parameters. We vary the number of Transformer layers $L = \{ 3 , 4 , 5 \}$ and select the model dimension $D = \{ 1 2 8 , 2 5 6 \}$ while the inner-layer of the feed forward network is $F = 2 D$ . In total, there are 6 different sets of model hyper-parameters to examine. Figure 5 shows the MSE scores of these combinations on different datasets. Except ILI dataset reveals high variance with different hyper-parameter settings, other datasets are robust to the choice of model hyper-parameters. 

<table><tr><td colspan="2">L</td><td colspan="2">PatchTST/42 supervised</td><td colspan="2">PatchTST/42 self-supervised</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.1525±0.0024</td><td>0.2002±0.0023</td><td>0.1450±0.0008</td><td>0.1937±0.0010</td></tr><tr><td>192</td><td>0.1975±0.0015</td><td>0.2434±0.0010</td><td>0.1893±0.0003</td><td>0.2364±0.0006</td></tr><tr><td>336</td><td>0.2494±0.0012</td><td>0.2841±0.0014</td><td>0.2413±0.0003</td><td>0.2774±0.0005</td></tr><tr><td>720</td><td>0.3194±0.0002</td><td>0.3352±0.0003</td><td>0.3156±0.0020</td><td>0.3316±0.0016</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.3669±0.0006</td><td>0.2504±0.0007</td><td>0.3528±0.0022</td><td>0.2443±0.0016</td></tr><tr><td>192</td><td>0.3858±0.0004</td><td>0.2586±0.0004</td><td>0.3729±0.0013</td><td>0.2531±0.0009</td></tr><tr><td>336</td><td>0.3994±0.0010</td><td>0.2672±0.0016</td><td>0.3846±0.0020</td><td>0.2588±0.0011</td></tr><tr><td>720</td><td>0.4383±0.0097</td><td>0.2913±0.0104</td><td>0.4241±0.0007</td><td>0.2816±0.0010</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.1304±0.0006</td><td>0.2234±0.0006</td><td>0.1256±0.0002</td><td>0.2210±0.0003</td></tr><tr><td>192</td><td>0.1482±0.0002</td><td>0.2403±0.0002</td><td>0.1451±0.0002</td><td>0.2397±0.0010</td></tr><tr><td>336</td><td>0.1659±0.0006</td><td>0.2596±0.0006</td><td>0.1624±0.0010</td><td>0.2576±0.0009</td></tr><tr><td>720</td><td>0.2019±0.0006</td><td>0.2917±0.0006</td><td>0.1990±0.0002</td><td>0.2916±0.0002</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.3752±0.0008</td><td>0.3999±0.0004</td><td>0.3700±0.0035</td><td>0.4001±0.0023</td></tr><tr><td>192</td><td>0.4127±0.0012</td><td>0.4207±0.0006</td><td>0.4146±0.0012</td><td>0.4287±0.0013</td></tr><tr><td>336</td><td>0.4278±0.0033</td><td>0.4334±0.0028</td><td>0.4285±0.0018</td><td>0.4402±0.0017</td></tr><tr><td>720</td><td>0.4462±0.0035</td><td>0.4637±0.0027</td><td>0.4670±0.0052</td><td>0.4768±0.0033</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.2749±0.0005</td><td>0.3363±0.0006</td><td>0.2869±0.0039</td><td>0.3439±0.0016</td></tr><tr><td>192</td><td>0.3385±0.0010</td><td>0.3789±0.0014</td><td>0.3523±0.0048</td><td>0.3855±0.0027</td></tr><tr><td>336</td><td>0.3288±0.0010</td><td>0.3823±0.0027</td><td>0.3779±0.0057</td><td>0.4112±0.0030</td></tr><tr><td>720</td><td>0.3784±0.0010</td><td>0.4212±0.0009</td><td>0.3993±0.0054</td><td>0.4385±0.0038</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.2893±0.0009</td><td>0.3415±0.0007</td><td>0.2876±0.0012</td><td>0.3427±0.0011</td></tr><tr><td>192</td><td>0.3316±0.0008</td><td>0.3695±0.0007</td><td>0.3296±0.0026</td><td>0.3688±0.0016</td></tr><tr><td>336</td><td>0.3661±0.0022</td><td>0.3914±0.0012</td><td>0.3583±0.0015</td><td>0.3879±0.0016</td></tr><tr><td>720</td><td>0.4200±0.0056</td><td>0.4243±0.0033</td><td>0.4094±0.0044</td><td>0.4193±0.0013</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.1647±0.0011</td><td>0.2538±0.0010</td><td>0.1637±0.0020</td><td>0.2537±0.0024</td></tr><tr><td>192</td><td>0.2223±0.0018</td><td>0.2936±0.0014</td><td>0.2175±0.0011</td><td>0.2908±0.0013</td></tr><tr><td>336</td><td>0.2775±0.0020</td><td>0.3297±0.0010</td><td>0.2706±0.0016</td><td>0.3260±0.0016</td></tr><tr><td>720</td><td>0.3648±0.0024</td><td>0.3833±0.0010</td><td>0.3539±0.0023</td><td>0.3799±0.0024</td></tr></table>


Table 14: Multivariate long-term forecasting results with different random seeds in supervised and self-supervised PatchTST/42. The best results are in bold.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/86ff0dabd2d25c6dba95f53b91846f624bacad4b7d0dcff4e9fe1db717ee5e19.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/e57e552681d255b540aa47f2edba0e481da2644d5d15f571e9aaf9921ba497a2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/32482da431adf9c619c5f46801a20da5a1468eec82108bd5b2aef5bef5eb9193.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/6b627fae94f59f8ae9f1198057e6e365fecccea3313580b40af4a92b49ad2eac.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/a94c048e6c2d984c809c1387da438755f68433514416abfab6e8067524cae123.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/f38b4854946276d39f5f4a7b8d9ee136bd8d8f0204f057b89e19aa8a24160eba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/9ec33aedf76fb36e8d60e6b6eedd8743e67157acafac9d39d902531021977edf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/0ddf27b4b593457d59304fa59341980e0f60de8a08e255e1411cc1158192364b.jpg)



Figure 5: MSE scores with varying model parameters. Each bar indicates the MSE score of a parameter combination. The combinations (L, D) = (3, 128), (3, 256), (4, 128), (4, 256), (5, 128), (5, 256) are orderly labeled from 1 to 6 in the figure. The model is run with supervised PatchTST/42 to forecast 96 steps. For Traffic and Electricity datasets, we reduce the maximum number of epochs to 50 to save computational time.


## A.7 CHANNEL-INDEPENDENCE ANALYSIS

Intuitively, channel-mixing models should outperform the channel-independent ones since they have more flexibility to explore the cross-channel information, while with channel-independence the correlation is indirectly learnt via weight sharing. However, this is contrast to what we have observed. In Section A.7.1 we will provide an in-depth analysis on why channel-independence has better forecasting performance than channel mixing, and in Section A.7.2 we show that channel-independence is a general technique that can be used not only for PatchTST but also for the other models. 

## A.7.1 CHANNEL-INDEPENDENCE VS CHANNEL-MIXING

## We find 3 key factors that makes channel-independent models more preferable:

• Adaptability: Since each time series is passed through the Transformer separately, it generates its own attention maps. That means different series can learn different attention patterns for their prediction, as shown in Figure 6. In contrast, with the channel mixing approach, all the series share the same attention patterns, which may be harmful if the underlying multivariate time series carries series of different behaviors. Figure 6 reveals an interesting observation that the prediction of unrelated time series relies on different attention patterns while similar series can produce similar maps (e.g. series 11, 25, and 81 contain similar patterns while they are different from others). We suspect this adaptability is one of the main reasons why PatchTST performs much better forecasting than Informer and other channel-mixing models. 

• Channel-mixing models may need more training data to match the performance of the channelindependent ones. The flexibility of learning cross-channel correlations could be a doubleedged sword, because it may need much more data to learn the information from different channels and different time steps jointly and appropriately, while channel-independent models only focus on learning information along the time axis. We examine this assumption by experiments where we train the models with varying training data size and and the result is shown on left panel of Figure 7. It is clear that channel-independent models converges faster against the size of training data. As what we have observed in the figure and Table 2, the size of those widely used time series datasets may not be large enough for channel-mixing models to obtain similar performances in supervised learning. 

• Channel-independent models are less likely to overfit data during training. We record the MSE loss on test data and plot on the right panel of Figure 7. Channel-mixing models show overfitting after a few initial epochs, while channel-independent models continue optimizing the loss with more training epochs. The best trained models are determined by validation data, which are approximately the lowest points in the test loss curves. It is clear that the forecasting performance of channel-independent models are better. 

Furthermore, we would like to comment on a few additional technical advantages of channelindependence: (1) Possibility of learning spatial correlations across series: Although we haven’t focused on this research in our paper, the channel-independence design can be naturally extended to learn cross-channel relationships by using methods like graph neural networks (Cao et al., 2020; Chen et al., 2021). (2) Multi-task learning where different loss types can be imposed on different time series where the same underlying Transformer model is shared. (3) More robust to noise: If noise is dominant in one or several series, this noise will be projected to other series in the embedding space if we mix channels. Channel independence can mitigate this problem by only retaining the noise in these noisy channels. We can further alleviate the noise by introducing smaller weights to the objective losses that associate with noisy channels. 

## A.7.2 PERFORMANCE OF CHANNEL-INDEPENDENCE ON OTHER MODELS

To show that channel-independence is a general technique that can be applied to the other models, we apply it to Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021), and FEDformer (Zhou et al., 2022). The results are shown in Table 15. The channel-independent technique can improve the forecasting performance of those models generally. Although they are still not able to outperform PatchTST which is based on vanilla attention mechanism, we believe that more performance boost and computational reduction can be obtained with more advanced attention designs incorporating the channel-independence architecture. 

<table><tr><td colspan="2">Models</td><td colspan="2">PatchTST/42</td><td colspan="2">Informer</td><td colspan="2">Informer-CI</td><td colspan="2">Autoformer</td><td colspan="2">Autoformer-CI</td><td colspan="2">FEDformer</td><td colspan="2">FEDformer-CI</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.152</td><td>0.199</td><td>0.300</td><td>0.384</td><td>0.174</td><td>0.232</td><td>0.266</td><td>0.336</td><td>0.227</td><td>0.289</td><td>0.217</td><td>0.296</td><td>0.214</td><td>0.278</td></tr><tr><td>192</td><td>0.197</td><td>0.243</td><td>0.598</td><td>0.544</td><td>0.214</td><td>0.270</td><td>0.307</td><td>0.367</td><td>0.269</td><td>0.318</td><td>0.276</td><td>0.336</td><td>0.258</td><td>0.322</td></tr><tr><td>336</td><td>0.249</td><td>0.283</td><td>0.578</td><td>0.523</td><td>0.266</td><td>0.310</td><td>0.359</td><td>0.395</td><td>0.315</td><td>0.344</td><td>0.339</td><td>0.380</td><td>0.302</td><td>0.336</td></tr><tr><td>720</td><td>0.320</td><td>0.335</td><td>1.059</td><td>0.741</td><td>0.327</td><td>0.356</td><td>0.419</td><td>0.428</td><td>0.384</td><td>0.389</td><td>0.403</td><td>0.428</td><td>0.374</td><td>0.369</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.367</td><td>0.251</td><td>0.719</td><td>0.391</td><td>0.705</td><td>0.402</td><td>0.613</td><td>0.388</td><td>-</td><td>-</td><td>0.587</td><td>0.366</td><td>-</td><td>-</td></tr><tr><td>192</td><td>0.385</td><td>0.259</td><td>0.696</td><td>0.379</td><td>0.720</td><td>0.407</td><td>0.616</td><td>0.382</td><td>-</td><td>-</td><td>0.604</td><td>0.373</td><td>-</td><td>-</td></tr><tr><td>336</td><td>0.398</td><td>0.265</td><td>0.777</td><td>0.420</td><td>0.750</td><td>0.421</td><td>0.622</td><td>0.337</td><td>-</td><td>-</td><td>0.621</td><td>0.383</td><td>-</td><td>-</td></tr><tr><td>720</td><td>0.434</td><td>0.287</td><td>0.864</td><td>0.472</td><td>-</td><td>-</td><td>0.660</td><td>0.408</td><td>-</td><td>-</td><td>0.626</td><td>0.382</td><td>-</td><td>-</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.130</td><td>0.222</td><td>0.274</td><td>0.368</td><td>0.203</td><td>0.299</td><td>0.201</td><td>0.317</td><td>-</td><td>-</td><td>0.193</td><td>0.308</td><td>-</td><td>-</td></tr><tr><td>192</td><td>0.148</td><td>0.240</td><td>0.296</td><td>0.386</td><td>0.221</td><td>0.316</td><td>0.222</td><td>0.334</td><td>-</td><td>-</td><td>0.201</td><td>0.315</td><td>-</td><td>-</td></tr><tr><td>336</td><td>0.167</td><td>0.261</td><td>0.300</td><td>0.394</td><td>0.241</td><td>0.337</td><td>0.231</td><td>0.338</td><td>-</td><td>-</td><td>0.214</td><td>0.329</td><td>-</td><td>-</td></tr><tr><td>720</td><td>0.202</td><td>0.291</td><td>0.373</td><td>0.439</td><td>0.314</td><td>0.391</td><td>0.254</td><td>0.361</td><td>-</td><td>-</td><td>0.246</td><td>0.355</td><td>-</td><td>-</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>1.522</td><td>0.814</td><td>5.764</td><td>1.677</td><td>5.514</td><td>1.629</td><td>3.483</td><td>1.287</td><td>4.210</td><td>1.500</td><td>3.228</td><td>1.260</td><td>3.280</td><td>1.264</td></tr><tr><td>36</td><td>1.430</td><td>0.834</td><td>4.755</td><td>1.467</td><td>5.515</td><td>1.628</td><td>3.103</td><td>1.148</td><td>2.809</td><td>1.162</td><td>2.679</td><td>1.080</td><td>2.862</td><td>1.126</td></tr><tr><td>48</td><td>1.673</td><td>0.854</td><td>4.763</td><td>1.469</td><td>5.263</td><td>1.574</td><td>2.669</td><td>1.085</td><td>3.218</td><td>1.267</td><td>2.622</td><td>1.078</td><td>2.834</td><td>1.150</td></tr><tr><td>60</td><td>1.529</td><td>0.862</td><td>5.264</td><td>1.564</td><td>5.330</td><td>1.602</td><td>2.770</td><td>1.125</td><td>3.627</td><td>1.396</td><td>2.857</td><td>1.157</td><td>3.115</td><td>1.240</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.375</td><td>0.399</td><td>0.865</td><td>0.713</td><td>0.590</td><td>0.517</td><td>0.449</td><td>0.459</td><td>0.414</td><td>0.421</td><td>0.376</td><td>0.419</td><td>0.387</td><td>0.407</td></tr><tr><td>192</td><td>0.414</td><td>0.421</td><td>1.008</td><td>0.792</td><td>0.677</td><td>0.566</td><td>0.500</td><td>0.482</td><td>0.453</td><td>0.448</td><td>0.420</td><td>0.448</td><td>0.439</td><td>0.438</td></tr><tr><td>336</td><td>0.431</td><td>0.436</td><td>1.107</td><td>0.809</td><td>0.710</td><td>0.600</td><td>0.521</td><td>0.496</td><td>0.496</td><td>0.468</td><td>0.459</td><td>0.465</td><td>0.479</td><td>0.455</td></tr><tr><td>720</td><td>0.449</td><td>0.466</td><td>1.181</td><td>0.865</td><td>0.777</td><td>0.660</td><td>0.514</td><td>0.512</td><td>0.662</td><td>0.568</td><td>0.506</td><td>0.507</td><td>0.485</td><td>0.478</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.274</td><td>0.336</td><td>3.755</td><td>1.525</td><td>0.390</td><td>0.410</td><td>0.358</td><td>0.397</td><td>0.337</td><td>0.373</td><td>0.346</td><td>0.388</td><td>0.297</td><td>0.348</td></tr><tr><td>192</td><td>0.339</td><td>0.379</td><td>5.602</td><td>1.931</td><td>0.456</td><td>0.463</td><td>0.456</td><td>0.452</td><td>0.409</td><td>0.419</td><td>0.429</td><td>0.439</td><td>0.382</td><td>0.399</td></tr><tr><td>336</td><td>0.331</td><td>0.380</td><td>4.721</td><td>1.835</td><td>0.523</td><td>0.503</td><td>0.482</td><td>0.486</td><td>0.432</td><td>0.443</td><td>0.496</td><td>0.487</td><td>0.410</td><td>0.428</td></tr><tr><td>720</td><td>0.379</td><td>0.422</td><td>3.647</td><td>1.625</td><td>0.843</td><td>0.661</td><td>0.515</td><td>0.511</td><td>0.443</td><td>0.463</td><td>0.463</td><td>0.474</td><td>0.422</td><td>0.444</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.290</td><td>0.342</td><td>0.672</td><td>0.571</td><td>0.383</td><td>0.414</td><td>0.505</td><td>0.475</td><td>0.455</td><td>0.441</td><td>0.379</td><td>0.419</td><td>0.408</td><td>0.413</td></tr><tr><td>192</td><td>0.332</td><td>0.369</td><td>0.795</td><td>0.669</td><td>0.420</td><td>0.434</td><td>0.553</td><td>0.496</td><td>0.598</td><td>0.512</td><td>0.426</td><td>0.441</td><td>0.445</td><td>0.432</td></tr><tr><td>336</td><td>0.366</td><td>0.392</td><td>1.212</td><td>0.871</td><td>0.465</td><td>0.467</td><td>0.621</td><td>0.537</td><td>0.566</td><td>0.504</td><td>0.445</td><td>0.459</td><td>0.476</td><td>0.452</td></tr><tr><td>720</td><td>0.420</td><td>0.424</td><td>1.166</td><td>0.823</td><td>0.529</td><td>0.502</td><td>0.671</td><td>0.561</td><td>0.680</td><td>0.557</td><td>0.543</td><td>0.490</td><td>0.533</td><td>0.481</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.165</td><td>0.255</td><td>0.365</td><td>0.453</td><td>0.208</td><td>0.298</td><td>0.255</td><td>0.339</td><td>0.218</td><td>0.308</td><td>0.203</td><td>0.287</td><td>0.198</td><td>0.284</td></tr><tr><td>192</td><td>0.220</td><td>0.292</td><td>0.533</td><td>0.563</td><td>0.274</td><td>0.345</td><td>0.281</td><td>0.340</td><td>0.281</td><td>0.339</td><td>0.269</td><td>0.328</td><td>0.259</td><td>0.320</td></tr><tr><td>336</td><td>0.278</td><td>0.329</td><td>1.363</td><td>0.887</td><td>0.351</td><td>0.394</td><td>0.339</td><td>0.372</td><td>0.336</td><td>0.370</td><td>0.325</td><td>0.366</td><td>0.315</td><td>0.353</td></tr><tr><td>720</td><td>0.367</td><td>0.385</td><td>3.379</td><td>1.338</td><td>0.482</td><td>0.474</td><td>0.433</td><td>0.432</td><td>0.428</td><td>0.418</td><td>0.421</td><td>0.415</td><td>0.412</td><td>0.406</td></tr></table>


Table 15: Channel-independence for other models. CI denote channel-independence. Baselines without CI are cited from Zeng et al. (2022). The better results between CI and non-CI versions are in bold. PatchTST/42 is placed on the left for easy reference to other CI-based models. ’-’ denotes running out of GPU memory even with batch size 1, or exceeding the maximum running time (12 hours).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/41f15967d0905d9f4a8b8564cb87ef1f125fb65e0837f609b12ce038665c592c.jpg)



Figure 6: Attention maps and the forecasting of a few time series from Electricity dataset run with supervised PatchTST/64. Attention map is calculated by averaging the attention matrices over all the heads and across all the layers. For each time series, we show the attention map and the prediction in orange. The blue curves are the actual data. The curves before the back lines are the actual input data. Channel-independence design allow each series to learn its own attention map for forecasting in which the pattern can be more similar for more correlated series and different otherwise.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/486cba5787fd1b96b62530d74a1b0fd4ae12389a1ffa968b359937581b4ccd15.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/0a7a58ed-6a20-43b0-beb2-f80cb55154eb/13ace895767b2244e421442afd5bbd899837923e2cf9e7f8e61f8f51738560c9.jpg)



Figure 7: Channel-independence vs channel-mixing on Weather dataset. The base model is PatchTST/42, and the prediction length is 96. We plot the mean values and error bars with 5 different random seeds: {2019, 2020, 2021, 2022, 2023}. Left Panel: Test loss vs train size. Here, train size denotes the fraction of the training data that is used to learn the model from scratch. Channelindependence contributes to a quicker convergence as more training data is available. Right Panel: Test loss vs epochs. Here, we use full train data and plot the first 20 epochs. Channel-mixing model quickly overfits the data.
