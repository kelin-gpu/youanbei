# ITRANSFORMER: INVERTED TRANSFORMERS ARE EFFECTIVE FOR TIME SERIES FORECASTING

Yong Liu<sup>∗</sup>, Tengge Hu<sup>∗</sup>, Haoran Zhang<sup>∗</sup>, Haixu Wu, Shiyu Wang<sup>§</sup>, Lintao Ma<sup>§</sup>, Mingsheng Long<sup>B</sup> School of Software, BNRist, Tsinghua University, Beijing 100084, China 

<sup>§</sup>Ant Group, Hangzhou, China 

{liuyong21,htg21,z-hr20,whx20}@mails.tsinghua.edu.cn 

{weiming.wsy,lintao.mlt}@antgroup.com, mingsheng@tsinghua.edu.cn 

## ABSTRACT

The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any modification to the basic components. We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves state-of-the-art on challenging real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting. Code is available at this repository: https://github.com/thuml/iTransformer. 

## 1 INTRODUCTION

Transformer (Vaswani et al., 2017) has achieved tremendous success in natural language processing (Brown et al., 2020) and computer vision (Dosovitskiy et al., 2021), growing into the foundation model that follows the scaling law (Kaplan et al., 2020). Inspired by the immense success in extensive fields, Transformer with strong capabilities of depicting pairwise dependencies and extracting multi-level representations in sequences is emerging in time series forecasting (Wu et al., 2021; Nie et al., 2023). 

However, researchers have recently begun to question the validity of Transformer-based forecasters, which typically embed multiple variates of the same timestamp into indistinguishable channels and apply attention on these temporal tokens to capture temporal dependencies. Considering the numerical but less semantic relationship among time points, researchers find that simple linear layers, which can be traced back to statistical forecasters (Box & Jenkins, 1968), have exceeded complicated Transformers on both et al., 2023; Das et al., 2023). Meanwhile, ensuring the independ information is ever more highlighted by recent research that explicitly models multivariate correlations to achieve accurate forecasting (Zhang & Yan, 2023; Ekambaram et al., 2023), but this goal can be hardly achieved without subverting the vanilla Transformer architecture. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/2dfab082e3192afb39b7fcee3340164508b98e75ccd85950a464c875d2cfa3a0.jpg)



Figure 1: Performance of iTransformer. Average results (MSE) are reported following TimesNet (2023).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/dbee56430797ad8af4e79617842c08f570e5b870732db900fb8d3837c571525e.jpg)



Figure 2: Comparison between the vanilla Transformer (top) and the proposed iTransformer (bottom). Transformer embeds the temporal token, which contains the multivariate representation of each time step. iTransformer embeds each series independently to the variate token, such that the attention mod ule depicts the multivariate correlations and the feed-forward network encodes series representations.


Considering the disputes of Transformer-based forecasters, we reflect on why Transformers perform even worse than linear models in time series forecasting while acting predominantly in many other fields. We notice that the existing structure of Transformer-based forecasters may be not suitable for multivariate time series forecasting. As shown on the top of Figure 2, it is notable that the points of the same time step that basically represent completely different physical meanings recorded by inconsistent measurements are embedded into one token with wiped-out multivariate correlations. And the token formed by a single time step can struggle to reveal beneficial information due to excessively local receptive field and time-unaligned events represented by simultaneous time points. Besides, while series variations can be greatly influenced by the sequence order, permutationinvariant attention mechanisms are improperly adopted on the temporal dimension (Zeng et al., 2023). Consequently, Transformer is weakened to capture essential series representations and portray multivariate correlations, limiting its capacity and generalization ability on diverse time series data. 

Concerning the potential risks of embedding multivariate points of a timestamp as a (temporal) token, we take an inverted view on time series and embed the whole time series of each variate independently into a (variate) token, the extreme case of Patching (Nie et al., 2023) that enlarges local receptive field. By inverting, the embedded token aggregates the global representations of series that can be more variate-centric and better leveraged by booming attention mechanisms for multivariate correlating. Meanwhile, the feed-forward network can be proficient enough to learn generalizable representations for distinct variates encoded from arbitrary lookback series and decoded to predict future series. 

Based on the above motivations, we believe it is not that Transformer is ineffective for time series forecasting, but rather it is improperly used. In this paper, we revisit the structure of Transformer and advocate iTransformer as a fundamental backbone for time series forecasting. Technically, we embed each time series as variate tokens, adopt the attention for multivariate correlations, and employ the feed-forward network for series representations. Experimentally, the proposed iTransformer achieves state-of-the-art performance on real-world forecasting benchmarks shown in Figure 1 and surprisingly tackles the pain points of Transformer-based forecasters. Our contributions lie in three aspects: 

• We reflect on the architecture of Transformer and refine that the competent capability of native Transformer components on multivariate time series is underexplored. 

• We propose iTransformer that regards independent time series as tokens to capture multivari ate correlations by self-attention and utilize layer normalization and feed-forward network modules to learn better series-global representations for time series forecasting. 

• Experimentally, iTransformer achieves comprehensive state-of-the-art on real-world benchmarks. We extensively analyze the inverted modules and architecture choices, indicating a promising direction for the future improvement of Transformer-based forecasters. 

## 2 RELATED WORK

With the progressive breakthrough made in natural language processing and computer vision areas, elaboratively designed Transformer variants are proposed to tackle ubiquitous time series forecasting applications. Going beyond contemporaneous TCNs (Bai et al., 2018; Liu et al., 2022a) and RNNbased forecasters (Zhao et al., 2017; Rangapuram et al., 2018; Salinas et al., 2020), Transformer has exhibited powerful sequence modeling capability and promising model scalability, leading to the trend of passionate modifications adapted for time series forecasting. 

Through a systematical review of Transformer-based forecasters, we conclude that existing modifications can be divided into four categories by whether to modify the component and architecture. As shown in Figure 3, the first category (Wu et al., 2021; Li et al., 2021; Zhou et al., 2022), which is the most common practice, mainly concerns the component adaptation, especially the attention module for the temporal dependency modeling and the complexity optimization on long sequences. Nevertheless, with the rapid emergence of linear forecasters (Oreshkin et al., 2019; Zeng et al., 2023; Das et al., 2023; Liu et al., 2023), the impressive performance and efficiency continuously challenge this direction. Soon afterward, the second category attempts to fully utilize Transformer. It pays more attention to the inherent processing of time series, such as Stationarization (Liu et al., 2022b), Channel Independence, and Patching (Nie et al., 2023), which bring about consistently improved performance. Moreover, faced with the increasing significance of the independence and mutual interactions of multiple variates, the third category refurbishes Transformer in both aspects of component and architecture. Representative (Zhang & Yan, 2023) explicitly captures the cross-time and cross-variate dependencies by the renovated attention mechanism and architecture. 

Unlike previous works, iTransformer modifies none of the native components of Transformer. Instead, we adopt the components on the inverted dimensions with the altered architecture, as the only one that belongs to the fourth category to our best knowledge. We believe the capabilities of the components have stood the test extensively, the truth is that the architecture of Transformer is improperly adopted. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/2236c654a277a5fe53cc7a7f596a89893a60b1c91f537d38ee18a0d2854a02c2.jpg)



Figure 3: Transformer-based forecasters categorized by component and architecture modifications.


## 3 ITRANSFORMER

In multivariate time series forecasting, given historical observations ${ \bf X } = \{ { \bf x } _ { 1 } , \ldots , { \bf x } _ { T } \} \in \mathbb { R } ^ { T \times N }$ with T time steps and N variates, we predict the future S time steps $\mathbf { Y } = \{ \mathbf { x } _ { T + 1 } , \dotsc , \mathbf { x } _ { T + S } \} \in$ $\mathbb { R } ^ { S \times N }$ . For convenience, we denote $\mathbf { X } _ { t , : }$ as the simultaneously recorded time points at the step t, and $\mathbf { X } _ { : , n }$ as the whole time series of each variate indexed by n. It is notable that $\mathbf { X } _ { t , : }$ may not contain time points that essentially reflect the same event in real-world scenarios because of the systematical time lags among variates in the dataset. Besides, the elements of $\mathbf { X } _ { t , : }$ can be distinct from each other in physical measurements and statistical distributions, for which a variate $\mathbf { X } _ { : , n }$ generally shares. 

## 3.1 STRUCTURE OVERVIEW

Our proposed iTransformer illustrated in Figure 4 adopts the encoder-only architecture of Transformer (Vaswani et al., 2017), including the embedding, projection, and Transformer blocks. 

Embedding the whole series as the token Most Transformer-based forecasters typically regard multiple variates of the same time as the (temporal) token and follow the generative formulation of forecasting tasks. However, we find the approach on the numerical modality can be less instructive for learning attention maps, which is supported by increasing applications of Patching (Dosovitskiy et al., 2021; Nie et al., 2023) that broadens the respective field. Meanwhile, the triumph of linear forecasters also challenges the necessity of adopting a heavy encoder-decoder Transformer for generating tokens. Instead, our proposed encoder-only iTransformer focuses on representation learning and adaptive correlating of multivariate series. Each time series driven by the underlying complicated process is firstly tokenized to describe the properties of the variate, applied by self-attention for mutual interactions, and individually processed by feed-forward networks for series representations. Notably, the task to generate the predicted series is essentially delivered to linear layers, which has been proven competent by previous work (Das et al., 2023) and we provide a detailed analysis in the next section. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/d828a8f734ee27489a500639c4056dee0641eec4881c965bb4b62be831197e64.jpg)



Figure 4: Overall structure of iTransformer, which shares the same modular arrangement with the encoder of Transformer. (a) Raw series of different variates are independently embedded as tokens. (b) Self-attention is applied to embedded variate tokens with enhanced interpretability revealing multivariate correlations. (c) Series representations of each token are extracted by the shared feedforward network. (d) Layer normalization is adopted to reduce the discrepancies among variates.


Based on the above considerations, in iTransformer, the process of predicting future series of each specific variate $\hat { \mathbf { Y } } _ { : , n }$ based on the lookback series $\mathbf { X } _ { : , n }$ is simply formulated as follows: 

$$
\begin{array}{c} \mathbf {h} _ {n} ^ {0} = \text {Embedding} (\mathbf {X} _ {:, n}), \\ \mathbf {H} ^ {l + 1} = \text {TrmBlock} (\mathbf {H} ^ {l}), l = 0, \ldots , L - 1, \\ \hat {\mathbf {Y}} _ {:, n} = \text {Projection} (\mathbf {h} _ {n} ^ {L}), \end{array}\tag{1}
$$

where $\mathbf { H } = \{ \mathbf { h } _ { 1 } , \dots , \mathbf { h } _ { N } \} \in \mathbb { R } ^ { N \times D }$ contains N embedded tokens of dimension D and the superscript denotes the layer index. Embedding $: \mathbb { R } ^ { T } \mapsto \mathbb { R } ^ { D }$ and Projection : $\mathbb { R } ^ { D } \mapsto \mathbb { R } ^ { S }$ are both implemented by multi-layer perceptron (MLP). The obtained variate tokens interact with each other by self-attention and are independently processed by the shared feed-forward network in each TrmBlock. Specifically, as the order of sequence is implicitly stored in the neuron permutation of the feed-forward network, the position embedding in the vanilla Transformer is no longer needed here. 

iTransformers The architecture essentially presupposes no more specific requirements on Transformer variants, other than the attention is applicable for multivariate correlation. Thus, a bundle of efficient attention mechanisms (Li et al., 2021; Wu et al., 2022; Dao et al., 2022) can be the plugins, reducing the complexity when the variate number grows large. Besides, with the input flexibility of attention, the token number can vary from training to inference, and the model is allowed to be trained on arbitrary numbers of variates. The inverted Transformers, named iTransformers, are extensively evaluated in experiments of Section 4.2 and demonstrate advantages on time series forecasting. 

## 3.2 INVERTED TRANSFORMER COMPONENTS

We organize a stack of L blocks composed of the layer normalization, feed-forward network, and self-attention modules. But their duties on the inverted dimension are carefully reconsidered. 

Layer normalization Layer normalization (Ba et al., 2016) is originally proposed to increase the convergence and training stability of deep networks. In typical Transformer-based forecasters, the module normalizes the multivariate representation of the same timestamp, gradually fusing the variates with each other. Once the collected time points do not represent the same event, the operation will also introduce interaction noises between noncausal or delayed processes. In our inverted version, the normalization is applied to the series representation of individual variate as Equation 2, which has been studied and proved effective in tackling non-stationary problems (Kim et al., 2021; Liu et al., 2022b). Besides, since all series as (variate) tokens are normalized to a Gaussian distribution, the discrepancies caused by inconsistent measurements can be diminished. By contrast, in previous architecture, different tokens of time steps will be normalized, leading to oversmooth time series. 

$$
\operatorname{LayerNorm} (\mathbf {H}) = \left\{\frac {\mathbf {h} _ {n} - \operatorname{Mean} (\mathbf {h} _ {n})}{\sqrt {\operatorname{Var} (\mathbf {h} _ {n})}} \Bigg | n = 1, \dots , N \right\}\tag{2}
$$

Feed-forward network Transformer adopts the feed-forward network (FFN) as the basic building block for encoding token representation and it is identically applied to each token. As aforementioned, in the vanilla Transformer, multiple variates of the same timestamp that form the token can be malpositioned and too localized to reveal enough information for predictions. In the inverted version, FFN is leveraged on the series representation of each variate token. By the universal approximation theorem (Hornik, 1991), they can extract complicated representations to describe a time series. With the stacking of inverted blocks, they are devoted to encoding the observed time series and decoding the representations for future series using dense non-linear connections, which work effectively as the recent works completely built on MLPs (Tolstikhin et al., 2021; Das et al., 2023). 

More interestingly, the identical linear operation on independent time series, which serves as the combination of the recent linear forecasters (Zeng et al., 2023) and Channel Independence (Nie et al., 2023), can be instructive for us to understand the series representations. Recent revisiting on linear forecasters (Li et al., 2023) highlights that temporal features extracted by MLPs are supposed to be shared within distinct time series. We propose a rational explanation that the neurons of MLP are taught to portray the intrinsic properties of any time series, such as the amplitude, periodicity, and even frequency spectrums (neuron as a filter), serving as a more advantageous predictive representation learner than the self-attention applied on time points. Experimentally, we validate that the division of labor helps enjoy the benefits of linear layers in Section 4.3, such as the promoted performance if providing enlarged lookback series, and the generalization ability on unseen variates. 

Self-attention While the attention mechanism is generally adopted for facilitating the temporal dependencies modeling in previous forecasters, the inverted model regards the whole series of one variate as an independent process. Concretely, with comprehensively extracted representations of each time series $\dot { \mathbf { H } } = \{ \mathbf { h } _ { 0 } , \dotsc , \mathbf { h } _ { N } \} \in \mathbb { R } ^ { N \times \check { D } }$ , the self-attention module adopts linear projections to get queries, keys, and values $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in \mathbb { R } ^ { N \times d _ { k } }$ , where $d _ { k }$ is the projected dimension. 

With denotation of $\mathbf { q } _ { i } , \mathbf { k } _ { j } \in \mathbb { R } ^ { d _ { k } }$ as the specific query and key of one (variate) token, we notice that each entry of the pre-Softmax scores is formulated as $\mathbf { A } _ { i , j } = ( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d _ { k } } ) _ { i , j } \propto \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j }$ . Since each token is previously normalized on its feature dimension, the entries can somewhat reveal the variate-wise correlation, and the whole score map $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ exhibits the multivariate correlations between paired variate tokens. Consequently, highly correlated variate will be more weighted for the next representation interaction with values V. Based on this intuition, the proposed mechanism is believed to be more natural and interpretable for multivariate series forecasting. We further provide the visualization analysis of the score map in Section 4.3 and Appendix E.1. 

## 4 EXPERIMENTS

We thoroughly evaluate the proposed iTransformer on various time series forecasting applications, validate the generality of the proposed framework and further dive into the effectiveness of applying the Transformer components on the inverted dimensions of time series. 

Datasets We extensively include 7 real-world datasets in our experiments, including ECL, ETT (4 subsets), Exchange, Traffic, Weather used by Autoformer (Wu et al., 2021), Solar-Energy datasets proposed in LSTNet (Lai et al., 2018), and PEMS (4 subsets) evaluated in SCINet (Liu et al., 2022a). We also provide the experiments on Market (6 subsets) in Appendix F.4. It records the minutesampled server load of Alipay online transaction application with hundreds of variates, where we consistently outperform other baselines. Detailed dataset descriptions are provided in Appendix A.1. 

## 4.1 FORECASTING RESULTS

In this section, we conduct extensive experiments to evaluate the forecasting performance of our proposed model together with advanced deep forecasters. 

Baselines We carefully choose 10 well-acknowledged forecasting models as our benchmark, including (1) Transformer-based methods: Autoformer (Wu et al., 2021), FEDformer (Zhou et al., 2022), Stationary (Liu et al., 2022b), Crossformer (Zhang & Yan, 2023), PatchTST (Nie et al., 2023); (2) Linear-based methods: DLinear (Zeng et al., 2023), TiDE (Das et al., 2023), RLinear (Li et al., 2023); and (3) TCN-based methods: SCINet (Liu et al., 2022a), TimesNet (Wu et al., 2023). 

Main results Comprehensive forecasting results are listed in Table 1 with the best in red and the second underlined. The lower MSE/MAE indicates the more accurate prediction result. Compared with other forecasters, iTransformer is particularly good at forecasting high-dimensional time series. Besides, PatchTST as the previous state-of-the-art, fails in many cases of PEMS, which can stem from the extremely fluctuating series of the dataset, and the patching mechanism of PatchTST may lose focus on specific locality to handle rapid fluctuation. By contrast, the proposed model aggregating the whole series variations for series representations can better cope with this situation. Notably, as the representative that explicitly captures multivariate correlations, the performance of Crossformer is still subpar to iTransformer, indicating the interaction of time-unaligned patches from different multivariate will bring about unnecessary noise for forecasting. Therefore, the native Transformer components are competent for temporal modeling and multivariate correlating, and the proposed inverted architecture can effectively tackle real-world time series forecasting scenarios. 


Table 1: Multivariate forecasting results with prediction lengths $S \in \{ 1 2 , 2 4 , 3 6 , 4 8 \}$ for PEMS and $S \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ for others and fixed lookback length $T = 9 6$ . Results are averaged from all prediction lengths. Avg means further averaged by subsets. Full results are listed in Appendix F.4.


<table><tr><td>Models</td><td colspan="2">iTransformer (Ours)</td><td colspan="2">RLinear (2023)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">TiDE (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">SCINet (2022a)</td><td colspan="2">FEDformer (2022)</td><td colspan="2">Stationary (2022b)</td><td colspan="2">Autoformer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ECL</td><td>0.178</td><td>0.270</td><td>0.219</td><td>0.298</td><td>0.205</td><td>0.290</td><td>0.244</td><td>0.334</td><td>0.251</td><td>0.344</td><td>0.192</td><td>0.295</td><td>0.212</td><td>0.300</td><td>0.268</td><td>0.365</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td></tr><tr><td>ETT (Avg)</td><td>0.383</td><td>0.399</td><td>0.380</td><td>0.392</td><td>0.381</td><td>0.397</td><td>0.685</td><td>0.578</td><td>0.482</td><td>0.470</td><td>0.391</td><td>0.404</td><td>0.442</td><td>0.444</td><td>0.689</td><td>0.597</td><td>0.408</td><td>0.428</td><td>0.471</td><td>0.464</td><td>0.465</td><td>0.459</td></tr><tr><td>Exchange</td><td>0.360</td><td>0.403</td><td>0.378</td><td>0.417</td><td>0.367</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.370</td><td>0.413</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.750</td><td>0.626</td><td>0.519</td><td>0.429</td><td>0.461</td><td>0.454</td><td>0.613</td><td>0.539</td></tr><tr><td>Traffic</td><td>0.428</td><td>0.282</td><td>0.626</td><td>0.378</td><td>0.481</td><td>0.304</td><td>0.550</td><td>0.304</td><td>0.760</td><td>0.473</td><td>0.620</td><td>0.336</td><td>0.625</td><td>0.383</td><td>0.804</td><td>0.509</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td></tr><tr><td>Weather</td><td>0.258</td><td>0.278</td><td>0.272</td><td>0.291</td><td>0.259</td><td>0.281</td><td>0.259</td><td>0.315</td><td>0.271</td><td>0.320</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.317</td><td>0.292</td><td>0.363</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td></tr><tr><td>Solar-Energy</td><td>0.233</td><td>0.262</td><td>0.369</td><td>0.356</td><td>0.270</td><td>0.307</td><td>0.641</td><td>0.639</td><td>0.347</td><td>0.417</td><td>0.301</td><td>0.319</td><td>0.330</td><td>0.401</td><td>0.282</td><td>0.375</td><td>0.291</td><td>0.381</td><td>0.261</td><td>0.381</td><td>0.885</td><td>0.711</td></tr><tr><td>PEMS (Avg)</td><td>0.119</td><td>0.218</td><td>0.514</td><td>0.482</td><td>0.217</td><td>0.305</td><td>0.220</td><td>0.304</td><td>0.375</td><td>0.440</td><td>0.148</td><td>0.246</td><td>0.320</td><td>0.394</td><td>0.121</td><td>0.222</td><td>0.224</td><td>0.327</td><td>0.151</td><td>0.249</td><td>0.614</td><td>0.575</td></tr></table>

## 4.2 ITRANSFORMERS GENERALITY

In this section, we evaluate iTransformers by applying our framework to Transformer and its variants, which generally address the quadratic complexity of the self-attention mechanism, including Reformer (Kitaev et al., 2020), Informer (Li et al., 2021), Flowformer (Wu et al., 2022) and FlashAt tention (Dao et al., 2022). Surprising and promising discoveries are exhibited, indicating the simple inverted perspective can enhance Transformer-based forecasters with promoted performance with efficiency, generalization on unseen variates, and better utilization of historical observations. 

Performance promotion We evaluate Transformers and the corresponding iTransformers with the reported performance promotions in Table 2. It is notable that the framework consistently improves various Transformers. Overall, it achieves averaged 38.9% promotion on Transformer, 36.1% on Reformer, 28.5% on Informer, 16.8% on Flowformer and 32.2% on Flashformer, revealing the previous improper usage of the Transformer architecture on time series forecasting. Moreover, since the attention mechanism is adopted on the variate dimension in our inverted structure, the introduction of efficient attentions with linear complexity essentially addresses the computational problem due to numerous variates, which is prevalent in real-world applications but can be resource-consuming for Channel Independence (Nie et al., 2023). Therefore, the idea of iTransformer can be widely practiced on Transformer-based forecasters to take advantage of booming efficient attention mechanisms. 


Table 2: Performance promotion obtained by our inverted framework. Flashformer means Transformer equipped with hardware-accelerated FlashAttention (Dao et al., 2022). We report the average performance and the relative MSE reduction (Promotion). Full results can be found in Appendix F.2.


<table><tr><td colspan="2">Models</td><td colspan="2">Transformer (2017)</td><td colspan="2">Reformer (2020)</td><td colspan="2">Informer (2021)</td><td colspan="2">Flowformer (2022)</td><td colspan="2">Flashformer (2022)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">ECL</td><td>Original</td><td>0.277</td><td>0.372</td><td>0.338</td><td>0.422</td><td>0.311</td><td>0.397</td><td>0.267</td><td>0.359</td><td>0.285</td><td>0.377</td></tr><tr><td>+Inverted</td><td>0.178</td><td>0.270</td><td>0.208</td><td>0.301</td><td>0.216</td><td>0.311</td><td>0.210</td><td>0.293</td><td>0.206</td><td>0.291</td></tr><tr><td>Promotion</td><td>35.6%</td><td>27.4%</td><td>38.4%</td><td>28.7%</td><td>30.5%</td><td>21.6%</td><td>21.3%</td><td>18.6%</td><td>27.8%</td><td>22.9%</td></tr><tr><td rowspan="3">Traffic</td><td>Original</td><td>0.665</td><td>0.363</td><td>0.741</td><td>0.422</td><td>0.764</td><td>0.416</td><td>0.750</td><td>0.421</td><td>0.658</td><td>0.356</td></tr><tr><td>+Inverted</td><td>0.428</td><td>0.282</td><td>0.647</td><td>0.370</td><td>0.662</td><td>0.380</td><td>0.524</td><td>0.355</td><td>0.492</td><td>0.333</td></tr><tr><td>Promotion</td><td>35.6%</td><td>22.3%</td><td>12.7%</td><td>12.3%</td><td>13.3%</td><td>8.6%</td><td>30.1%</td><td>15.6%</td><td>25.2%</td><td>6.4%</td></tr><tr><td rowspan="3">Weather</td><td>Original</td><td>0.657</td><td>0.572</td><td>0.803</td><td>0.656</td><td>0.634</td><td>0.548</td><td>0.286</td><td>0.308</td><td>0.659</td><td>0.574</td></tr><tr><td>+Inverted</td><td>0.258</td><td>0.279</td><td>0.248</td><td>0.292</td><td>0.271</td><td>0.330</td><td>0.266</td><td>0.285</td><td>0.262</td><td>0.282</td></tr><tr><td>Promotion</td><td>60.2%</td><td>50.8%</td><td>69.2%</td><td>55.5%</td><td>57.3%</td><td>39.8%</td><td>7.2%</td><td>7.7%</td><td>60.2%</td><td>50.8%</td></tr></table>

Variate generalization By inverting vanilla Transformers, it is notable that the models are empowered with the generalization capability on unseen variates. Firstly, benefiting from the flexibility of the number of input tokens, the amount of variate channels is no longer restricted and thus feasible to vary from training and inference. Besides, feed-forward networks are identically applied on independent variate tokens in iTransformer. As aforementioned, the neurons as filters learn the intrinsic patterns of any time series, which are inclined to be shared and transferable among distinct variates. 

To verify the hypothesis, we compare inverting with another generalizing strategy: Channel Independence, training a shared backbone to forecast all variates. We partition the variates of each dataset into five folders, train models with only 20% of variates of one folder, and directly forecast all variates without fine-tuning. We compare the performance in Figure 5 and each bar presents the averaged results of all folders to avoid the randomness of partition. CI-Transformers take a long time to predict each variate one by one during inference while iTransformers directly predict all variates and generally present smaller increases, indicating FFN is competent to learn transferable time series representations. It leaves a potential direction to build a foundation model upon iTransformer, where diverse multivariate time series with different numbers of variates can be feasibly trained together. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/52cc9f48f0a37c2999a242c338543431ad76a094e28af0b966d85402714bb845.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/31caf24d52530617bca20863c159606375071e621b42e81182ce6aca671b006d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/24210c7a15e6c9d792c9e6dfe34dda36ed373fdf2a651621ec35177363766bca.jpg)



Figure 5: Performance of generalization on unseen variates. We partition the variates of each dataset into five folders, train models with 20% variates, and use the partially trained model to forecast all varieties. iTransformers can be trained efficiently and forecast with good generalizability.


Increasing lookback length Previous works have witnessed the phenomenon that the forecasting performance does not necessarily improve with the increase of lookback length on Transformers (Nie et al., 2023; Zeng et al., 2023), which can be attributed to the distracted attention on the growing input. However, the desired performance improvement is generally held on linear forecasts, theoretically supported by statistical methods (Box & Jenkins, 1968) with enlarged historical information to be utilized. As the working dimensions of attention and feed-forward network are inverted, we evaluate the performance of Transformers and iTransformer in Figure 6 with increased lookback length. The results surprisingly verify the rationality of leveraging MLPs on the temporal dimension such that Transformers can benefit from the extended lookback window for more precise predictions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/9a38ad4908a5a82b42af78ba51b1ea270a9b8310dbc6641f89a4149a74c74367.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/37002f64b6f3f48e886053d27b4c3899af3052f09364d60668d530cecddb1738.jpg)



Figure 6: Forecasting performance with the lookback length $T \in \{ 4 8 , 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ and fixed prediction length $S ~ = ~ 9 6$ . While the performance of Transformer-based forecasters does not necessarily benefit from the increased lookback length, the inverted framework empowers the vanilla Transformer and its variants with improved performance on the enlarged lookback window.


## 4.3 MODEL ANALYSIS

Ablation study To verify the rational business of Transformer components, we provide detailed ablations covering both replacing components (Replace) and removing components (w/o) experiments. The results are listed in Table 3. iTransformer that utilizes attention on the variate dimension and feed-forward on the temporal dimension generally achieves the best performance. Notably, the performance of vanilla Transformer (the third row) performs the worst among these designs, revealing the potential risks of the conventional architecture, which we describe in detail in Appendix E.3. 


Table 3: Ablations on iTransformer. We replace different components on the respective dimension to learn multivariate correlations (Variate) and series representations (Temporal), in addition to component removal. The average results of all predicted lengths are listed here.


<table><tr><td rowspan="2">Design</td><td rowspan="2">Variate</td><td rowspan="2">Temporal</td><td colspan="2">ECL</td><td colspan="2">Traffic</td><td colspan="2">Weather</td><td colspan="2">Solar-Energy</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>iTransformer</td><td>Attention</td><td>FFN</td><td>0.178</td><td>0.270</td><td>0.428</td><td>0.282</td><td>0.258</td><td>0.278</td><td>0.233</td><td>0.262</td></tr><tr><td rowspan="3">Replace</td><td>Attention</td><td>Attention</td><td>0.193</td><td>0.293</td><td>0.913</td><td>0.500</td><td>0.255</td><td>0.280</td><td>0.261</td><td>0.291</td></tr><tr><td>FFN</td><td>Attention</td><td>0.202</td><td>0.300</td><td>0.863</td><td>0.499</td><td>0.258</td><td>0.283</td><td>0.285</td><td>0.317</td></tr><tr><td>FFN</td><td>FFN</td><td>0.182</td><td>0.287</td><td>0.599</td><td>0.348</td><td>0.248</td><td>0.274</td><td>0.269</td><td>0.287</td></tr><tr><td rowspan="2">w/o</td><td>Attention</td><td>w/o</td><td>0.189</td><td>0.278</td><td>0.456</td><td>0.306</td><td>0.261</td><td>0.281</td><td>0.258</td><td>0.289</td></tr><tr><td>w/o</td><td>FFN</td><td>0.193</td><td>0.276</td><td>0.461</td><td>0.294</td><td>0.265</td><td>0.283</td><td>0.261</td><td>0.283</td></tr></table>

Analysis of series representations To further validate the claim that feed-forward networks are more favored to extract the series representations. We conduct representation analysis based on the centered kernel alignment (CKA) similarity (Kornblith et al., 2019). A higher CKA indicates more similar representations. For Transformer variants and iTransformers, we calculate the CKA between the output features of the first and the last block. Notably, previous works have demonstrated that time series forecasting, as a low-level generative task, prefers the higher CKA similarity (Wu et al., 2023; Dong et al., 2023) for the better performance. As shown in Figure 7, a clear division line is exhibited, implying that iTransformers have learned more appropriate series representations by inverting the dimension and thus achieve more accurate predictions. The results also advocate inverting Transformer deserves a fundamental renovation of the forecasting backbone. 

Analysis of multivariate correlations By assigning the duty of multivariate correlation to the attention mechanism, the learned map enjoys enhanced interpretability. We present the case visualization on series from Solar-Energy in Figure 7, which has distinct correlations in the lookback and future windows. It can be observed that in the shallow attention layer, the learned map shares lots of similarities to the correlations of raw input series. As it dives into deeper layers, the learned map become gradually alike to the correlations of future series, which validates the inverted operation empowers interpretable attention for correlating, and the processes of encoding the past and decoding for the future are essentially conducted in series representations during feed-forwarding. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/8b363ebe1e63fbe6d1a7b74fc2a618f866010578fd2545fccc2ef474e6d8b0f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/be4e21086af17f966852d63def99feb02fa01034a59b1069f5f0658b0b97acab.jpg)



Figure 7: Analysis of series representations and multivariate correlations. Left: MSE and CKA similarity of representations comparison between Transformers and iTransformers. A higher CKA similarity indicates more favored representations for accurate predictions. Right: A case visualization of multivariate correlations of raw time series and the learned score maps by inverted self-attention.


Efficient training strategy Due to the quadratic complexity of self-attention, it can be overwhelming for training on numerous variates, which is very common in real-world scenarios. In addition to efficient attention mechanisms, we propose a novel training strategy for high-dimensional multivariate series by taking advantage of previously demonstrated variate generation capability. Concretely, we randomly choose part of the variates in each batch and only train the model with selected variates. Since the number of variate channels is flexible because of our inverting, the model can predict all the variates for predictions. As shown in Figure 8, the performance of our proposed strategy is still comparable with full-variate training, while the memory footprint can be reduced significantly. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/2096c2dd843075a1cb66f52e75c74548767c81324ca014d1fc2a1334080ff6be.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/bb8eac0ecfbf9a4df28f7bd7172c2c84714d177103a0aee3c6261db34c15a29a.jpg)



Figure 8: Analysis of the efficient training strategy. While the performance (left) remains stable on partially trained variates of each batch with different sampled ratios, the memory footprint (right) can be cut off greatly. We provide the comprehensive model efficiency analysis in Appendix D.


## 5 CONCLUSION AND FUTURE WORK

Considering the characteristics of multivariate time series, we propose iTransformer that inverts the structure of Transformer without modifying any native modules. iTransformer regards independent series as variate tokens to capture multivariate correlations by attention and utilize layer normalization and feed-forward networks to learn series representations. Experimentally, iTransformer achieves state-of-the-art performance and exhibits remarkable framework generality supported by promising analysis. In the future, we will explore large-scale pre-training and more time series analysis tasks. 

## 6 ETHICS STATEMENT

Our work only focuses on the time series forecasting problem, so there is no potential ethical risk. 

## 7 REPRODUCIBILITY STATEMENT

In the main text, we have strictly formalized the model architecture with equations. All the implementation details are included in the Appendix, including dataset descriptions, metrics, model, and experiment configurations. The code will be made public once the paper is accepted. 

## ACKNOWLEDGMENTS

This work was supported by the National Key Research and Development Plan (2021YFB1715200), the National Natural Science Foundation of China (U2342217 and 62022050), the BNRist Innovation Fund (BNR2024RC01010), Ant Group through CCF-Ant Research Fund, and the National Engineering Research Center for Big Data Software. 

## REFERENCES



Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. https://arxiv.org/pdf/1607.06450.pdf, 2016. 





Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2, 2018. 





George EP Box and Gwilym M Jenkins. Some recent advances in forecasting and control. Journal of the Royal Statistical Society. Series C (Applied Statistics), 17(2):91–109, 1968. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 





Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-´ efficient exact attention with io-awareness. NeurIPS, 2022. 





Abhimanyu Das, Weihao Kong, Andrew Leach, Rajat Sen, and Rose Yu. Long-term forecasting with tide: Time-series dense encoder. arXiv preprint arXiv:2304.08424, 2023. 





Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, and Mingsheng Long. Simmtm: A simple pre-training framework for masked time-series modeling. arXiv preprint arXiv:2302.00861, 2023. 





Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 





Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. KDD, 2023. 





Lu Han, Han-Jia Ye, and De-Chuan Zhan. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. arXiv preprint arXiv:2304.05206, 2023. 





Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2): 251–257, 1991. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 





Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. ICLR, 2021. 





Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015. 





Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. ICLR, 2020. 





Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. Similarity of neural network representations revisited. ICML, 2019. 





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. SIGIR, 2018. 





Jianxin Li, Xiong Hui, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. arXiv: 2012.07436, 2021. 





Zhe Li, Shiyi Qi, Yiduo Li, and Zenglin Xu. Revisiting long-term time series forecasting: An investigation on linear mapping. arXiv preprint arXiv:2305.10721, 2023. 





Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: time series modeling and forecasting with sample convolution and interaction. NeurIPS, 2022a. 





Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Rethinking the stationarity in time series forecasting. NeurIPS, 2022b. 





Yong Liu, Chenyu Li, Jianmin Wang, and Mingsheng Long. Koopa: Learning non-stationary time series dynamics with koopman predictors. arXiv preprint arXiv:2305.18803, 2023. 





Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. ICLR, 2023. 





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR, 2019. 





Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito,¨ Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019. 





Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. NeurIPS, 2018. 





David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal ofForecasting, 36(3): 1181–1191, 2020. 





Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlp-mixer: An all-mlp architecture for vision. NeurIPS, 2021. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. NeurIPS, 2021. 





Haixu Wu, Jialong Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Flowformer: Linearizing transformers with conservation flows. ICML, 2022. 





Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. ICLR, 2023. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? AAAI, 2023. 





Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. ICLR, 2023. 





Zheng Zhao, Weihai Chen, Xingming Wu, Peter CY Chen, and Jingmeng Liu. Lstm network: a deep learning approach for short-term traffic forecast. IET Intelligent Transport Systems, 11(2):68–75, 2017. 





Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. ICML, 2022. 



## A IMPLEMENTATION DETAILS

## A.1 DATASET DESCRIPTIONS

We conduct experiments on 7 real-world datasets to evaluate the performance of the proposed iTransformer including (1) ETT (Li et al., 2021) contains 7 factors of electricity transformer from July 2016 to July 2018. There are four subsets where ETTh1 and ETTh2 are recorded every hour, and ETTm1 and ETTm2 are recorded every 15 minutes. (2) Exchange (Wu et al., 2021) collects the panel data of daily exchange rates from 8 countries from 1990 to 2016. (3) Weather (Wu et al., 2021) includes 21 meteorological factors collected every 10 minutes from the Weather Station of the Max Planck Biogeochemistry Institute in 2020. (4) ECL (Wu et al., 2021) records the hourly electricity consumption data of 321 clients. (5) Traffic (Wu et al., 2021) collects hourly road occupancy rates measured by 862 sensors of San Francisco Bay area freeways from January 2015 to December 2016. (6) Solar-Energy (Lai et al., 2018) records the solar power production of 137 PV plants in 2006, which are sampled every 10 minutes. (7) PEMS contains the public traffic network data in California collected by 5-minute windows. We use the same four public subsets (PEMS03, PEMS04, PEMS07, PEMS08) adopted in SCINet (Liu et al., 2022a). 

Apart from the public datasets widely used as forecasting benchmarks, we also collect a set of Market datasets of a real-world application, which records the minute-sampled server load of Alipay online transactions between January 30th, 2023, and April 9th, 2023 with the number of variates varied from 285 to 759. It includes 6 sub-datasets, which are divided according to diverse transaction domains. 

We follow the same data processing and train-validation-test set split protocol used in TimesNet (Wu et al., 2023), where the train, validation, and test datasets are strictly divided according to chronological order to make sure there are no data leakage issues. As for the forecasting settings, we fix the length of the lookback series as 96 in ETT, Weather, ECL, Solar-Energy, PEMS, and Traffic, and the prediction length varies in {96, 192, 336, 720}. For the PEMS dataset, the prediction length varies in {12, 24, 36, 48}, which is the same as SCINet, the previous state-of-the-art on this dataset. For the Market dataset, the lookback contains the past one day observations with 144 time points and the forecasting length varies in {12, 24, 72, 144}. The details of datasets are provided in Table 4. 

## A.2 IMPLEMENTATION DETAILS

Algorithm 1 iTransformer - Overall Architecture.

Require: Input lookback time series $X \in R^{T \times N}$ ; input Length T; predicted length S; variates number N; token dimension D; iTransformer block number L.

1: X = X.transpose ▷ $X \in R^{N \times T}$ 2: ▷ Multi-layer Perceptron works on the last dimension to embed series into variate tokens.

3: $H^{0} = \text{MLP}(X)$ ▷ $H^{0} \in R^{N \times D}$ 4: for l in $\{1, \ldots, L\}$ : ▷ Run through iTransformer blocks.

5: ▷ Self-attention layer is applied on variate tokens.

6: $H^{l-1} = \text{LayerNorm}(H^{l-1} + \text{Self-Attn}(H^{l-1}))$ ▷ $H^{l-1} \in R^{N \times D}$ 7: ▷ Feed-forward network is utilized for series representations, broadcasting to each token.

8: $H^{l} = \text{LayerNorm}(H^{l-1} + \text{Feed-Forward}(H^{l-1}))$ ▷ $H^{l} \in R^{N \times D}$ 9: ▷ LayerNorm is adopted on series representations to reduce variates discrepancies.

10: End for

11: $\hat{Y} = \text{MLP}(H^{L})$ ▷ Project tokens back to predicted series, $\hat{Y} \in R^{N \times S}$ 12: $\hat{Y} = \hat{Y}.transpose$ ▷ $\hat{Y} \in R^{S \times N}$ 13: Return $\hat{Y}$ ▷ Return the prediction result $\hat{Y}$ 


Table 4: Detailed dataset descriptions. Dim denotes the variate number of each dataset. Dataset Size denotes the total number of time points in (Train, Validation, Test) split respectively. Prediction Length denotes the future time points to be predicted and four prediction settings are included in each dataset. Frequency denotes the sampling interval of time points.


<table><tr><td>Dataset</td><td>Dim</td><td>Prediction Length</td><td>Dataset Size</td><td>Frequency</td><td>Information</td></tr><tr><td>ETTh1, ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Hourly</td><td>Electricity</td></tr><tr><td>ETTm1, ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>Electricity</td></tr><tr><td>Exchange</td><td>8</td><td>{96, 192, 336, 720}</td><td>(5120, 665, 1422)</td><td>Daily</td><td>Economy</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10min</td><td>Weather</td></tr><tr><td>ECL</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Hourly</td><td>Electricity</td></tr><tr><td>Traffic</td><td>862</td><td>{96, 192, 336, 720}</td><td>(12185, 1757, 3509)</td><td>Hourly</td><td>Transportation</td></tr><tr><td>Solar-Energy</td><td>137</td><td>{96, 192, 336, 720}</td><td>(36601, 5161, 10417)</td><td>10min</td><td>Energy</td></tr><tr><td>PEMS03</td><td>358</td><td>{12, 24, 48, 96}</td><td>(15617, 5135, 5135)</td><td>5min</td><td>Transportation</td></tr><tr><td>PEMS04</td><td>307</td><td>{12, 24, 48, 96}</td><td>(10172, 3375, 3375)</td><td>5min</td><td>Transportation</td></tr><tr><td>PEMS07</td><td>883</td><td>{12, 24, 48, 96}</td><td>(16911, 5622, 5622)</td><td>5min</td><td>Transportation</td></tr><tr><td>PEMS08</td><td>170</td><td>{12, 24, 48, 96}</td><td>(10690, 3548, 3548)</td><td>5min</td><td>Transportation</td></tr><tr><td>Market-Merchant</td><td>285</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr><tr><td>Market-Wealth</td><td>485</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr><tr><td>Market-Finance</td><td>405</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr><tr><td>Market-Terminal</td><td>307</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr><tr><td>Market-Payment</td><td>759</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr><tr><td>Market-Customer</td><td>395</td><td>{12, 24, 72, 144}</td><td>(7045, 1429, 1429)</td><td>10min</td><td>Transaction</td></tr></table>

All the experiments are implemented in PyTorch (Paszke et al., 2019) and conducted on a single NVIDIA P100 16GB GPU. We utilize ADAM (Kingma & Ba, 2015) with an initial learning rate in $\{ 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } \}$ and L2 loss for the model optimization. The batch size is uniformly set to 32 and the number of training epochs is fixed to 10. We set the number of inverted Transformer blocks in our proposed model $L \in { \bar { \{ 2 , 3 , 4 \} } }$ . The dimension of series representations D is set from {256, 512}. All the compared baseline models that we reproduced are implemented based on the benchmark of TimesNet (Wu et al., 2023) Repository, which is fairly built on the configurations provided by each model’s original paper or official code. We provide the pseudo-code of iTransformer in Algorithm 1. We also report the standard deviation of iTransformer performance under five runs with different random seeds in Table 5, which exhibits that the performance of iTransformer is stable. 


Table 5: Robustness of iTransformer performance. The results are obtained from five random seeds.


<table><tr><td rowspan="2">Dataset Horizon</td><td colspan="2">ECL</td><td colspan="2">ETTh2</td><td colspan="2">Exchange</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td>0.148±0.000</td><td>0.240±0.000</td><td>0.297±0.002</td><td>0.349±0.001</td><td>0.088±0.001</td><td>0.209±0.001</td></tr><tr><td>192</td><td>0.162±0.002</td><td>0.253±0.002</td><td>0.380±0.001</td><td>0.400±0.001</td><td>0.181±0.001</td><td>0.304±0.001</td></tr><tr><td>336</td><td>0.178±0.000</td><td>0.269±0.001</td><td>0.428±0.002</td><td>0.432±0.001</td><td>0.334±0.001</td><td>0.419±0.001</td></tr><tr><td>720</td><td>0.225±0.006</td><td>0.317±0.007</td><td>0.427±0.004</td><td>0.445±0.002</td><td>0.829±0.012</td><td>0.691±0.005</td></tr><tr><td rowspan="2">Dataset Horizon</td><td colspan="2">Solar-Energy</td><td colspan="2">Traffic</td><td colspan="2">Weather</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td>0.203±0.002</td><td>0.237±0.002</td><td>0.395±0.001</td><td>0.268±0.001</td><td>0.174±0.000</td><td>0.214±0.000</td></tr><tr><td>192</td><td>0.233±0.002</td><td>0.261±0.001</td><td>0.417±0.002</td><td>0.276±0.001</td><td>0.221±0.002</td><td>0.254±0.001</td></tr><tr><td>336</td><td>0.248±0.000</td><td>0.273±0.000</td><td>0.433±0.004</td><td>0.283±0.000</td><td>0.278±0.002</td><td>0.296±0.001</td></tr><tr><td>720</td><td>0.249±0.001</td><td>0.275±0.000</td><td>0.467±0.003</td><td>0.302±0.000</td><td>0.358±0.000</td><td>0.349±0.000</td></tr></table>

## B ABLATION STUDIES

To elaborate on the rational business of Transformer components, we conduct detailed ablations covering replacing components (Replace) and removing components (w/o). Since the average results are listed in Table 3 due to the paper limit, we provide detailed results and analysis here. 

As shown in Table 6, among various architectural designs, iTransformer generally exhibits superior performance, which learns multivariate correlations by self-attention and encodes series representations by FFN. Nevertheless, the arrangement of the vanilla Transformer can lead to degenerated performance, indicating the misuse of Transformer components on the time series modality. Based on the relatively poor results of the second (both attentions) and the third (the vanilla Transformer) designs, one of the reasons for that may lie in the attention module over the temporal tokens of the lagged time series, which we elaborate more with the datasets support in Section E.3. 

It is also notable that applying FFN on both dimensions can also lead to fair performance on datasets with small variate numbers (such as Weather with 21 variates). Still, with the increasing of variate numbers in challenging multivariate forecasting tasks, the importance of capturing multivariate corre lations is ever more highlighted. We note that the heterogeneity of variates can be hardly considered by the vanilla Transformer. During embedding, the variates are projected into indistinguishable channels, which ignores the inconsistent physical measurements and thus fails to maintain the independence of variates, let alone capture and utilize the multivariate correlation. Consequently, by incorporating the advanced attention module for the variate correlating, the first (iTransformer) and the fifth (attention on variates) designs perform more effectively in challenging multivariate datasets. 

In a nutshell, both temporal dependencies and multivariate correlations are of importance for multivariate time series forecasting. The proposed iTransformer employing the self-attention module to disentangle the correlations between variate tokens proves to be more powerful and interpretable than feed-forward networks, thereby further boosting the performance on challenging multivariate datasets and enhancing the model capacity. 

## C HYPERPARAMETER SENSITIVITY

We evaluate the hyperparameter sensitivity of iTransformer with respect to the following factors: the learning rate $l r ,$ the number of Transformer blocks $L ,$ and the hidden dimension D of variate tokens. The results are shown in Figure 9. We find that the learning rate, as the most common influencing factor, should be carefully selected when the number of variates is large (ECL, Traffic). The block number and hidden dimension are not essentially favored to be as large as possible in iTransformer. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/a4e1bbd5f5a15300f1b90f9afbc5b0708e3dfc7be2420238eca8dc41c9479c1d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/35d3050b6b70b85892e1a41bea11e689c012b5f1822b19ba2992745c763d175d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/c977587b76e2f8252ded0668a2a6af31defbcf39637e5d17796f11bceabbc540.jpg)



Figure 9: Hyperparameter sensitivity with respect to the learning rate, the number of Transformer blocks, and the hidden dimension of variate tokens. The results are recorded with the lookback window length $T = 9 6$ and the forecast window length $S = 9 6$


## D MODEL EFFICIENCY

We comprehensively compare the forecasting performance, training speed, and memory footprint of the following models: iTransformer, iTransformer with our efficient training strategy and iTransformer with the efficient flow attention module (Wu et al., 2022); linear models: DLinear (Zeng et al., 2023) and TiDE (Das et al., 2023); Transformers: Transformer (Vaswani et al., 2017), PatchTST (Nie et al., 2023), and Crossformer (Zhang & Yan, 2023). The results are recorded with the official model configuration and the same batch size. In Figure 10, we compare the efficiency under two representative datasets (21 variates in Weather and 862 in Traffic) with 96 time steps for lookback. 


Table 6: Full results of the ablation on iTransformer. We apply different components on the respective dimension to learn multivariate correlations (Variate) and series representations (Temporal), in addition to removing the specific component of Transformer.


<table><tr><td rowspan="2">Design</td><td rowspan="2">Variate</td><td rowspan="2">Temporal</td><td>Prediction</td><td colspan="2">ECL</td><td colspan="2">Traffic</td><td colspan="2">Weather</td><td colspan="2">Solar-Energy</td></tr><tr><td>Lengths</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">iTransformer</td><td rowspan="4">Attention</td><td rowspan="4">FFN</td><td>96</td><td>0.148</td><td>0.240</td><td>0.395</td><td>0.268</td><td>0.174</td><td>0.214</td><td>0.203</td><td>0.237</td></tr><tr><td>192</td><td>0.162</td><td>0.253</td><td>0.417</td><td>0.276</td><td>0.221</td><td>0.254</td><td>0.233</td><td>0.261</td></tr><tr><td>336</td><td>0.178</td><td>0.269</td><td>0.433</td><td>0.283</td><td>0.278</td><td>0.296</td><td>0.248</td><td>0.273</td></tr><tr><td>720</td><td>0.225</td><td>0.317</td><td>0.467</td><td>0.302</td><td>0.358</td><td>0.349</td><td>0.249</td><td>0.275</td></tr><tr><td></td><td></td><td>Avg</td><td>0.178</td><td>0.270</td><td>0.428</td><td>0.282</td><td>0.258</td><td>0.279</td><td>0.233</td><td>0.262</td></tr><tr><td rowspan="15">Replace</td><td rowspan="4">Attention</td><td rowspan="4">Attention</td><td>96</td><td>0.161</td><td>0.263</td><td>1.021</td><td>0.581</td><td>0.168</td><td>0.213</td><td>0.227</td><td>0.270</td></tr><tr><td>192</td><td>0.180</td><td>0.280</td><td>0.834</td><td>0.447</td><td>0.217</td><td>0.256</td><td>0.255</td><td>0.292</td></tr><tr><td>336</td><td>0.194</td><td>0.296</td><td>0.906</td><td>0.493</td><td>0.277</td><td>0.299</td><td>0.279</td><td>0.301</td></tr><tr><td>720</td><td>0.238</td><td>0.331</td><td>0.892</td><td>0.477</td><td>0.356</td><td>0.351</td><td>0.283</td><td>0.300</td></tr><tr><td></td><td></td><td>Avg</td><td>0.193</td><td>0.293</td><td>0.913</td><td>0.500</td><td>0.255</td><td>0.280</td><td>0.261</td><td>0.291</td></tr><tr><td rowspan="4">FFN</td><td rowspan="4">Attention</td><td>96</td><td>0.169</td><td>0.270</td><td>0.907</td><td>0.540</td><td>0.176</td><td>0.221</td><td>0.247</td><td>0.299</td></tr><tr><td>192</td><td>0.189</td><td>0.292</td><td>0.839</td><td>0.489</td><td>0.224</td><td>0.261</td><td>0.275</td><td>0.305</td></tr><tr><td>336</td><td>0.204</td><td>0.304</td><td>0.248</td><td>0.364</td><td>0.279</td><td>0.301</td><td>0.317</td><td>0.337</td></tr><tr><td>720</td><td>0.245</td><td>0.335</td><td>1.059</td><td>0.606</td><td>0.354</td><td>0.347</td><td>0.301</td><td>0.329</td></tr><tr><td></td><td></td><td>Avg</td><td>0.202</td><td>0.300</td><td>0.863</td><td>0.499</td><td>0.258</td><td>0.283</td><td>0.285</td><td>0.317</td></tr><tr><td rowspan="4">FFN</td><td rowspan="4">FFN</td><td>96</td><td>0.159</td><td>0.261</td><td>0.606</td><td>0.342</td><td>0.162</td><td>0.207</td><td>0.237</td><td>0.277</td></tr><tr><td>192</td><td>0.171</td><td>0.271</td><td>0.559</td><td>0.342</td><td>0.211</td><td>0.252</td><td>0.273</td><td>0.293</td></tr><tr><td>336</td><td>0.187</td><td>0.287</td><td>0.569</td><td>0.348</td><td>0.270</td><td>0.293</td><td>0.284</td><td>0.287</td></tr><tr><td>720</td><td>0.211</td><td>0.307</td><td>0.664</td><td>0.359</td><td>0.349</td><td>0.345</td><td>0.284</td><td>0.289</td></tr><tr><td></td><td></td><td>Avg</td><td>0.182</td><td>0.287</td><td>0.599</td><td>0.348</td><td>0.248</td><td>0.274</td><td>0.269</td><td>0.287</td></tr><tr><td rowspan="10">w/o</td><td rowspan="4">Attention</td><td rowspan="4">w/o</td><td>96</td><td>0.163</td><td>0.254</td><td>0.427</td><td>0.296</td><td>0.177</td><td>0.219</td><td>0.226</td><td>0.266</td></tr><tr><td>192</td><td>0.174</td><td>0.263</td><td>0.446</td><td>0.300</td><td>0.226</td><td>0.259</td><td>0.255</td><td>0.288</td></tr><tr><td>336</td><td>0.191</td><td>0.280</td><td>0.459</td><td>0.306</td><td>0.281</td><td>0.298</td><td>0.275</td><td>0.301</td></tr><tr><td>720</td><td>0.228</td><td>0.315</td><td>0.492</td><td>0.324</td><td>0.359</td><td>0.249</td><td>0.275</td><td>0.301</td></tr><tr><td></td><td></td><td>Avg</td><td>0.189</td><td>0.278</td><td>0.456</td><td>0.306</td><td>0.261</td><td>0.281</td><td>0.258</td><td>0.289</td></tr><tr><td rowspan="4">w/o</td><td rowspan="4">FFN</td><td>96</td><td>0.169</td><td>0.253</td><td>0.437</td><td>0.283</td><td>0.183</td><td>0.220</td><td>0.228</td><td>0.263</td></tr><tr><td>192</td><td>0.177</td><td>0.261</td><td>0.449</td><td>0.287</td><td>0.231</td><td>0.262</td><td>0.261</td><td>0.283</td></tr><tr><td>336</td><td>0.194</td><td>0.278</td><td>0.464</td><td>0.294</td><td>0.285</td><td>0.300</td><td>0.279</td><td>0.294</td></tr><tr><td>720</td><td>0.233</td><td>0.311</td><td>0.496</td><td>0.313</td><td>0.362</td><td>0.350</td><td>0.276</td><td>0.291</td></tr><tr><td></td><td></td><td>Avg</td><td>0.193</td><td>0.276</td><td>0.461</td><td>0.294</td><td>0.265</td><td>0.283</td><td>0.261</td><td>0.283</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/0f3d5084594560ff93961c34fed25d4876ab80eeb2469c8e4e08c5e3da8518d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/eef904d36aff5edf7017a94494c5c85ca22237697982cc7c49bd6a46539b785e.jpg)



Figure 10: Model efficiency comparison under input-96-predict-96 of Weather and Traffic.


In a nutshell, the efficiency of iTransformer exceeds other Transformers in datasets with a relatively small number of variates (Weather). In datasets with numerous variates (Traffic), the memory footprints are basically the same as Transformers variates, but iTransformer can be trained faster. Based on the complexity of $\mathcal { O } ( N ^ { 2 } )$ of the attention module, where N is the number of tokens, Transformer surpasses iTransformer on efficiency in this case because of $N = 9 6$ for the temporal token and $N = { 8 6 2 }$ for the variate token. Meanwhile, iTransformer achieves better performance on numerous variates, since the multivariate correlations can be explicitly utilized. By adopting a linear-complexity attention (Wu et al., 2022) or the proposed efficient training strategy as mentioned in Figure 8 (trained on 20% variates and forecast all variates), iTransformer can enjoy a comparable speed and memory footprint with linear models. Also, the two strategies can be adopted together. 

## E SHOWCASES

## E.1 VISUALIZATION OF MULTIVARIATE CORRELATIONS

By using the attention mechanism on variate tokens, the resulting learned map becomes more interpretable. To present an intuitive understanding of the multivariate correlations, we provide three randomly chosen case visualizations of the time series from Solar-Energy in Figure 11. We provide the Pearson Correlation coefficients of each variate of the raw series by the following equation: 

$$
\rho_ {x y} = \frac {\sum_ {i} (x _ {i} - \bar {x}) (y _ {i} - \bar {y})}{\sqrt {\sum_ {i} (x _ {i} - \bar {x}) ^ {2}} \sqrt {\sum_ {i} (y _ {i} - \bar {y}) ^ {2}}},
$$

where $x _ { i } , y _ { i } \in \mathbb { R }$ run through all time points of the paired variates to be correlated. All the cases have distinct multivariate correlations in the lookback and forecast window because the dataset exhibits obvious seasonal changes in the daytime and night. On the second row of each case, we provide the learned pre-Softmax maps of the self-attention module in both the first and the last layers. As we observe in the shallow attention layer (left), we find that the learned map is similar to the correlations of the raw lookback series. As we go deeper into the layers (right), the learned map gradually becomes more similar to the correlations of the future series to be predicted. This demonstrates that the inverted operation allows for interpretable attention in correlating, and that encoding of the past and decoding for the future are conducted through series representations during layer stacking. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/41af7f230c7d8561ef0040621d036e99e9b4a4ae95849f8323660bff813c2e28.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/135a876319ecf3a107bf7d5ccb42ef605de81d5b3f0090857b4d187262d6a9c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/677c5d0f7918da3066f4ab07c89079af63def74dc6b84cc8a68851237774fdd4.jpg)



Figure 11: Multivariate correlations of the lookback series and future series and the learned score maps by inverted self-attention of different layers. Cases all come from the Solar-Energy dataset.


We present another interesting observation in Figure 12 to show that the attention module of iTransformer has enhanced interpretability. We provide randomly chosen multivariate time series from Market. In this dataset, each variate represents the monitored values of a service interface of a kind, and the service can be further grouped into refined application categories. We divide these variates into corresponding applications (as listed on the top bar App), such that adjacent variates belong to the same application and we reveal the application index by the top bar. 

We visualize the time series of the variates and plot the learned multivariate correlations with the marks of specific correlations between variates. On the one hand, we observe clear partitioning in the multivariate correlations map, indicating the grouping of variates. On the one hand, the marked correlation values can reflect the correlation of the raw series, where the similarity of variates from the same application becomes closer than the pairs from the different groups. Therefore, highly correlated variate will be leveraged for the next interaction and thus benefit for multivariate forecasting. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/3d0a1eb1d0620052cf18c38a3549efdd5929fb709c13c03ee9bbf5897fba1c35.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/88e0cf9638045ae7070e7c15d56978ac0a1c88bca4d8c80df09ad679f7849f8a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/3d4240bc5885511a06ddb699c082477302e363adec7db5fec25223b52f188a4a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/2e7e046bfd2c6fba7bcd033a2d72c6fbf68823d9d4f899d5c794325aaf9ab7e3.jpg)



Figure 12: Visualization of the variates from the Market dataset and the learned multivariate correlations. Each variate represents the monitored interface values of an application, and the applications can be further grouped into refined categories. The color bar is shared with Figure 11.


## E.2 VISUALIZATION OF PREDICTION RESULTS

To provide a clear comparison among different models, we list supplementary prediction showcases of four representative datasets in Figures 13- 16, which are given by the following models: iTransfomrer, PatchTST (Nie et al., 2023), DLinear (Zeng et al., 2023), Crossformer (Zhang & Yan, 2023), Autoformer (Wu et al., 2021), Transformer (Vaswani et al., 2017). Among the various models, iTransformer predicts the most precise future series variations and exhibits superior performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/26258ab3dee7d11eed55375b257b828bad949fb05ac091ecda979594c940a3f0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/08805ad83405e752b49ddbac1a3b415510a486f4eb16d2af2541978dd80a9505.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/80f7889d07932c88c4279dde13e1bf03d996ee4215be5dcdba5a933d96fb0a37.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/ea41a3afff5287bb4c185634b70b894d2ed792c5aef5118e468964b148ae4598.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/a626fb11c44472fa48b01351d83c0b6325ea77c6bd90e2df431d3d4d4d7c7b68.jpg)



Figure 13: Visualization of input-96-predict-96 results on the Traffic dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/824f6a9b07c0e1dbb8012fc27e8eec1309669105efb12d5cb9525e4e7d54d343.jpg)


## E.3 RISKS OF EMBEDDING MULTIVARIATE POINTS OF A TIMESTAMP

As aforementioned, the embedding approach of the previous Transformer fuses multiple variates representing potentially delayed events and distinct physical measurements, which may fail to learn variate-centric representations and result in meaningless attention maps. We provide the visualization case of Traffic (Liu et al., 2022a), which is collected from sensors on Los Angeles city roads in different areas. As shown in Figure 17, we can observe a strong correlation between the multivariate time series of the dataset, while they also exhibit obvious phase offset, which is due to the systematical time lags in the road occupancy that each series describes. Since the sensors are installed in different areas of the highway, an event (such as a traffic jam) can affect road occupancy with different delays. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/dbcd45472d8f34e1bb77f17e438dcba152a282dc51f6a7ef2680b65765e8e899.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/ead540b8291ec6164b85cfb501d499eddb2d74ec72ee61e59aca5f8e1a08c278.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/7bf2fdddc3c8ef5df7dfc6938104a965b32716fc9f4be529def7dcaf8e0fbae6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/7b87cedcb9f92e87432f35bceff7377f2598e410269c5a85e21c80a60b0337c3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/655991f813a2bc052faa6d7aa214262af62acab3bbeba3b99d03f040af01ee22.jpg)



Figure 14: Visualization of input-96-predict-96 results on the ECL dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/13f59100e1b0c8dfa42526008974a234f6622dbc988324659838035b9d8c5b21.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/7b3682df04481899ac229c588dcf08396dd3559d490ba947228d4a4ca67587b8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/51c2ffe84969a08cbc69b3afe1ddd98bb40db8abf0fc3027933c698754f7ee4f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/c8354918a223fe9f3e331a58e9b4e9451287a020f5e3136587e64a6d7c20c136.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/9d1e661c5a813b831fd789f9ea93651674e7ed4619afe81beb9b0b19dc25fad7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/af144b86a3dc3a9a49cb3579ab6f4f4ea93c3e901ac3ec5103a8b3f377186176.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/d4e27473506fdb9febee1960c1172949b76f9bfc5d457d61e8b19bb8353385d8.jpg)



Figure 15: Visualization of input-96-predict-96 results on the Weather dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/a1770908f919ac362ff13b4e38a72c4c8eea96f30d15b170d81e47f3a825fdc9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/c62abda9446edcf8eb04c8bd0ac9b730aaacdf3d558f1a99bf1d824bc6149a74.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/ee40bb9d01de0911e351b69160a55287572de8420359cdd6611dda991abc741c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/53b49e5d24cbffda7bcfbd53338c2abffaec82a4eaf45af88bb030990988f8d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/562912eb4ff700112a92ba018a3b0b35f822dcce853ba9e690c5d58b72c394b2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/bc1e72dd529d1c3181cd8967b51b335dc8d770e74a3022ff59584f8c46dae0f8.jpg)



Figure 16: Visualization of input-96-predict-96 results on the PEMS dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/68cd6e63a8fdb2ea7d0a80ce32026feb8174571296445348c156aca169ccc7a3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/17601a8fc141c3d1154a7a55f8899bcad5467b9cc590975a74d1af8447ac675c.jpg)



Figure 17: Visualization of partial variates of Traffic. We can observe that several series exhibit strong synchronization (such as Sensor 2 and Sensor 4), and there also exist obvious delays and advances between series (such as Sensor 1 and Sensor 2, Sensor 859 and Sensor 861).


Besides, we observe the significantly declined performance on the second and third designs of Traffic in Table 6, which apply attention to temporal tokens. In our opinion, capturing temporal dependencies by attention is not a big problem. But it is based on the fact that the time points of each timestamp essentially reflect the same event to enclose a semantic representation. Since there are inherent delays between the time points, the performance can degrade a lot because of the meaningless attention map, unless the model has an enlarged respective field to learn about the decay or causal process. 

Other risks can be aroused from the distinct variate measurements, such as organizing together different meteorological indicators (the temperature and rainfall) in the Weather dataset (Wu et al., 2021), and the quantity and proportion of the same observation in ILI (Wu et al., 2023). Given these potential risks, iTransformer proposes a new paradigm that embeds the whole series as the variate token, which can be more robust to extensive real-world scenarios, such as delayed events, inconsistent measurements, irregular (unevenly spaced) time series, systematical delay of monitors, and the time interval of generating and recording different time series. 

## F FULL RESULTS

## F.1 FULL PROMOTION RESULTS

We compare the performance of Transformer and iTransformer on all datasets in Table 7. Consistent and great promotions can be achieved, indicating that the attention and feed-forward network on the inverted dimensions greatly empower Transformers in multivariate time series forecasting, leaving an instructive direction to build up the foundation model of extensive time series data. 


Table 7: Full performance comparison between the vanilla Transformer and the proposed iTransformer. The results are averaged from all four prediction lengths.


<table><tr><td rowspan="2">Datasets Metric</td><td colspan="2">ETT</td><td colspan="2">ECL</td><td colspan="2">PEMS</td><td colspan="2">Solar-Energy</td><td colspan="2">Traffic</td><td colspan="2">Weather</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Transformer</td><td>2.750</td><td>1.375</td><td>0.277</td><td>0.372</td><td>0.157</td><td>0.263</td><td>0.256</td><td>0.276</td><td>0.665</td><td>0.363</td><td>0.657</td><td>0.572</td></tr><tr><td>iTransformer</td><td>0.383</td><td>0.407</td><td>0.178</td><td>0.270</td><td>0.113</td><td>0.221</td><td>0.233</td><td>0.262</td><td>0.428</td><td>0.282</td><td>0.258</td><td>0.279</td></tr><tr><td>Promotion</td><td>86.1%</td><td>70.4%</td><td>35.6%</td><td>27.4%</td><td>28.0%</td><td>16.0%</td><td>9.0%</td><td>5.1%</td><td>35.6%</td><td>22.3%</td><td>60.2%</td><td>50.8%</td></tr></table>

## F.2 FULL FRAMEWORK GENERALITY RESULTS

We apply the proposed inverting framework to Transformer and its variants: Transformer (Vaswani et al., 2017), Reformer (Kitaev et al., 2020), Informer (Li et al., 2021), Flowformer (Wu et al., 

2022), Flashformer (Dao et al., 2022). The averaged results are shown in Table 2 due to the limited pages. We provide the supplementary forecasting results in Table 8. The results demonstrate that our iTransformers framework can consistently promote these Transformer variants, and take advantage of the booming efficient attention mechanisms. 


Table 8: Full results of Transformers with our inverted framework. Flashformer means Transformer equipped with the hardware-accelerated FlashAttention (Dao et al., 2022).


<table><tr><td colspan="3">Models</td><td colspan="2">Transformer (2017)</td><td colspan="2">Reformer (2020)</td><td colspan="2">Informer (2021)</td><td colspan="2">Flowformer (2022)</td><td colspan="2">Flashformer (2022)</td></tr><tr><td colspan="3">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="10">ECL</td><td rowspan="4">Original</td><td>96</td><td>0.260</td><td>0.358</td><td>0.312</td><td>0.402</td><td>0.274</td><td>0.368</td><td>0.215</td><td>0.320</td><td>0.259</td><td>0.357</td></tr><tr><td>192</td><td>0.266</td><td>0.367</td><td>0.348</td><td>0.433</td><td>0.296</td><td>0.386</td><td>0.259</td><td>0.355</td><td>0.274</td><td>0.374</td></tr><tr><td>336</td><td>0.280</td><td>0.375</td><td>0.350</td><td>0.433</td><td>0.300</td><td>0.394</td><td>0.296</td><td>0.383</td><td>0.310</td><td>0.396</td></tr><tr><td>720</td><td>0.302</td><td>0.386</td><td>0.340</td><td>0.420</td><td>0.373</td><td>0.439</td><td>0.296</td><td>0.380</td><td>0.298</td><td>0.383</td></tr><tr><td></td><td>Avg</td><td>0.277</td><td>0.372</td><td>0.338</td><td>0.422</td><td>0.311</td><td>0.397</td><td>0.267</td><td>0.359</td><td>0.285</td><td>0.377</td></tr><tr><td rowspan="4">+Inverted</td><td>96</td><td>0.148</td><td>0.240</td><td>0.182</td><td>0.275</td><td>0.190</td><td>0.286</td><td>0.183</td><td>0.267</td><td>0.178</td><td>0.265</td></tr><tr><td>192</td><td>0.162</td><td>0.253</td><td>0.192</td><td>0.286</td><td>0.201</td><td>0.297</td><td>0.192</td><td>0.277</td><td>0.189</td><td>0.276</td></tr><tr><td>336</td><td>0.178</td><td>0.269</td><td>0.210</td><td>0.304</td><td>0.218</td><td>0.315</td><td>0.210</td><td>0.295</td><td>0.207</td><td>0.294</td></tr><tr><td>720</td><td>0.225</td><td>0.317</td><td>0.249</td><td>0.339</td><td>0.255</td><td>0.347</td><td>0.255</td><td>0.332</td><td>0.251</td><td>0.329</td></tr><tr><td></td><td>Avg</td><td>0.178</td><td>0.270</td><td>0.208</td><td>0.301</td><td>0.216</td><td>0.311</td><td>0.210</td><td>0.293</td><td>0.206</td><td>0.291</td></tr><tr><td rowspan="10">Traffic</td><td rowspan="4">Original</td><td>96</td><td>0.647</td><td>0.357</td><td>0.732</td><td>0.423</td><td>0.719</td><td>0.391</td><td>0.691</td><td>0.393</td><td>0.641</td><td>0.348</td></tr><tr><td>192</td><td>0.649</td><td>0.356</td><td>0.733</td><td>0.420</td><td>0.696</td><td>0.379</td><td>0.729</td><td>0.419</td><td>0.648</td><td>0.358</td></tr><tr><td>336</td><td>0.667</td><td>0.364</td><td>0.742</td><td>0.420</td><td>0.777</td><td>0.420</td><td>0.756</td><td>0.423</td><td>0.670</td><td>0.364</td></tr><tr><td>720</td><td>0.697</td><td>0.376</td><td>0.755</td><td>0.432</td><td>0.864</td><td>0.472</td><td>0.825</td><td>0.449</td><td>0.673</td><td>0.354</td></tr><tr><td></td><td>Avg</td><td>0.665</td><td>0.363</td><td>0.741</td><td>0.422</td><td>0.764</td><td>0.416</td><td>0.750</td><td>0.421</td><td>0.658</td><td>0.356</td></tr><tr><td rowspan="4">+Inverted</td><td>96</td><td>0.395</td><td>0.268</td><td>0.617</td><td>0.356</td><td>0.632</td><td>0.367</td><td>0.493</td><td>0.339</td><td>0.464</td><td>0.320</td></tr><tr><td>192</td><td>0.417</td><td>0.276</td><td>0.629</td><td>0.361</td><td>0.641</td><td>0.370</td><td>0.506</td><td>0.345</td><td>0.479</td><td>0.326</td></tr><tr><td>336</td><td>0.433</td><td>0.283</td><td>0.648</td><td>0.370</td><td>0.663</td><td>0.379</td><td>0.526</td><td>0.355</td><td>0.501</td><td>0.337</td></tr><tr><td>720</td><td>0.467</td><td>0.302</td><td>0.694</td><td>0.394</td><td>0.713</td><td>0.405</td><td>0.572</td><td>0.381</td><td>0.524</td><td>0.350</td></tr><tr><td></td><td>Avg</td><td>0.428</td><td>0.282</td><td>0.647</td><td>0.370</td><td>0.662</td><td>0.380</td><td>0.524</td><td>0.355</td><td>0.492</td><td>0.333</td></tr><tr><td rowspan="10">Weather</td><td rowspan="4">Original</td><td>96</td><td>0.395</td><td>0.427</td><td>0.689</td><td>0.596</td><td>0.300</td><td>0.384</td><td>0.182</td><td>0.233</td><td>0.388</td><td>0.425</td></tr><tr><td>192</td><td>0.619</td><td>0.560</td><td>0.752</td><td>0.638</td><td>0.598</td><td>0.544</td><td>0.250</td><td>0.288</td><td>0.619</td><td>0.560</td></tr><tr><td>336</td><td>0.689</td><td>0.594</td><td>0.639</td><td>0.596</td><td>0.578</td><td>0.523</td><td>0.309</td><td>0.329</td><td>0.698</td><td>0.600</td></tr><tr><td>720</td><td>0.926</td><td>0.710</td><td>1.130</td><td>0.792</td><td>1.059</td><td>0.741</td><td>0.404</td><td>0.385</td><td>0.930</td><td>0.711</td></tr><tr><td></td><td>Avg</td><td>0.657</td><td>0.572</td><td>0.803</td><td>0.656</td><td>0.634</td><td>0.548</td><td>0.286</td><td>0.308</td><td>0.659</td><td>0.574</td></tr><tr><td rowspan="4">+Inverted</td><td>96</td><td>0.174</td><td>0.214</td><td>0.169</td><td>0.225</td><td>0.180</td><td>0.251</td><td>0.183</td><td>0.223</td><td>0.177</td><td>0.218</td></tr><tr><td>192</td><td>0.221</td><td>0.254</td><td>0.213</td><td>0.265</td><td>0.244</td><td>0.318</td><td>0.231</td><td>0.262</td><td>0.229</td><td>0.261</td></tr><tr><td>336</td><td>0.278</td><td>0.296</td><td>0.268</td><td>0.317</td><td>0.282</td><td>0.343</td><td>0.286</td><td>0.301</td><td>0.283</td><td>0.300</td></tr><tr><td>720</td><td>0.358</td><td>0.349</td><td>0.340</td><td>0.361</td><td>0.377</td><td>0.409</td><td>0.363</td><td>0.352</td><td>0.359</td><td>0.251</td></tr><tr><td></td><td>Avg</td><td>0.258</td><td>0.279</td><td>0.248</td><td>0.292</td><td>0.271</td><td>0.330</td><td>0.266</td><td>0.285</td><td>0.262</td><td>0.282</td></tr></table>

## F.3 FULL RESULTS OF VARIATE GENERALIZATION

We divide the variates of each dataset into five folders, train models with only 20% of variates of one folder, and directly forecast all variates without fine-tuning. We adopt two strategies for Transformers to generalize on unseen variates: (1) CI-Transformers (Nie et al., 2023): Channel Independence regards each variate of time series as independent channels, and trains with a shared backbone. During inference, the model predicts variates one by one, but the procedure can be time-consuming. (2) iTransformers: with the flexibility of the attention mechanism that the number of input tokens can be dynamically changeable, the amount of variates as tokens is no longer restricted and thus feasible to vary from training and inference, and can even allow the model to be trained on arbitrary variates. 

As shown in Table 18, iTransformers can be naturally trained with 20% variates and accomplish forecast on all variates with the ability to learn transferable representations. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/5feca4e21ec60a90bc3868766744e0e4d392bb2c045a01e0c78ebcc4d29375e9.jpg)



Figure 18: Full performance of generalization on unseen variates, comparing the iTransformers with CI-Transfomers. We divide the variates of each dataset into five folders, train with 20% variates, and use the trained model to forecast all varieties. We plot the averaged results of all five folders.


## F.4 FULL FORECASTING RESULTS

The full multivariate forecasting results are provided in the following section due to the space limita tion of the main text. We extensively evaluate competitive counterparts on challenging forecasting tasks. Table 9 contains the forecasting results on the four public subsets from PEMS (Liu et al., 2022a). Table 10 contains the detailed results of all prediction lengths of the nine well-acknowledged forecasting benchmarks. And Table 11 records the Market results for Alipay server load forecasting. The proposed model achieves comprehensive state-of-the-art in real-world forecasting applications. 


Table 9: Full results of the PEMS forecasting task. We compare extensive competitive models under different prediction lengths following the setting of SCINet (2022a). The input length is set to 96 for all baselines. Avg means the average results from all four prediction lengths.


<table><tr><td colspan="2">Models</td><td colspan="2">iTransformer(Ours)</td><td colspan="2">RLinear(2023)</td><td colspan="2">PatchTST(2023)</td><td colspan="2">Crossformer(2023)</td><td colspan="2">TiDE(2023)</td><td colspan="2">TimesNet(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">SCINet(2022a)</td><td colspan="2">FEDformer(2022)</td><td colspan="2">Stationary(2022b)</td><td colspan="2">Autoformer(2021)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">PEMS03</td><td>12</td><td>0.071</td><td>0.174</td><td>0.126</td><td>0.236</td><td>0.099</td><td>0.216</td><td>0.090</td><td>0.203</td><td>0.178</td><td>0.305</td><td>0.085</td><td>0.192</td><td>0.122</td><td>0.243</td><td>0.066</td><td>0.172</td><td>0.126</td><td>0.251</td><td>0.081</td><td>0.188</td><td>0.272</td><td>0.385</td></tr><tr><td>24</td><td>0.093</td><td>0.201</td><td>0.246</td><td>0.334</td><td>0.142</td><td>0.259</td><td>0.121</td><td>0.240</td><td>0.257</td><td>0.371</td><td>0.118</td><td>0.223</td><td>0.201</td><td>0.317</td><td>0.085</td><td>0.198</td><td>0.149</td><td>0.275</td><td>0.105</td><td>0.214</td><td>0.334</td><td>0.440</td></tr><tr><td>48</td><td>0.125</td><td>0.236</td><td>0.551</td><td>0.529</td><td>0.211</td><td>0.319</td><td>0.202</td><td>0.317</td><td>0.379</td><td>0.463</td><td>0.155</td><td>0.260</td><td>0.333</td><td>0.425</td><td>0.127</td><td>0.238</td><td>0.227</td><td>0.348</td><td>0.154</td><td>0.257</td><td>1.032</td><td>0.782</td></tr><tr><td>96</td><td>0.164</td><td>0.275</td><td>1.057</td><td>0.787</td><td>0.269</td><td>0.370</td><td>0.262</td><td>0.367</td><td>0.490</td><td>0.539</td><td>0.228</td><td>0.317</td><td>0.457</td><td>0.515</td><td>0.178</td><td>0.287</td><td>0.348</td><td>0.434</td><td>0.247</td><td>0.336</td><td>1.031</td><td>0.796</td></tr><tr><td>Avg</td><td>0.113</td><td>0.221</td><td>0.495</td><td>0.472</td><td>0.180</td><td>0.291</td><td>0.169</td><td>0.281</td><td>0.326</td><td>0.419</td><td>0.147</td><td>0.248</td><td>0.278</td><td>0.375</td><td>0.114</td><td>0.224</td><td>0.213</td><td>0.327</td><td>0.147</td><td>0.249</td><td>0.667</td><td>0.601</td></tr><tr><td rowspan="5">PEMS04</td><td>12</td><td>0.078</td><td>0.183</td><td>0.138</td><td>0.252</td><td>0.105</td><td>0.224</td><td>0.098</td><td>0.218</td><td>0.219</td><td>0.340</td><td>0.087</td><td>0.195</td><td>0.148</td><td>0.272</td><td>0.073</td><td>0.177</td><td>0.138</td><td>0.262</td><td>0.088</td><td>0.196</td><td>0.424</td><td>0.491</td></tr><tr><td>24</td><td>0.095</td><td>0.205</td><td>0.258</td><td>0.348</td><td>0.153</td><td>0.275</td><td>0.131</td><td>0.256</td><td>0.292</td><td>0.398</td><td>0.103</td><td>0.215</td><td>0.224</td><td>0.340</td><td>0.084</td><td>0.193</td><td>0.177</td><td>0.293</td><td>0.104</td><td>0.216</td><td>0.459</td><td>0.509</td></tr><tr><td>48</td><td>0.120</td><td>0.233</td><td>0.572</td><td>0.544</td><td>0.229</td><td>0.339</td><td>0.205</td><td>0.326</td><td>0.409</td><td>0.478</td><td>0.136</td><td>0.250</td><td>0.355</td><td>0.437</td><td>0.099</td><td>0.211</td><td>0.270</td><td>0.368</td><td>0.137</td><td>0.251</td><td>0.646</td><td>0.610</td></tr><tr><td>96</td><td>0.150</td><td>0.262</td><td>1.137</td><td>0.820</td><td>0.291</td><td>0.389</td><td>0.402</td><td>0.457</td><td>0.492</td><td>0.532</td><td>0.190</td><td>0.303</td><td>0.452</td><td>0.504</td><td>0.114</td><td>0.227</td><td>0.341</td><td>0.427</td><td>0.186</td><td>0.297</td><td>0.912</td><td>0.748</td></tr><tr><td>Avg</td><td>0.111</td><td>0.221</td><td>0.526</td><td>0.491</td><td>0.195</td><td>0.307</td><td>0.209</td><td>0.314</td><td>0.353</td><td>0.437</td><td>0.129</td><td>0.241</td><td>0.295</td><td>0.388</td><td>0.092</td><td>0.202</td><td>0.231</td><td>0.337</td><td>0.127</td><td>0.240</td><td>0.610</td><td>0.590</td></tr><tr><td rowspan="5">PEMS07</td><td>12</td><td>0.067</td><td>0.165</td><td>0.118</td><td>0.235</td><td>0.095</td><td>0.207</td><td>0.094</td><td>0.200</td><td>0.173</td><td>0.304</td><td>0.082</td><td>0.181</td><td>0.115</td><td>0.242</td><td>0.068</td><td>0.171</td><td>0.109</td><td>0.225</td><td>0.083</td><td>0.185</td><td>0.199</td><td>0.336</td></tr><tr><td>24</td><td>0.088</td><td>0.190</td><td>0.242</td><td>0.341</td><td>0.150</td><td>0.262</td><td>0.139</td><td>0.247</td><td>0.271</td><td>0.383</td><td>0.101</td><td>0.204</td><td>0.210</td><td>0.329</td><td>0.119</td><td>0.225</td><td>0.125</td><td>0.244</td><td>0.102</td><td>0.207</td><td>0.323</td><td>0.420</td></tr><tr><td>48</td><td>0.110</td><td>0.215</td><td>0.562</td><td>0.541</td><td>0.253</td><td>0.340</td><td>0.311</td><td>0.369</td><td>0.446</td><td>0.495</td><td>0.134</td><td>0.238</td><td>0.398</td><td>0.458</td><td>0.149</td><td>0.237</td><td>0.165</td><td>0.288</td><td>0.136</td><td>0.240</td><td>0.390</td><td>0.470</td></tr><tr><td>96</td><td>0.139</td><td>0.245</td><td>1.096</td><td>0.795</td><td>0.346</td><td>0.404</td><td>0.396</td><td>0.442</td><td>0.628</td><td>0.577</td><td>0.181</td><td>0.279</td><td>0.594</td><td>0.553</td><td>0.141</td><td>0.234</td><td>0.262</td><td>0.376</td><td>0.187</td><td>0.287</td><td>0.554</td><td>0.578</td></tr><tr><td>Avg</td><td>0.101</td><td>0.204</td><td>0.504</td><td>0.478</td><td>0.211</td><td>0.303</td><td>0.235</td><td>0.315</td><td>0.380</td><td>0.440</td><td>0.124</td><td>0.225</td><td>0.329</td><td>0.395</td><td>0.119</td><td>0.234</td><td>0.165</td><td>0.283</td><td>0.127</td><td>0.230</td><td>0.367</td><td>0.451</td></tr><tr><td rowspan="5">PEMS08</td><td>12</td><td>0.079</td><td>0.182</td><td>0.133</td><td>0.247</td><td>0.168</td><td>0.232</td><td>0.165</td><td>0.214</td><td>0.227</td><td>0.343</td><td>0.112</td><td>0.212</td><td>0.154</td><td>0.276</td><td>0.087</td><td>0.184</td><td>0.173</td><td>0.273</td><td>0.109</td><td>0.207</td><td>0.436</td><td>0.485</td></tr><tr><td>24</td><td>0.115</td><td>0.219</td><td>0.249</td><td>0.343</td><td>0.224</td><td>0.281</td><td>0.215</td><td>0.260</td><td>0.318</td><td>0.409</td><td>0.141</td><td>0.238</td><td>0.248</td><td>0.353</td><td>0.122</td><td>0.221</td><td>0.210</td><td>0.301</td><td>0.140</td><td>0.236</td><td>0.467</td><td>0.502</td></tr><tr><td>48</td><td>0.186</td><td>0.235</td><td>0.569</td><td>0.544</td><td>0.321</td><td>0.354</td><td>0.315</td><td>0.355</td><td>0.497</td><td>0.510</td><td>0.198</td><td>0.283</td><td>0.440</td><td>0.470</td><td>0.189</td><td>0.270</td><td>0.320</td><td>0.394</td><td>0.211</td><td>0.294</td><td>0.966</td><td>0.733</td></tr><tr><td>96</td><td>0.221</td><td>0.267</td><td>1.166</td><td>0.814</td><td>0.408</td><td>0.417</td><td>0.377</td><td>0.397</td><td>0.721</td><td>0.592</td><td>0.320</td><td>0.351</td><td>0.674</td><td>0.565</td><td>0.236</td><td>0.300</td><td>0.442</td><td>0.465</td><td>0.345</td><td>0.367</td><td>1.385</td><td>0.915</td></tr><tr><td>Avg</td><td>0.150</td><td>0.226</td><td>0.529</td><td>0.487</td><td>0.280</td><td>0.321</td><td>0.268</td><td>0.307</td><td>0.441</td><td>0.464</td><td>0.193</td><td>0.271</td><td>0.379</td><td>0.416</td><td>0.158</td><td>0.244</td><td>0.286</td><td>0.358</td><td>0.201</td><td>0.276</td><td>0.814</td><td>0.659</td></tr><tr><td colspan="2"><eq>1^{st}</eq>Count</td><td>13</td><td>13</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>7</td><td>7</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr></table>


Table 10: Full results of the long-term forecasting task. We compare extensive competitive models under different prediction lengths following the setting of TimesNet (2023). The input sequence length is set to 96 for all baselines. Avg means the average results from all four prediction lengths.


<table><tr><td colspan="2">Models</td><td colspan="2">iTransformer (Ours)</td><td colspan="2">RLinear (2023)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">TiDE (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">SCINet (2022a)</td><td colspan="2">FEDformer (2022)</td><td colspan="2">Stationary (2022b)</td><td colspan="2">Autoformer (2021)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.334</td><td>0.368</td><td>0.355</td><td>0.376</td><td>0.329</td><td>0.367</td><td>0.404</td><td>0.426</td><td>0.364</td><td>0.387</td><td>0.338</td><td>0.375</td><td>0.345</td><td>0.372</td><td>0.418</td><td>0.438</td><td>0.379</td><td>0.419</td><td>0.386</td><td>0.398</td><td>0.505</td><td>0.475</td></tr><tr><td>192</td><td>0.377</td><td>0.391</td><td>0.391</td><td>0.392</td><td>0.367</td><td>0.385</td><td>0.450</td><td>0.451</td><td>0.398</td><td>0.404</td><td>0.374</td><td>0.387</td><td>0.380</td><td>0.389</td><td>0.439</td><td>0.450</td><td>0.426</td><td>0.441</td><td>0.459</td><td>0.444</td><td>0.553</td><td>0.496</td></tr><tr><td>336</td><td>0.426</td><td>0.420</td><td>0.424</td><td>0.415</td><td>0.399</td><td>0.410</td><td>0.532</td><td>0.515</td><td>0.428</td><td>0.425</td><td>0.410</td><td>0.411</td><td>0.413</td><td>0.413</td><td>0.490</td><td>0.485</td><td>0.445</td><td>0.459</td><td>0.495</td><td>0.464</td><td>0.621</td><td>0.537</td></tr><tr><td>720</td><td>0.491</td><td>0.459</td><td>0.487</td><td>0.450</td><td>0.454</td><td>0.439</td><td>0.666</td><td>0.589</td><td>0.487</td><td>0.461</td><td>0.478</td><td>0.450</td><td>0.474</td><td>0.453</td><td>0.595</td><td>0.550</td><td>0.543</td><td>0.490</td><td>0.585</td><td>0.516</td><td>0.671</td><td>0.561</td></tr><tr><td></td><td>Avg</td><td>0.407</td><td>0.410</td><td>0.414</td><td>0.407</td><td>0.387</td><td>0.400</td><td>0.513</td><td>0.496</td><td>0.419</td><td>0.419</td><td>0.400</td><td>0.406</td><td>0.403</td><td>0.407</td><td>0.485</td><td>0.481</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.180</td><td>0.264</td><td>0.182</td><td>0.265</td><td>0.175</td><td>0.259</td><td>0.287</td><td>0.366</td><td>0.207</td><td>0.305</td><td>0.187</td><td>0.267</td><td>0.193</td><td>0.292</td><td>0.286</td><td>0.377</td><td>0.203</td><td>0.287</td><td>0.192</td><td>0.274</td><td>0.255</td><td>0.339</td></tr><tr><td>192</td><td>0.250</td><td>0.309</td><td>0.246</td><td>0.304</td><td>0.241</td><td>0.302</td><td>0.414</td><td>0.492</td><td>0.290</td><td>0.364</td><td>0.249</td><td>0.309</td><td>0.284</td><td>0.362</td><td>0.399</td><td>0.445</td><td>0.269</td><td>0.328</td><td>0.280</td><td>0.339</td><td>0.281</td><td>0.340</td></tr><tr><td>336</td><td>0.311</td><td>0.348</td><td>0.307</td><td>0.342</td><td>0.305</td><td>0.343</td><td>0.597</td><td>0.542</td><td>0.377</td><td>0.422</td><td>0.321</td><td>0.351</td><td>0.369</td><td>0.427</td><td>0.637</td><td>0.591</td><td>0.325</td><td>0.366</td><td>0.334</td><td>0.361</td><td>0.339</td><td>0.372</td></tr><tr><td>720</td><td>0.412</td><td>0.407</td><td>0.407</td><td>0.398</td><td>0.402</td><td>0.400</td><td>1.730</td><td>1.042</td><td>0.558</td><td>0.524</td><td>0.408</td><td>0.403</td><td>0.554</td><td>0.522</td><td>0.960</td><td>0.735</td><td>0.421</td><td>0.415</td><td>0.417</td><td>0.413</td><td>0.433</td><td>0.432</td></tr><tr><td></td><td>Avg</td><td>0.288</td><td>0.332</td><td>0.286</td><td>0.327</td><td>0.281</td><td>0.326</td><td>0.757</td><td>0.610</td><td>0.358</td><td>0.404</td><td>0.291</td><td>0.333</td><td>0.350</td><td>0.401</td><td>0.571</td><td>0.537</td><td>0.305</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.327</td><td>0.371</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.386</td><td>0.405</td><td>0.386</td><td>0.395</td><td>0.414</td><td>0.419</td><td>0.423</td><td>0.448</td><td>0.479</td><td>0.464</td><td>0.384</td><td>0.402</td><td>0.386</td><td>0.400</td><td>0.654</td><td>0.599</td><td>0.376</td><td>0.419</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td></tr><tr><td>192</td><td>0.441</td><td>0.436</td><td>0.437</td><td>0.424</td><td>0.460</td><td>0.445</td><td>0.471</td><td>0.474</td><td>0.525</td><td>0.492</td><td>0.436</td><td>0.429</td><td>0.437</td><td>0.432</td><td>0.719</td><td>0.631</td><td>0.420</td><td>0.448</td><td>0.534</td><td>0.504</td><td>0.500</td><td>0.482</td></tr><tr><td>336</td><td>0.487</td><td>0.458</td><td>0.479</td><td>0.446</td><td>0.501</td><td>0.466</td><td>0.570</td><td>0.546</td><td>0.565</td><td>0.515</td><td>0.491</td><td>0.469</td><td>0.481</td><td>0.459</td><td>0.778</td><td>0.659</td><td>0.459</td><td>0.465</td><td>0.588</td><td>0.535</td><td>0.521</td><td>0.496</td></tr><tr><td>720</td><td>0.503</td><td>0.491</td><td>0.481</td><td>0.470</td><td>0.500</td><td>0.488</td><td>0.653</td><td>0.621</td><td>0.594</td><td>0.558</td><td>0.521</td><td>0.500</td><td>0.519</td><td>0.516</td><td>0.836</td><td>0.699</td><td>0.506</td><td>0.507</td><td>0.643</td><td>0.616</td><td>0.514</td><td>0.512</td></tr><tr><td></td><td>Avg</td><td>0.454</td><td>0.447</td><td>0.446</td><td>0.434</td><td>0.469</td><td>0.454</td><td>0.529</td><td>0.522</td><td>0.541</td><td>0.507</td><td>0.458</td><td>0.450</td><td>0.456</td><td>0.452</td><td>0.747</td><td>0.647</td><td>0.440</td><td>0.460</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.297</td><td>0.349</td><td>0.288</td><td>0.338</td><td>0.302</td><td>0.348</td><td>0.745</td><td>0.584</td><td>0.400</td><td>0.440</td><td>0.340</td><td>0.374</td><td>0.333</td><td>0.387</td><td>0.707</td><td>0.621</td><td>0.358</td><td>0.397</td><td>0.476</td><td>0.458</td><td>0.346</td><td>0.388</td></tr><tr><td>192</td><td>0.380</td><td>0.400</td><td>0.374</td><td>0.390</td><td>0.388</td><td>0.400</td><td>0.877</td><td>0.656</td><td>0.528</td><td>0.509</td><td>0.402</td><td>0.414</td><td>0.477</td><td>0.476</td><td>0.860</td><td>0.689</td><td>0.429</td><td>0.439</td><td>0.512</td><td>0.493</td><td>0.456</td><td>0.452</td></tr><tr><td>336</td><td>0.428</td><td>0.432</td><td>0.415</td><td>0.426</td><td>0.426</td><td>0.433</td><td>1.043</td><td>0.731</td><td>0.643</td><td>0.571</td><td>0.452</td><td>0.452</td><td>0.594</td><td>0.541</td><td>1.000</td><td>0.744</td><td>0.496</td><td>0.487</td><td>0.552</td><td>0.551</td><td>0.482</td><td>0.486</td></tr><tr><td>720</td><td>0.427</td><td>0.445</td><td>0.420</td><td>0.440</td><td>0.431</td><td>0.446</td><td>1.104</td><td>0.763</td><td>0.874</td><td>0.679</td><td>0.462</td><td>0.468</td><td>0.831</td><td>0.657</td><td>1.249</td><td>0.838</td><td>0.463</td><td>0.474</td><td>0.562</td><td>0.560</td><td>0.515</td><td>0.511</td></tr><tr><td></td><td>Avg</td><td>0.383</td><td>0.407</td><td>0.374</td><td>0.398</td><td>0.387</td><td>0.407</td><td>0.942</td><td>0.684</td><td>0.611</td><td>0.550</td><td>0.414</td><td>0.427</td><td>0.559</td><td>0.515</td><td>0.954</td><td>0.723</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td>0.459</td></tr><tr><td rowspan="4">ECL</td><td>96</td><td>0.148</td><td>0.240</td><td>0.201</td><td>0.281</td><td>0.181</td><td>0.270</td><td>0.219</td><td>0.314</td><td>0.237</td><td>0.329</td><td>0.168</td><td>0.272</td><td>0.197</td><td>0.282</td><td>0.247</td><td>0.345</td><td>0.193</td><td>0.308</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td></tr><tr><td>192</td><td>0.162</td><td>0.253</td><td>0.201</td><td>0.283</td><td>0.188</td><td>0.274</td><td>0.231</td><td>0.322</td><td>0.236</td><td>0.330</td><td>0.184</td><td>0.289</td><td>0.196</td><td>0.285</td><td>0.257</td><td>0.355</td><td>0.201</td><td>0.315</td><td>0.182</td><td>0.286</td><td>0.222</td><td>0.334</td></tr><tr><td>336</td><td>0.178</td><td>0.269</td><td>0.215</td><td>0.298</td><td>0.204</td><td>0.293</td><td>0.246</td><td>0.337</td><td>0.249</td><td>0.344</td><td>0.198</td><td>0.300</td><td>0.209</td><td>0.301</td><td>0.269</td><td>0.369</td><td>0.214</td><td>0.329</td><td>0.200</td><td>0.304</td><td>0.231</td><td>0.338</td></tr><tr><td>720</td><td>0.225</td><td>0.317</td><td>0.257</td><td>0.331</td><td>0.246</td><td>0.324</td><td>0.280</td><td>0.363</td><td>0.284</td><td>0.373</td><td>0.220</td><td>0.320</td><td>0.245</td><td>0.333</td><td>0.299</td><td>0.390</td><td>0.246</td><td>0.355</td><td>0.222</td><td>0.321</td><td>0.254</td><td>0.361</td></tr><tr><td></td><td>Avg</td><td>0.178</td><td>0.270</td><td>0.219</td><td>0.298</td><td>0.205</td><td>0.290</td><td>0.244</td><td>0.334</td><td>0.251</td><td>0.344</td><td>0.192</td><td>0.295</td><td>0.212</td><td>0.300</td><td>0.268</td><td>0.365</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.086</td><td>0.206</td><td>0.093</td><td>0.217</td><td>0.088</td><td>0.205</td><td>0.256</td><td>0.367</td><td>0.094</td><td>0.218</td><td>0.107</td><td>0.234</td><td>0.088</td><td>0.218</td><td>0.267</td><td>0.396</td><td>0.148</td><td>0.278</td><td>0.111</td><td>0.237</td><td>0.197</td><td>0.323</td></tr><tr><td>192</td><td>0.177</td><td>0.299</td><td>0.184</td><td>0.307</td><td>0.176</td><td>0.299</td><td>0.470</td><td>0.509</td><td>0.184</td><td>0.307</td><td>0.226</td><td>0.344</td><td>0.176</td><td>0.315</td><td>0.351</td><td>0.459</td><td>0.271</td><td>0.315</td><td>0.219</td><td>0.335</td><td>0.300</td><td>0.369</td></tr><tr><td>336</td><td>0.331</td><td>0.417</td><td>0.351</td><td>0.432</td><td>0.301</td><td>0.397</td><td>1.268</td><td>0.883</td><td>0.349</td><td>0.431</td><td>0.367</td><td>0.448</td><td>0.313</td><td>0.427</td><td>1.324</td><td>0.853</td><td>0.460</td><td>0.427</td><td>0.421</td><td>0.476</td><td>0.509</td><td>0.524</td></tr><tr><td>720</td><td>0.847</td><td>0.691</td><td>0.886</td><td>0.714</td><td>0.901</td><td>0.714</td><td>1.767</td><td>1.068</td><td>0.852</td><td>0.698</td><td>0.964</td><td>0.746</td><td>0.839</td><td>0.695</td><td>1.058</td><td>0.797</td><td>1.195</td><td>0.695</td><td>1.092</td><td>0.769</td><td>1.447</td><td>0.941</td></tr><tr><td></td><td>Avg</td><td>0.360</td><td>0.403</td><td>0.378</td><td>0.417</td><td>0.367</td><td>0.404</td><td>0.940</td><td>0.707</td><td>0.370</td><td>0.413</td><td>0.416</td><td>0.443</td><td>0.354</td><td>0.414</td><td>0.750</td><td>0.626</td><td>0.519</td><td>0.429</td><td>0.461</td><td>0.454</td><td>0.613</td><td>0.539</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.395</td><td>0.268</td><td>0.649</td><td>0.389</td><td>0.462</td><td>0.295</td><td>0.522</td><td>0.290</td><td>0.805</td><td>0.493</td><td>0.593</td><td>0.321</td><td>0.650</td><td>0.396</td><td>0.788</td><td>0.499</td><td>0.587</td><td>0.366</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td></tr><tr><td>192</td><td>0.417</td><td>0.276</td><td>0.601</td><td>0.366</td><td>0.466</td><td>0.296</td><td>0.530</td><td>0.293</td><td>0.756</td><td>0.474</td><td>0.617</td><td>0.336</td><td>0.598</td><td>0.370</td><td>0.789</td><td>0.505</td><td>0.604</td><td>0.373</td><td>0.613</td><td>0.340</td><td>0.616</td><td>0.382</td></tr><tr><td>336</td><td>0.433</td><td>0.283</td><td>0.609</td><td>0.369</td><td>0.482</td><td>0.304</td><td>0.558</td><td>0.305</td><td>0.762</td><td>0.477</td><td>0.629</td><td>0.336</td><td>0.605</td><td>0.373</td><td>0.797</td><td>0.508</td><td>0.621</td><td>0.383</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.337</td></tr><tr><td>720</td><td>0.467</td><td>0.302</td><td>0.647</td><td>0.387</td><td>0.514</td><td>0.322</td><td>0.589</td><td>0.328</td><td>0.719</td><td>0.449</td><td>0.640</td><td>0.350</td><td>0.645</td><td>0.394</td><td>0.841</td><td>0.523</td><td>0.626</td><td>0.382</td><td>0.653</td><td>0.355</td><td>0.660</td><td>0.408</td></tr><tr><td></td><td>Avg</td><td>0.428</td><td>0.282</td><td>0.626</td><td>0.378</td><td>0.481</td><td>0.304</td><td>0.550</td><td>0.304</td><td>0.760</td><td>0.473</td><td>0.620</td><td>0.336</td><td>0.625</td><td>0.383</td><td>0.804</td><td>0.509</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.174</td><td>0.214</td><td>0.192</td><td>0.232</td><td>0.177</td><td>0.218</td><td>0.158</td><td>0.230</td><td>0.202</td><td>0.261</td><td>0.172</td><td>0.220</td><td>0.196</td><td>0.255</td><td>0.221</td><td>0.306</td><td>0.217</td><td>0.296</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td></tr><tr><td>192</td><td>0.221</td><td>0.254</td><td>0.240</td><td>0.271</td><td>0.225</td><td>0.259</td><td>0.206</td><td>0.277</td><td>0.242</td><td>0.298</td><td>0.219</td><td>0.261</td><td>0.237</td><td>0.296</td><td>0.261</td><td>0.340</td><td>0.276</td><td>0.336</td><td>0.245</td><td>0.285</td><td>0.307</td><td>0.367</td></tr><tr><td>336</td><td>0.278</td><td>0.296</td><td>0.292</td><td>0.307</td><td>0.278</td><td>0.297</td><td>0.272</td><td>0.335</td><td>0.287</td><td>0.335</td><td>0.280</td><td>0.306</td><td>0.283</td><td>0.335</td><td>0.309</td><td>0.378</td><td>0.339</td><td>0.380</td><td>0.321</td><td>0.338</td><td>0.359</td><td>0.395</td></tr><tr><td>720</td><td>0.358</td><td>0.347</td><td>0.364</td><td>0.353</td><td>0.354</td><td>0.348</td><td>0.398</td><td>0.418</td><td>0.351</td><td>0.386</td><td>0.365</td><td>0.359</td><td>0.345</td><td>0.381</td><td>0.377</td><td>0.427</td><td>0.403</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.419</td><td>0.428</td></tr><tr><td></td><td>Avg</td><td>0.258</td><td>0.278</td><td>0.272</td><td>0.291</td><td>0.259</td><td>0.281</td><td>0.259</td><td>0.315</td><td>0.271</td><td>0.320</td><td>0.259</td><td>0.287</td><td>0.265</td><td>0.317</td><td>0.292</td><td>0.363</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td></tr><tr><td rowspan="4">Solar-Energy</td><td>96</td><td>0.203</td><td>0.237</td><td>0.322</td><td>0.339</td><td>0.234</td><td>0.286</td><td>0.310</td><td>0.331</td><td>0.312</td><td>0.399</td><td>0.250</td><td>0.292</td><td>0.290</td><td>0.378</td><td>0.237</td><td>0.344</td><td>0.242</td><td>0.342</td><td>0.215</td><td>0.249</td><td>0.884</td><td>0.711</td></tr><tr><td>192</td><td>0.233</td><td>0.261</td><td>0.359</td><td>0.356</td><td>0.267</td><td>0.310</td><td>0.734</td><td>0.725</td><td>0.339</td><td>0.416</td><td>0.296</td><td>0.318</td><td>0.320</td><td>0.398</td><td>0.280</td><td>0.380</td><td>0.285</td><td>0.380</td><td>0.254</td><td>0.272</td><td>0.834</td><td>0.692</td></tr><tr><td>336</td><td>0.248</td><td>0.273</td><td>0.397</td><td>0.369</td><td>0.290</td><td>0.315</td><td>0.750</td><td>0.735</td><td>0.368</td><td>0.430</td><td>0.319</td><td>0.330</td><td>0.353</td><td>0.415</td><td>0.304</td><td>0.389</td><td>0.282</td><td>0.376</td><td>0.290</td><td>0.296</td><td>0.941</td><td>0.723</td></tr><tr><td>720</td><td>0.249</td><td>0.275</td><td>0.397</td><td>0.356</td><td>0.289</td><td>0.317</td><td>0.769</td><td>0.765</td><td>0.370</td><td>0.425</td><td>0.338</td><td>0.337</td><td>0.356</td><td>0.413</td><td>0.308</td><td>0.388</td><td>0.357</td><td>0.427</td><td>0.285</td><td>0.295</td><td>0.882</td><td>0.717</td></tr><tr><td></td><td>Avg</td><td>0.233</td><td>0.262</td><td>0.369</td><td>0.356</td><td>0.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>


Table 11: Full results of the Market dataset. We compare extensive competitive models on the real-world transaction forecasting task. Avg means the average results from all prediction lengths.


<table><tr><td colspan="2">Models</td><td colspan="2">iTransformer(Ours)</td><td colspan="2">RLinear(2023)</td><td colspan="2">PatchTST(2023)</td><td colspan="2">Crossformer(2023)</td><td colspan="2">TiDE(2023)</td><td colspan="2">TimesNet(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">SCINet(2022a)</td><td colspan="2">FEDformer(2022)</td><td colspan="2">Stationary(2022b)</td><td colspan="2">Autoformer(2021)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Merchant</td><td>12</td><td>0.058</td><td>0.126</td><td>0.139</td><td>0.232</td><td>0.072</td><td>0.155</td><td>0.068</td><td>0.141</td><td>0.173</td><td>0.273</td><td>0.088</td><td>0.177</td><td>0.093</td><td>0.183</td><td>0.202</td><td>0.310</td><td>0.277</td><td>0.384</td><td>0.143</td><td>0.243</td><td>0.365</td><td>0.444</td></tr><tr><td>24</td><td>0.066</td><td>0.138</td><td>0.155</td><td>0.250</td><td>0.079</td><td>0.164</td><td>0.091</td><td>0.161</td><td>0.170</td><td>0.274</td><td>0.103</td><td>0.195</td><td>0.105</td><td>0.200</td><td>0.215</td><td>0.323</td><td>0.268</td><td>0.378</td><td>0.167</td><td>0.270</td><td>0.669</td><td>0.636</td></tr><tr><td>72</td><td>0.079</td><td>0.157</td><td>0.156</td><td>0.252</td><td>0.090</td><td>0.180</td><td>0.123</td><td>0.202</td><td>0.197</td><td>0.298</td><td>0.089</td><td>0.180</td><td>0.116</td><td>0.215</td><td>0.388</td><td>0.431</td><td>0.281</td><td>0.390</td><td>0.193</td><td>0.300</td><td>0.404</td><td>0.479</td></tr><tr><td>144</td><td>0.086</td><td>0.167</td><td>0.157</td><td>0.253</td><td>0.093</td><td>0.185</td><td>0.185</td><td>0.218</td><td>0.208</td><td>0.311</td><td>0.091</td><td>0.183</td><td>0.124</td><td>0.225</td><td>0.459</td><td>0.477</td><td>0.359</td><td>0.453</td><td>0.183</td><td>0.294</td><td>0.536</td><td>0.566</td></tr><tr><td></td><td>Avg</td><td>0.072</td><td>0.147</td><td>0.152</td><td>0.247</td><td>0.084</td><td>0.171</td><td>0.117</td><td>0.181</td><td>0.187</td><td>0.289</td><td>0.093</td><td>0.184</td><td>0.110</td><td>0.206</td><td>0.316</td><td>0.385</td><td>0.296</td><td>0.401</td><td>0.172</td><td>0.277</td><td>0.494</td><td>0.531</td></tr><tr><td rowspan="4">Wealth</td><td>12</td><td>0.189</td><td>0.205</td><td>0.479</td><td>0.411</td><td>0.255</td><td>0.250</td><td>0.270</td><td>0.208</td><td>0.486</td><td>0.427</td><td>0.275</td><td>0.277</td><td>0.380</td><td>0.355</td><td>0.525</td><td>0.451</td><td>0.553</td><td>0.508</td><td>0.355</td><td>0.332</td><td>0.653</td><td>0.555</td></tr><tr><td>24</td><td>0.254</td><td>0.244</td><td>0.543</td><td>0.446</td><td>0.320</td><td>0.291</td><td>0.329</td><td>0.233</td><td>0.545</td><td>0.463</td><td>0.300</td><td>0.285</td><td>0.456</td><td>0.397</td><td>0.583</td><td>0.479</td><td>0.567</td><td>0.514</td><td>0.430</td><td>0.377</td><td>0.761</td><td>0.611</td></tr><tr><td>72</td><td>0.421</td><td>0.327</td><td>0.634</td><td>0.481</td><td>0.459</td><td>0.360</td><td>0.484</td><td>0.324</td><td>0.651</td><td>0.510</td><td>0.384</td><td>0.326</td><td>0.555</td><td>0.438</td><td>0.761</td><td>0.558</td><td>0.636</td><td>0.548</td><td>0.573</td><td>0.454</td><td>0.857</td><td>0.658</td></tr><tr><td>144</td><td>0.517</td><td>0.379</td><td>0.683</td><td>0.504</td><td>0.541</td><td>0.404</td><td>0.633</td><td>0.388</td><td>0.698</td><td>0.526</td><td>0.481</td><td>0.383</td><td>0.611</td><td>0.459</td><td>0.770</td><td>0.568</td><td>0.744</td><td>0.604</td><td>0.637</td><td>0.498</td><td>0.817</td><td>0.627</td></tr><tr><td></td><td>Avg</td><td>0.345</td><td>0.289</td><td>0.585</td><td>0.461</td><td>0.394</td><td>0.326</td><td>0.429</td><td>0.288</td><td>0.595</td><td>0.481</td><td>0.360</td><td>0.318</td><td>0.501</td><td>0.412</td><td>0.660</td><td>0.514</td><td>0.625</td><td>0.543</td><td>0.499</td><td>0.415</td><td>0.772</td><td>0.612</td></tr><tr><td rowspan="4">Finance</td><td>12</td><td>0.123</td><td>0.170</td><td>0.329</td><td>0.304</td><td>0.164</td><td>0.206</td><td>4.630</td><td>0.520</td><td>0.512</td><td>0.350</td><td>0.465</td><td>0.291</td><td>0.321</td><td>0.271</td><td>1.865</td><td>0.602</td><td>1.537</td><td>0.538</td><td>0.537</td><td>0.384</td><td>1.651</td><td>0.593</td></tr><tr><td>24</td><td>0.158</td><td>0.197</td><td>0.386</td><td>0.332</td><td>0.198</td><td>0.228</td><td>4.987</td><td>0.568</td><td>0.635</td><td>0.388</td><td>0.503</td><td>0.297</td><td>0.464</td><td>0.318</td><td>2.228</td><td>0.664</td><td>1.553</td><td>0.547</td><td>0.551</td><td>0.386</td><td>1.671</td><td>0.594</td></tr><tr><td>72</td><td>0.212</td><td>0.240</td><td>0.436</td><td>0.353</td><td>0.268</td><td>0.273</td><td>5.631</td><td>0.675</td><td>1.239</td><td>0.490</td><td>0.534</td><td>0.310</td><td>0.986</td><td>0.423</td><td>3.084</td><td>0.793</td><td>1.612</td><td>0.554</td><td>2.004</td><td>0.853</td><td>2.054</td><td>0.758</td></tr><tr><td>144</td><td>0.245</td><td>0.257</td><td>0.429</td><td>0.355</td><td>0.293</td><td>0.286</td><td>6.083</td><td>0.708</td><td>1.562</td><td>0.538</td><td>0.564</td><td>0.333</td><td>1.287</td><td>0.473</td><td>4.089</td><td>0.875</td><td>1.784</td><td>0.636</td><td>2.379</td><td>0.947</td><td>2.114</td><td>0.778</td></tr><tr><td></td><td>Avg</td><td>0.184</td><td>0.216</td><td>0.395</td><td>0.336</td><td>0.231</td><td>0.248</td><td>5.333</td><td>0.618</td><td>0.987</td><td>0.442</td><td>0.516</td><td>0.308</td><td>0.765</td><td>0.372</td><td>2.817</td><td>0.734</td><td>1.621</td><td>0.569</td><td>1.368</td><td>0.643</td><td>1.872</td><td>0.681</td></tr><tr><td rowspan="4">Terminal</td><td>12</td><td>0.051</td><td>0.127</td><td>0.168</td><td>0.272</td><td>0.068</td><td>0.164</td><td>0.055</td><td>0.140</td><td>0.212</td><td>0.304</td><td>0.074</td><td>0.169</td><td>0.096</td><td>0.198</td><td>0.199</td><td>0.301</td><td>0.268</td><td>0.379</td><td>0.140</td><td>0.252</td><td>0.386</td><td>0.461</td></tr><tr><td>24</td><td>0.059</td><td>0.139</td><td>0.185</td><td>0.290</td><td>0.074</td><td>0.173</td><td>0.065</td><td>0.155</td><td>0.201</td><td>0.301</td><td>0.081</td><td>0.178</td><td>0.105</td><td>0.209</td><td>0.225</td><td>0.325</td><td>0.256</td><td>0.370</td><td>0.174</td><td>0.289</td><td>0.708</td><td>0.644</td></tr><tr><td>72</td><td>0.071</td><td>0.160</td><td>0.183</td><td>0.291</td><td>0.081</td><td>0.187</td><td>0.077</td><td>0.170</td><td>0.222</td><td>0.316</td><td>0.077</td><td>0.178</td><td>0.109</td><td>0.215</td><td>0.317</td><td>0.338</td><td>0.285</td><td>0.396</td><td>0.202</td><td>0.321</td><td>0.510</td><td>0.552</td></tr><tr><td>144</td><td>0.079</td><td>0.171</td><td>0.184</td><td>0.292</td><td>0.085</td><td>0.193</td><td>0.085</td><td>0.181</td><td>0.229</td><td>0.322</td><td>0.088</td><td>0.192</td><td>0.113</td><td>0.220</td><td>0.378</td><td>0.425</td><td>0.372</td><td>0.468</td><td>0.204</td><td>0.322</td><td>0.468</td><td>0.528</td></tr><tr><td></td><td>Avg</td><td>0.065</td><td>0.150</td><td>0.180</td><td>0.286</td><td>0.077</td><td>0.179</td><td>0.071</td><td>0.162</td><td>0.216</td><td>0.311</td><td>0.080</td><td>0.179</td><td>0.106</td><td>0.210</td><td>0.280</td><td>0.360</td><td>0.295</td><td>0.403</td><td>0.180</td><td>0.296</td><td>0.518</td><td>0.547</td></tr><tr><td rowspan="4">Payment</td><td>12</td><td>0.050</td><td>0.121</td><td>0.123</td><td>0.230</td><td>0.065</td><td>0.156</td><td>0.152</td><td>0.145</td><td>0.184</td><td>0.265</td><td>0.094</td><td>0.171</td><td>0.090</td><td>0.180</td><td>0.164</td><td>0.249</td><td>0.272</td><td>0.349</td><td>0.129</td><td>0.229</td><td>0.382</td><td>0.437</td></tr><tr><td>24</td><td>0.062</td><td>0.135</td><td>0.144</td><td>0.249</td><td>0.077</td><td>0.167</td><td>0.178</td><td>0.165</td><td>0.183</td><td>0.266</td><td>0.099</td><td>0.178</td><td>0.108</td><td>0.196</td><td>0.216</td><td>0.280</td><td>0.265</td><td>0.343</td><td>0.157</td><td>0.266</td><td>0.345</td><td>0.412</td></tr><tr><td>72</td><td>0.082</td><td>0.155</td><td>0.151</td><td>0.251</td><td>0.094</td><td>0.184</td><td>0.236</td><td>0.193</td><td>0.226</td><td>0.287</td><td>0.111</td><td>0.189</td><td>0.129</td><td>0.209</td><td>0.360</td><td>0.370</td><td>0.284</td><td>0.360</td><td>0.183</td><td>0.291</td><td>0.437</td><td>0.471</td></tr><tr><td>144</td><td>0.093</td><td>0.166</td><td>0.154</td><td>0.251</td><td>0.101</td><td>0.190</td><td>0.260</td><td>0.214</td><td>0.240</td><td>0.294</td><td>0.115</td><td>0.189</td><td>0.138</td><td>0.215</td><td>0.410</td><td>0.391</td><td>0.379</td><td>0.441</td><td>0.194</td><td>0.296</td><td>0.501</td><td>0.518</td></tr><tr><td></td><td>Avg</td><td>0.072</td><td>0.144</td><td>0.143</td><td>0.245</td><td>0.084</td><td>0.174</td><td>0.207</td><td>0.179</td><td>0.208</td><td>0.278</td><td>0.105</td><td>0.182</td><td>0.116</td><td>0.200</td><td>0.288</td><td>0.322</td><td>0.300</td><td>0.373</td><td>0.166</td><td>0.271</td><td>0.417</td><td>0.460</td></tr><tr><td rowspan="4">Customer</td><td>12</td><td>0.065</td><td>0.129</td><td>0.191</td><td>0.247</td><td>0.091</td><td>0.160</td><td>0.243</td><td>0.156</td><td>0.267</td><td>0.289</td><td>0.123</td><td>0.180</td><td>0.143</td><td>0.195</td><td>0.310</td><td>0.326</td><td>0.309</td><td>0.366</td><td>0.175</td><td>0.243</td><td>0.640</td><td>0.580</td></tr><tr><td>24</td><td>0.078</td><td>0.141</td><td>0.214</td><td>0.264</td><td>0.107</td><td>0.173</td><td>0.293</td><td>0.177</td><td>0.267</td><td>0.291</td><td>0.130</td><td>0.183</td><td>0.170</td><td>0.212</td><td>0.338</td><td>0.344</td><td>0.313</td><td>0.369</td><td>0.188</td><td>0.264</td><td>0.763</td><td>0.642</td></tr><tr><td>72</td><td>0.108</td><td>0.161</td><td>0.222</td><td>0.266</td><td>0.131</td><td>0.190</td><td>0.331</td><td>0.215</td><td>0.334</td><td>0.317</td><td>0.149</td><td>0.196</td><td>0.202</td><td>0.228</td><td>0.511</td><td>0.408</td><td>0.330</td><td>0.374</td><td>0.267</td><td>0.324</td><td>0.616</td><td>0.564</td></tr><tr><td>144</td><td>0.126</td><td>0.172</td><td>0.227</td><td>0.268</td><td>0.141</td><td>0.195</td><td>0.368</td><td>0.226</td><td>0.363</td><td>0.332</td><td>0.166</td><td>0.206</td><td>0.222</td><td>0.239</td><td>0.687</td><td>0.461</td><td>0.450</td><td>0.456</td><td>0.336</td><td>0.373</td><td>0.658</td><td>0.586</td></tr><tr><td></td><td>Avg</td><td>0.094</td><td>0.150</td><td>0.214</td><td>0.261</td><td>0.118</td><td>0.180</td><td>0.309</td><td>0.194</td><td>0.308</td><td>0.307</td><td>0.142</td><td>0.191</td><td>0.184</td><td>0.219</td><td>0.461</td><td>0.385</td><td>0.350</td><td>0.391</td><td>0.242</td><td>0.301</td><td>0.669</td><td>0.593</td></tr><tr><td colspan="2"><eq>1^{st}</eq>Count</td><td>28</td><td>27</td><td>0</td><td>0</td><td>0</td><td>0</td><td colspan="16">3</td></tr></table>

## G DISCUSSIONS AND FURTHER IMPROVEMENT

## G.1 DISCUSSIONS ON ARCHITECTURE-FREE METHODS

Channel Independence (CI) (Nie et al., 2023), regarding variates of time series independently and adopting the shared backbone, have gained increasing popularity in forecasting with performance promotions as an architecture-free method. Recent works (Han et al., 2023; Li et al., 2023) found that while Channel Dependence (CD) benefits from a higher capacity ideally, CI can greatly boost the performance because of sample scarcity, since most of the current forecasting benchmarks are not large enough. We think it is essential to make variates independent, especially when there are potential risks of embedding as mentioned in Appendix E.3, inducing the ideal model capacity of CD limited by the excessively localized receptive field. However, the essence of CI, regarding multivariate time series univariately, can lead to time-consuming training and inference and become an obstacle to scalability. Still, multivariate correlations can not be explicitly utilized. Perpendicular to these works, iTransformer repurposes an architecture with the native Transformer modules to tackle the issues. 

RevIN (Kim et al., 2021) and Stationarization (Liu et al., 2022b) have been widely applied for the distribution shift (non-stationarity) as architecture-free techniques. These works strive to reveal the temporal dependency better. This is accomplished by layer normalization in iTransformer and still leaves further improvement for us to tackle the distribution shift. 

## G.2 DISCUSSIONS ON LINEAR FORECASTERS

Linear forecasters have natural advantages in modeling temporal dependencies. The dense weighting (Zeng et al., 2023; Li et al., 2023) can reveal measurement-free relationships among the time points of the same variate. More advanced linear forecasters focus on structural point-wise modeling (Oreshkin et al., 2019; Liu et al., 2022a; 2023). By contrast, iTransformer is particularly good at forecasting high-dimensional time series (numerous variates with complicated correlations, which can be common and realistic for practitioners in real forecasting applications). For variate correlating, the embedding keeps the variate independent and the attention module can be applied to dig it out. Under univariate scenarios, iTransformer actually becomes a stackable linear forecaster (attention degradation), which leaves further enhancement to exploit the temporal dependency better. 

## G.3 DISCUSSIONS ON TRANSFORMERS

We emphasize that iTransformer actually proposes a new perspective to think about the multivariate time series modality, specifically, how to consider the variates and the tokenization. We list several representatives in Figure 19. Transformer treats time series as the natural language but the timealigned embedding may bring about risks in multi-dimensional series. The problem can be alleviated by expanding the receptive field. Although it is believed that Patching (Zhang & Yan, 2023; Nie et al., 2023) can be more fine-grained, it also brings higher computational complexity and the potential interaction noise between time-unaligned patches. If the current embedding (implemented by MLP) is enhanced with more inductive bias (such as TCN), it may handle more robust cases with the variate token paradigm and enjoy the flexibility of Transformer with changeable numbers of tokens. 

We believe the capability and scalability of Transformer have stood the test by extensive fields, but there is still improvement room to elaborately design components based on the inverted architecture, such as efficient attention for multivariate correlation, structural temporal dependency modeling under distribution shift, fine-grained variate tokenization and well-designed embedding mechanisms. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/a3ad66e89ecf82f469ef5e26cb5ca182fe341880353f61cf2718c54f3d0fba84.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/953130dce95aa68f38dff1e6e686dcfb01a09ffc371c96ec5e38b2f79edc2fdf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/ab125b2a-7d36-43af-bb9b-ff4765a96aab/006dd33c8ba0f272b7511646abfe6b87b245426037cb3f0761c12209dbc68d4e.jpg)



Figure 19: Tokenizations for multivariate time series modality of representative Transformers.
