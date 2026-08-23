# TIMER-XL: LONG-CONTEXT TRANSFORMERS FOR UNIFIED TIME SERIES FORECASTING

Yong Liu<sup>∗</sup>, Guo Qin<sup>∗</sup>, Xiangdong Huang, Jianmin Wang, Mingsheng Long School of Software, BNRist, Tsinghua University, Beijing 100084, China {liuyong21,qinguo24}@mails.tsinghua.edu.cn {huangxdong,jimwang,mingsheng}@tsinghua.edu.cn 

## ABSTRACT

We present Timer-XL, a causal Transformer for unified time series forecasting. To uniformly predict multidimensional time series, we generalize next token prediction, predominantly adopted for 1D token sequences, to multivariate next token prediction. The paradigm formulates various forecasting tasks as a long-context prediction problem. We opt for decoder-only Transformers that capture causal dependencies from varying-length contexts for unified forecasting, making predictions on non-stationary univariate time series, multivariate series with complicated dynamics and correlations, as well as covariate-informed contexts that include exogenous variables. Technically, we propose a universal TimeAttention to capture fine-grained intra- and inter-series dependencies of flattened time series tokens (patches), which is further enhanced by deft position embedding for temporal causality and variable equivalence. Timer-XL achieves state-of-the-art performance across task-specific forecasting benchmarks through a unified approach. Based on large-scale pre-training, Timer-XL achieves state-of-the-art zero-shot performance, making it a promising architecture for pre-trained time series models. Code is available at this repository: https://github.com/thuml/Timer-XL. 

## 1 INTRODUCTION

Transformers have been extensively applied to time series forecasting, becoming the backbone of task-specific models (Zhou et al., 2021; Wu et al., 2021) and pre-trained models (Das et al., 2023). While the majority of prior works have focused on long-term forecasting, reliable predictions are made by considering endogenous variations and exogenous correlations in the context (Box, 2013). Besides, the context length of pre-trained Transformers determines the maximum input and output length during inference. Therefore, long-context Transformers are more versatile than shorter ones, facilitating long-sequence and high-resolution generation (Yin et al., 2023; Wang et al., 2024a). 

However, existing Transformers in the time series field crucially encounter the context bottleneck. As shown in Figure 1, unlike Transformers for natural language and vision that learn dependencies among thousands to millions of tokens (Kirillov et al., 2023; OpenAI, 2023), time-series Transformers typically operate around limited contexts of up to hundreds of time series tokens (patches) (Nie et al., 2022). For univariate forecasting, a short-context input leads to insufficient learning of global tendencies, struggling to address non-stationarity in real-world time series (Hyndman, 2018). For multivariate forecasting, increasing research has demonstrated the effectiveness of explicitly capturing intra- and inter-channel dependencies (Zhang & Yan, 2022; Liu et al., 2023; 2024a), highlighting the practical urgency of extending the context length to encompass inter-correlated time series. 

Recently, causal Transformers characterized by the decoder-only architecture have become a predominant choice of large language models (Zhao et al., 2023) and garnered increasing attention in the development of large time series models (Rasul et al., 2023; Ansari et al., 2024). Based on contextual flexibility and autoregressive next token prediction, one model can accommodate varying lookback and prediction lengths (Liu et al., 2024b). Therefore, pre-training on longer contexts not only empowers them with the fundamental capability to incorporate more contextual information but also enhances the model versatility toward a one-for-all foundation model. Regarding any-variate and any-length time series as one context, previous work (Liu et al., 2024a) has achieved unified modeling on flattened tokens based on noncausal Transformers. However, our empirical results (Figure 3) reveal that encoder-only forecasters may encounter performance degradation in long-context forecasting, while decoder-only Transformers can mitigate this degradation well. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/7a4617c1b65047a3f5a072118ede40670c11bc38427031ce84d210157f5774be.jpg)



Figure 1: We compare the context length (measured by token number) of Transformers in different modalities and propose Timer-XL that increases the length to thousands of patch tokens. Given the generality across contexts, Timer-XL is a versatile solution for various forecasting tasks.


In this work, we generalize the training objective of language modeling to multivariate next token prediction, achieving unified time series forecasting that covers tasks in Figure 1 (right). Based on the decoder-only architecture, we propose TimeAttention to facilitate Transformers on multidimensional time series, presenting Kronecker-based masking mechanism to train time-series Transformers in a channel-dependent approach. With specialized position embedding for multivariate series, TimeAtten tion is aware of the chronological order of time points and achieves permutation-equivalence (Zaheer et al., 2017) on variables. We enlarge the context to thousands of patch tokens and achieve state-ofthe-art on univariate, multivariate, and covariate-informed forecasting benchmarks. By pre-training on large-scale datasets, we present Timer-XL as an extra long version of pre-trained time-series Trans formers (Timer) (Liu et al., 2024c), which outperforms recent large models in zero-shot forecasting. Our contributions lie in three aspects: 

• We propose multivariate next token prediction and unified time series forecasting, strengthening Transformers with enlarged contexts to make information-complete predictions. 

• We introduce TimeAttention, a novel causal self-attention tailored for multidimensional time series, facilitating intra- and inter-series modeling with positional awareness and maintaining causality and scalability of Transformers. 

• We propose Timer-XL, a versatile Transformer for one-for-all forecasting, which mitigates performance degradation in long-context time series, achieves state-of-the-art performance in task-specific benchmarks, and presents notable zero-shot performance by pre-training. 

## 2 RELATED WORK

Transformers (Vaswani et al., 2017) for time series forecasting have undergone rapid advancements. Initial Transformer-based forecasters primarily focused on long-term forecasting (Li et al., 2019; Zhou et al., 2021; Wu et al., 2021; Sun & Zhang, 2024). However, the context length is not growing in pace, which hinders Transformers from making information-complete predictions. Another advancement has focused on multivariate forecasting. Unlike natural language, time series are multidimensional and inherently correlated (Hyndman, 2018). To learn intra- and inter-series dependencies, different tokenization of time-series Transformers has been proposed, including point-wise (Lim et al., 2021), patch-wise (Nie et al., 2022), and variable-wise (Liu et al., 2023) approaches, with deftly tailored architectures (Zhang & Yan, 2022; Wang et al., 2024b). However, few works highlight that multidimensional time series can be uniformly tackled by long-context Transformers without architectural modification. In this work, we leverage causal Transformers, which excel at handling long-context sequences, and unify time series forecasting tasks into multivariate next token prediction. 

Recently, time-series Transformers have experienced the evolution from small task-specific models to pre-trained large models (Das et al., 2023; Woo et al., 2024; Ansari et al., 2024). Among them, decoder-only Transformer is predominantly adopted as the backbone of large language models (Touvron et al., 2023; OpenAI, 2023), positioning as a scalable choice for general time series analysis (Liu et al., 2024c). By independently predicting each token with supervision, decoder-only models are also multi-length forecasters (Liu et al., 2024b), avoiding resource-intensive training and lookback-search. However, existing decoder-only Transformers are generally pre-trained in a channel-independent approach, making them inaccessible to inter-series dependencies. 

Prior work has employed encoder-only Transformers to capture dependencies of multivariate time series (Liu et al., 2024a). However, our empirical study found that this architecture can be incompatible with causal forecasting, limiting the performance of Transformers. To implement next token prediction and multivariate forecasting in a single Transformer, we renovate the attention module, which disentangles fine-grained token dependencies into variable dependencies and temporal causal masks, capturing intra- and inter-series dependencies with causality and scalability maintained. In Table 1, we list representative time-series Transformers and highlight their differences. 


Table 1: Comparison among representative time-series Transformers.


<table><tr><td>Model</td><td>PatchTST(2022)</td><td>iTrans.(2023)</td><td>TimeXer(2024b)</td><td>UniTST(2024a)</td><td>Moirai(2024)</td><td>Timer(2024c)</td><td>Timer-XL(Ours)</td></tr><tr><td>Intra-Series</td><td>✓</td><td>✕</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>Inter-Series</td><td>✕</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Causal Trm.</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✓</td><td>✓</td></tr><tr><td>Pre-Trained</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

## 3 APPROACH

In this section, we first introduce a decoder-only Transformer to illustrate the procedure of next token prediction on univariate time series. As an extension, we design TimeAttention and propose Timer-XL for unified time series forecasting. It is applicable to univariate, multivariate, and covariate-informed scenarios by generalizing the context from 1D sequences to 2D time series. 

## 3.1 TIMER

Timer (Liu et al., 2024c) is a time-series Transformer trained by next token prediction (Bengio et al., 2000), which regards single-dimensional time series as non-overlapping patch tokens. 

Next Token Prediction Given an univariate time series $\mathbf { X } = \{ x _ { 1 } , \dots , x _ { T P } \}$ of length TP, a time series token is defined as P consecutive time points, also termed as the patch token: 

$$
\mathbf {x} _ {i} = \left\{x _ {(i - 1) P + 1}, \dots , x _ {i P} \right\} \in \mathbb {R} ^ {P}, i = 1, \dots , T.\tag{1}
$$

The training objective is to independently predict the next patch token to maximize the likelihood: 

$$
P (\mathbf {X}) = \prod_ {i = 1} ^ {T} p (\mathbf {x} _ {i + 1} | \mathbf {x} _ {\leq i}),\tag{2}
$$

which is realized by a decoder-only architecture with the block number L and model dimension D: 

$$
\mathbf {h} _ {i} ^ {0} = \mathbf {W} _ {e} \mathbf {x} _ {i}, i = 1, \ldots , T,
$$

$$
\mathbf {H} ^ {l} = \operatorname{TrmBlock} (\mathbf {H} ^ {l - 1}), l = 1, \dots , L,\tag{3}
$$

$$
\{\hat {\mathbf {x}} _ {i + 1} \} = \mathbf {H} ^ {L} \mathbf {W} _ {d}, i = 1, \dots , T.
$$

For simplicity, we omit the block index l. Timer adopts $\mathbf { W } _ { e } , \mathbf { W } _ { d } \in \mathbb { R } ^ { D \times P }$ that independently embed and project the token embeddings as ${ \bf H } = \{ { \bf h } _ { i } \} \in \mathbb { R } ^ { \mathbf { \bar { \Gamma } } \times D }$ . TrmBlock includes feed-forward network and self-attention with the temporal causal mask $\mathcal { T } \in \mathbb { R } ^ { T \times T } . \ \mathbf { h } _ { i } \in \mathbb { R } ^ { D }$ is the context representation of the previous i tokens. All predicted $\hat { \mathbf { x } } _ { i + 1 }$ are supervised with ground truth via MSE loss. 

## 3.2 GENERALIZE 1D SEQUENCES TO 2D TIME SERIES

For the enlarged context with the additional dimension, our proposed attention mechanism aims to (1) thoroughly capture intra- and inter-series dependencies and (2) preserve causality within the temporal dimension. Without loss of generality, we illustrate this with the case of multivariate forecasting. 

Multivariate Next Token Prediction Given a multivariate time series $\mathbf { X } \in \mathbb { R } ^ { N \times T P }$ with the number of variables N, the time series token $\mathbf { x } _ { m , i }$ is defined as the i-th patch of the m-th variable: 

$$
\mathbf {x} _ {m, i} = \{\mathbf {X} _ {m, (i - 1) P + 1}, \ldots , \mathbf {X} _ {m, i P} \} \in \mathbb {R} ^ {P}, m = 1, \ldots , N, i = 1, \ldots , T.\tag{4}
$$

The training objective is still to independently predict the next token. Unlike before, each prediction is made based on tokens of the previous time $( \leq i )$ from all $N$ variables: 

$$
P (\mathbf {X}) = \prod_ {m = 1} ^ {N} \prod_ {i = 1} ^ {T} p \left(\mathbf {x} _ {m, i + 1} \mid \mathbf {x} _ {:, \leq i}\right) = \prod_ {m = 1} ^ {N} \prod_ {i = 1} ^ {T} p \left(\mathbf {x} _ {m, i + 1} \mid \mathbf {x} _ {1, \leq i}, \dots , \mathbf {x} _ {N, \leq i}\right).\tag{5}
$$

Compared with Equation $^ { 2 , }$ the multivariate context length increases from $T$ to $N T$ . By contrast, the benefit is that this paradigm learns causal dependencies within each sequence while incorporating exogenous variable correlations from other sequences, making it a universal forecasting paradigm that outperforms channel-independent (Nie et al., 2022) or variable-centric models (Liu et al., 2023). 

Technically, we independently apply $\mathbf { W } _ { e } \in \mathbb { R } ^ { D \times P }$ on each token to obtain patch-wise representation $\mathbf { h } _ { m , i } \in \mathbb { R } ^ { \breve { D } }$ , which will encompass contextual information from $N i$ tokens through Transformer blocks and be eventually projected by $\mathbf { W } _ { d } \in \mathbb { R } ^ { D \times P }$ into the predicted patch token $\hat { \mathbf { x } } _ { m , i + 1 }$ 

Position Embedding Position embedding has not been sufficiently explored in time-series Transformers. To avoid inherent permutation-invariance of self-attention, positional embedding is required to reflect the chronological order of tokens on the temporal dimension. As for the variable dimension, shuffling the input order of variables should not affect anything other than the output order of variables. Formally, the processing on multiple variables should be permutation-equivalent (Zaheer et al., 2017). 

To meet the above requirements, we adopt RoPE (Su et al., 2024), a widely utilized position embed ding on the temporal dimension. For the variable dimension, we use two learnable scalars in each head to keep the permutation-equivalence of variables (Woo et al., 2024). Beyond simply incorporating them together, we provide detailed ablations in Section E.3 to demonstrate the effectiveness: 

$$
\mathcal {A} _ {m n, i j} = \mathbf {h} _ {m, i} ^ {\top} \mathbf {W} _ {q} \mathbf {R} _ {\theta , i - j} \mathbf {W} _ {k} ^ {\top} \mathbf {h} _ {n, j} + u \cdot \mathbb {1} (m = n) + v \cdot \mathbb {1} (m \neq n),\tag{6}
$$

where $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } \in \mathbb { R } ^ { D \times d _ { k } }$ and $d _ { k }$ is the dimension of the query, key, and value. $\mathbf { R } _ { \theta , t } \in \mathbb { R } ^ { d _ { k } \times d _ { k } }$ is the rotary matrix with rotation degree $t \cdot \theta , \mathbb { 1 } ( \cdot )$ is the indicator function, and $u , v \in$ R are learnable parameters for the token to distinguish its endogenous and exogenous time series. 

TimeAttention In contrast to variable-wise (Liu et al., 2023) and non-causal patch-wise tokens (Nie et al., 2022; Woo et al., 2024), our TimeAttention aims to capture causal patch-wise dependencies within and among all variables. Concretely, we sort patch tokens by flattening their 2D indices into 1D indices in the temporal-first manner, which is illustrated in the upper left of Figure 2. Note that the order of variables does not matter, since Equation 6 guarantees their permutation-equivalence. 

We provide an intuitive example to illustrate the causal dependencies within multivariate time series: considering the 2nd token of time series A. To predict its next token, its representation h should be exactly dependent on the tokens- 1, 2, 4, 5 . Similarly, we provide all causal dependencies of each token in Figure 12. Based on the visualized attention mask and variable dependencies presented in Figure 2, where all variables are inter-correlated, all token dependencies in $\mathcal { A }$ can be formally disentangled by the Kronecker product into (1) the adjacency matrix of the variable dependency graph $\mathcal { C } \in \mathbb { R } ^ { N \times N }$ and (2) the causal temporal mask $\mathcal { T } \overset { v } { \in } \mathbb { R } ^ { T \times \check { T } }$ 

$$
\mathcal {T} _ {i, j} = \left\{ \begin{array}{l l} 1 & \text { if } j \leq i, \\ 0 & \text { otherwise }, \end{array} \right. \mathcal {C} _ {m, n} = \left\{ \begin{array}{l l} 1 & \text { if   variable } m \text { is   dependent   on } n, \\ 0 & \text { otherwise }. \end{array} \right.\tag{7}
$$

Let the Kronecker product $\otimes : \left( \mathbb { R } ^ { N \times N } , \mathbb { R } ^ { T \times T } \right) \mapsto \mathbb { R } ^ { N T \times N T }$ take two matrices and produce a block matrix. Consequently, TimeAttention is formulated as follows: 

$$
\text {TimeAttention} (\mathbf {H}) = \text {Softmax} \left(\frac {\text {Mask} (\mathcal {C} \otimes \mathcal {T}) + \mathcal {A}}{\sqrt {d _ {k}}}\right) \mathbf {H} \mathbf {W} _ {v}, \text {Mask} (\mathcal {M}) = \left\{ \begin{array}{l l} 0 & \text {if} \mathcal {M} _ {i, j} = 1, \\ - \infty & \text {if} \mathcal {M} _ {i, j} = 0. \end{array} \right.\tag{8}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/96572b515f6b3bbfe57a817350fbc520406195f0d6c93af0c4835e066ee3601b.jpg)



Figure 2: Illustration of TimeAttention. For univariate series, temporal mask keeps the causality. Given multivariate patch tokens sorted in a temporal-first order, we adopt the variable dependencies , an all-one matrix, as the left-operand of Kronecker product, expanding temporal mask to a block matrix, which exactly reflects dependencies of multivariate next token prediction. The formulation is also generalizable to univariate and covariate-informed contexts with pre-defined variable dependency.


Eventually, token representations in ${ \bf H } = \{ { \bf h } _ { m , i } \} \in \mathbb { R } ^ { N T \times D }$ will be independently processed by feed-forward network and layer normalization, and fed into the next Transformer block. 

Unified Time Series Forecasting In multivariate forecasting, the variable dependency forms the complete graph, presenting an all-one matrix . By generalizing TimeAttention on multiple sequences, Transformers can leverage its length-flexibility to encompass relevant covariates as well. In this case, Timer-XL is adapted in two steps: (1) formulate the customized variable dependency as and (2) optimize the model using the supervision of target variables. An example (target-A-covariate-B) of TimeAttention is illustrated on the right of Figure 2. In a nutshell, we adopt position embeddings for the temporal and variable dimensions. To achieve unified time series forecasting, we flatten 2D time series into a unified context and capture fine-grained causal token dependencies. 

## 4 EXPERIMENTS

We conduct evaluations of Timer-XL in three aspects, including (1) supervised training as a taskspecific forecaster, (2) large-scale pre-training as a zero-shot forecaster, and (3) assessing the effectiveness of TimeAttention and model efficiency. Given that the long-context forecasting paradigm receives less attention in the community, which can be concealed due to the performance saturation on previous benchmarks (Makridakis et al., 2020; Wu et al., 2022), we established new long-context forecasting benchmarks. Detailed experimental configurations are provided in Appendix B. 

## 4.1 UNIVARIATE TIME SERIES FORECASTING

Setups Due to the insufficient dataset length when extending contexts in univariate datasets (Makridakis et al., 2020), we adopt multivariate datasets from Liu et al. (2023). Although these datasets are originally multivariate, they aim to be predicted in a univariate approach with the implementation of channel independence. Different from the previous long-term forecasting setting, we focus on reliable prediction based on a long context. Therefore, we fix the prediction horizon and increase the lookback length to monthly and yearly levels. We also establish a long-context univariate benchmark based on the challenging 40-year ECMWF Reanalysis v5 dataset (Hersbach et al., 2020), where yearly contexts are adopted to predict the land-surface temperature of a single site (ERA5-S). 

Results As shown in Figure 3, the accuracy of univariate prediction can generally be improved by extending the daily context to monthly. We draw a similar conclusion on ERA5 (Table 15), where extending the context consistently helps in the specific model architecture. Notably, Timer-XL with decoder-only architecture outperforms encoder-only Transformer and linear forecaster in excessively long contexts. Further, we conduct representation analysis in Appendix E.4, revealing that Timer-XL is proficient at adaptively selecting information in vast observations and thus achieves breakthrough performance. It is also noteworthy that the performance of monthly and yearly contexts improves slowly and deteriorates, which may stem from increased noise and training difficulty inherent in data, which leaves a future direction to improve the context efficiency. Table 2 provides results on ERA5-S. Timer-XL consistently outperforms PatchTST on all sites, which can be credited to the maintenance of causality and token-wise supervision in the decoder-only architecture. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/d596f62897e3201735bdad5aa5fe9629575fff63d29dd4ee74eeab18eb87a28d.jpg)



Figure 3: Univariate forecasting (pred-96) of well-acknowledged benchmarks under channel independence (Nie et al., 2022). We increase the lookback length to encompass monthly and yearly contexts.


Non-stationary Forecasting We delve into widespread non-stationarity in univariate tasks. It is commonly tackled by normalization (Kim et al., 2021) that greatly improves Transformer performance in previous benchmarks. However, we find it may be caused by the insufficient time span and training samples in these datasets. While normalization simplifies learning by aligning series with different means and variances to the same distribution, it limits the model capacity of Transformers, preventing them from learning variations among windows. The by-product can be mode collapse and oversmooth predictions. In Table 2 and Table 16, we evaluate the performance on ERA5 and datasets from Wu et al. (2022), which validates that Timer-XL can achieve better results even without instance normalization. 


Table 2: Univariate forecasting (input-3072-pred-96) of ERA5-S, encompassing 117k time points in each station (40-years). We evaluate PatchTST and Timer-XL with and without normalization (Kim et al., 2021). + Norm. indicates using the normalization. We train one model for each site separately.


<table><tr><td>Station</td><td colspan="2">Beijing</td><td colspan="2">Hongkong</td><td colspan="2">London</td><td colspan="2">New York</td><td colspan="2">Paris</td><td colspan="2">Seoul</td><td colspan="2">Shanghai</td><td colspan="2">Average</td></tr><tr><td>Model</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="2">PatchTST+ Norm.</td><td>0.0791</td><td>0.221</td><td>0.189</td><td>0.327</td><td>0.277</td><td>0.415</td><td>0.186</td><td>0.334</td><td>0.266</td><td>0.407</td><td>0.0940</td><td>0.238</td><td>0.137</td><td>0.289</td><td>0.175</td><td>0.319</td></tr><tr><td>0.0797</td><td>0.220</td><td>0.191</td><td>0.323</td><td>0.281</td><td>0.419</td><td>0.184</td><td>0.334</td><td>0.272</td><td>0.411</td><td>0.0914</td><td>0.233</td><td>0.136</td><td>0.287</td><td>0.176</td><td>0.319</td></tr><tr><td rowspan="2">Timer-XL+ Norm.</td><td>0.0739</td><td>0.210</td><td>0.179</td><td>0.316</td><td>0.262</td><td>0.404</td><td>0.182</td><td>0.327</td><td>0.254</td><td>0.399</td><td>0.0901</td><td>0.229</td><td>0.134</td><td>0.282</td><td>0.168</td><td>0.310</td></tr><tr><td>0.0742</td><td>0.210</td><td>0.183</td><td>0.317</td><td>0.278</td><td>0.418</td><td>0.181</td><td>0.330</td><td>0.264</td><td>0.407</td><td>0.0896</td><td>0.227</td><td>0.133</td><td>0.281</td><td>0.172</td><td>0.313</td></tr></table>

## 4.2 MULTIVARIATE TIME SERIES FORECASTING

Setups We follow iTransformer (Liu et al., 2023) to evaluate multivariate forecasting performance. Toward a one-for-all forecaster, we evaluate performance of rolling forecast, that is, we trained one model for all prediction horizons by integrating the previous prediction into the lookback window in the next iteration. We further establish long-context multivariate forecasting benchmarks: ERA5 multi-station land-surface temperature prediction (ERA5-MS), and the global temperature and wind speed forecasting challenge (GTWSF) (Wu et al., 2023), to learn complex temporal dynamics and variable correlations with sufficient training samples. 

Results As shown in Tables 3-4 and Figure 4, Timer-XL achieves the best results on both previous and new benchmarks. Essentially, Transformers that explicitly capture inter-series dependencies, such as UniTST (Liu et al., 2024a) and iTransformer, reasonably achieve decent performance in Table 3. Beyond iTransformer, Timer-XL can model fine-grained patch-wise temporal dependencies. With 

TimeAttention, Timer-XL outperforms Timer especially on high-dimensional time series (13.2% in ECL and 6.3% in Traffic, with thousands of tokens in the context). Compared with the encoder-only UniTST, decoder-only Transformers excel at generalizing across varying prediction lengths in Table 4. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/9e0aefff28c29fe34e615e5d58902e94abf08af0a39c9d71cdc1b3316a98e360.jpg)



Figure 4: Multivariate forecasting of GTWSF (2-day-pred-1-day), involving 3850 worldwide stations spanning two years. Results of the baseline models are officially reported by Ding et al. (2024).



Table 3: Multivariate forecasting (96-pred-96) of well-acknowledged benchmarks. All models are trained from scratch. Results of baseline models are officially reported by Liu et al. (2023).


<table><tr><td>Models</td><td colspan="2">Timer-XL(Ours)</td><td colspan="2">Timer(2024c)</td><td colspan="2">UniTST(2024a)</td><td colspan="2">iTransformer(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">PatchTST(2022)</td><td colspan="2">TimesNet(2022)</td><td colspan="2">Stationary(2022b)</td><td colspan="2">Autoformer(2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ECL</td><td>0.138</td><td>0.233</td><td>0.159</td><td>0.244</td><td>0.139</td><td>0.235</td><td>0.148</td><td>0.240</td><td>0.197</td><td>0.282</td><td>0.181</td><td>0.270</td><td>0.168</td><td>0.272</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td></tr><tr><td>ETTh1</td><td>0.381</td><td>0.399</td><td>0.386</td><td>0.401</td><td>0.385</td><td>0.402</td><td>0.386</td><td>0.405</td><td>0.386</td><td>0.400</td><td>0.414</td><td>0.419</td><td>0.384</td><td>0.402</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td></tr><tr><td>Traffic</td><td>0.387</td><td>0.260</td><td>0.413</td><td>0.265</td><td>0.389</td><td>0.265</td><td>0.395</td><td>0.268</td><td>0.650</td><td>0.396</td><td>0.462</td><td>0.295</td><td>0.593</td><td>0.321</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td></tr><tr><td>Weather</td><td>0.165</td><td>0.209</td><td>0.176</td><td>0.215</td><td>0.165</td><td>0.210</td><td>0.174</td><td>0.214</td><td>0.196</td><td>0.255</td><td>0.177</td><td>0.218</td><td>0.172</td><td>0.220</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td></tr><tr><td>Solar-Energy</td><td>0.200</td><td>0.229</td><td>0.204</td><td>0.234</td><td>0.203</td><td>0.232</td><td>0.203</td><td>0.237</td><td>0.290</td><td>0.378</td><td>0.234</td><td>0.286</td><td>0.250</td><td>0.292</td><td>0.215</td><td>0.249</td><td>0.884</td><td>0.711</td></tr></table>


Table 4: Multivariate forecasting (672-pred- 96, 192, 336, 720 ) of well-acknowledged benchmarks. We evaluate one-for-all forecasters following Liu et al. (2024b): rolling forecasting for four forecast lengths with one model. Averaged results are reported here and full results are provided in Table 12.


<table><tr><td>Models</td><td colspan="2">Timer-XL (Ours)</td><td colspan="2">Timer (2024c)</td><td colspan="2">UniTST (2024a)</td><td colspan="2">iTransformer (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">PatchTST (2022)</td><td colspan="2">TimesNet (2022)</td><td colspan="2">Stationary (2022b)</td><td colspan="2">Autoformer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ECL</td><td>0.155</td><td>0.246</td><td>0.161</td><td>0.251</td><td>0.163</td><td>0.257</td><td>0.164</td><td>0.258</td><td>0.165</td><td>0.265</td><td>0.169</td><td>0.268</td><td>0.201</td><td>0.303</td><td>0.265</td><td>0.358</td><td>0.289</td><td>0.379</td></tr><tr><td>ETTh1</td><td>0.409</td><td>0.430</td><td>0.418</td><td>0.436</td><td>0.429</td><td>0.447</td><td>0.421</td><td>0.445</td><td>0.426</td><td>0.444</td><td>0.412</td><td>0.435</td><td>0.495</td><td>0.491</td><td>0.505</td><td>0.513</td><td>0.517</td><td>0.528</td></tr><tr><td>Traffic</td><td>0.374</td><td>0.255</td><td>0.384</td><td>0.259</td><td>0.385</td><td>0.265</td><td>0.384</td><td>0.274</td><td>0.423</td><td>0.298</td><td>0.391</td><td>0.275</td><td>0.602</td><td>0.322</td><td>0.630</td><td>0.347</td><td>0.684</td><td>0.433</td></tr><tr><td>Weather</td><td>0.240</td><td>0.273</td><td>0.232</td><td>0.270</td><td>0.231</td><td>0.272</td><td>0.266</td><td>0.291</td><td>0.239</td><td>0.291</td><td>0.226</td><td>0.268</td><td>0.264</td><td>0.293</td><td>0.308</td><td>0.329</td><td>0.435</td><td>0.455</td></tr><tr><td>Solar-Energy</td><td>0.198</td><td>0.249</td><td>0.233</td><td>0.249</td><td>0.241</td><td>0.275</td><td>0.213</td><td>0.291</td><td>0.222</td><td>0.283</td><td>0.202</td><td>0.269</td><td>0.213</td><td>0.295</td><td>0.254</td><td>0.315</td><td>0.265</td><td>0.325</td></tr></table>

Ablation Study Patching (Nie et al., 2022) has been demonstrated as an effective tokenization approach for time series, leading to the boom of Transformers in supervised deep forecasters and large time series models. To better cope with multivariate time series forecasting, we compared typical models on real-world benchmarks to address key questions: (1) whether to conduct explicit interseries modeling or not (channel independence) and (2) whether to use decoder-only or encoder-only Transformers. The combination presents four Transformers in Table 5, which shows that Timer-XL combines the advantages of explicit inter-series modeling and the decoder-only architecture, which is suitable for multivariate time series forecasting with sufficient training samples. 

## 4.3 COVARIATE-INFORMED TIME SERIES FORECASTING

Setups For the covariate-informed forecasting, we adopt the well-acknowledged electricity price forecasting (EPF) task (Lago et al., 2021). Each subset contains electricity price as the endogenous variable and two exogenous variables. Therefore, the variable dependency for Timer-XL is formulated 


Table 5: Multivariate forecasting (input-3072-pred-96) of ERA5-MS (40 years and 7 stations). We fairly evaluate Transformers that adopt patched time series. CI. indicates whether the Transformer uses channel independence (Nie et al., 2022). Arch. categorizes them into the encoder-only (E) and decoder-only (D) architectures. Different from ERA5-S in Table 2, we train one model for all sites.


<table><tr><td colspan="3">Station</td><td colspan="2">Beijing</td><td colspan="2">Hongkong</td><td colspan="2">London</td><td colspan="2">New York</td><td colspan="2">Paris</td><td colspan="2">Seoul</td><td colspan="2">Shanghai</td><td colspan="2">Average</td></tr><tr><td>Model</td><td>CI.</td><td>Arch.</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>PatchTST</td><td>Yes</td><td>E</td><td>0.0815</td><td>0.222</td><td>0.190</td><td>0.326</td><td>0.275</td><td>0.414</td><td>0.185</td><td>0.333</td><td>0.265</td><td>0.407</td><td>0.0977</td><td>0.240</td><td>0.139</td><td>0.290</td><td>0.176</td><td>0.319</td></tr><tr><td>UniTST</td><td>No</td><td>E</td><td>0.0753</td><td>0.213</td><td>0.179</td><td>0.318</td><td>0.269</td><td>0.410</td><td>0.185</td><td>0.330</td><td>0.256</td><td>0.401</td><td>0.0901</td><td>0.230</td><td>0.135</td><td>0.284</td><td>0.170</td><td>0.312</td></tr><tr><td>Timer</td><td>Yes</td><td>D</td><td>0.0734</td><td>0.210</td><td>0.182</td><td>0.319</td><td>0.268</td><td>0.407</td><td>0.183</td><td>0.329</td><td>0.255</td><td>0.399</td><td>0.0877</td><td>0.226</td><td>0.132</td><td>0.281</td><td>0.169</td><td>0.310</td></tr><tr><td>Timer-XL</td><td>No</td><td>D</td><td>0.0736</td><td>0.209</td><td>0.174</td><td>0.309</td><td>0.263</td><td>0.404</td><td>0.182</td><td>0.327</td><td>0.252</td><td>0.396</td><td>0.0872</td><td>0.225</td><td>0.130</td><td>0.278</td><td>0.166</td><td>0.307</td></tr></table>


as  = [[1, 1, 1], [0, 1, 0], [0, 0, 1]]. To investigate whether to learn causal or noncausal patch-wise dependencies in covariates, we implement two versions of Timer-XL: the original one with temporal causal mask , and the noncausal one with replaced by an all-one matrix. 


Results As shown in Table 6, Timer-XL outperforms state-of-the-art models in covariate-informed tasks. Compared with TimeXer (Wang et al., 2024b), which treats an entire covariate as a token, Timer-XL learns fine-grained patch-wise dependencies. By the noncausal version of Timer-XL, we surprisingly find consistent conclusions with endogenous variables: results will be better if Timer-XL learns causal dependencies within exogenous variables. It again validates that next token prediction that maintains causality has a higher upper limit of performance. 


Table 6: Covariate-informed forecasting (168-pred-24) of EPF. We implement two versions of Timer-XL: Noncausal indicates that we do not maintain the causality within covariates by replacing temporal causal mask with all-one matrix. Results of baselines are officially reported by Wang et al. (2024b).


<table><tr><td>Models</td><td colspan="2">Timer-XL (Ours)</td><td colspan="2">Timer-XL (Noncausal)</td><td colspan="2">TimeXer (2024b)</td><td colspan="2">iTransformer (2023)</td><td colspan="2">DLinear (2023)</td><td colspan="2">PatchTST (2022)</td><td colspan="2">Crossformer (2022)</td><td colspan="2">TimesNet (2022)</td><td colspan="2">Autoformer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>NP</td><td>0.234</td><td>0.262</td><td>0.237</td><td>0.265</td><td>0.238</td><td>0.268</td><td>0.265</td><td>0.300</td><td>0.309</td><td>0.321</td><td>0.267</td><td>0.284</td><td>0.245</td><td>0.289</td><td>0.250</td><td>0.289</td><td>0.402</td><td>0.398</td></tr><tr><td>PJM</td><td>0.089</td><td>0.187</td><td>0.092</td><td>0.188</td><td>0.088</td><td>0.188</td><td>0.097</td><td>0.197</td><td>0.108</td><td>0.215</td><td>0.106</td><td>0.209</td><td>0.149</td><td>0.198</td><td>0.097</td><td>0.195</td><td>0.168</td><td>0.267</td></tr><tr><td>BE</td><td>0.371</td><td>0.243</td><td>0.410</td><td>0.279</td><td>0.379</td><td>0.243</td><td>0.394</td><td>0.270</td><td>0.463</td><td>0.313</td><td>0.403</td><td>0.264</td><td>0.436</td><td>0.294</td><td>0.419</td><td>0.288</td><td>0.500</td><td>0.333</td></tr><tr><td>FR</td><td>0.381</td><td>0.204</td><td>0.406</td><td>0.220</td><td>0.384</td><td>0.208</td><td>0.439</td><td>0.233</td><td>0.429</td><td>0.260</td><td>0.411</td><td>0.220</td><td>0.440</td><td>0.216</td><td>0.431</td><td>0.234</td><td>0.519</td><td>0.295</td></tr><tr><td>DE</td><td>0.434</td><td>0.415</td><td>0.435</td><td>0.415</td><td>0.440</td><td>0.418</td><td>0.479</td><td>0.443</td><td>0.520</td><td>0.463</td><td>0.461</td><td>0.432</td><td>0.540</td><td>0.423</td><td>0.502</td><td>0.446</td><td>0.674</td><td>0.544</td></tr><tr><td>Average</td><td>0.302</td><td>0.262</td><td>0.316</td><td>0.273</td><td>0.306</td><td>0.265</td><td>0.335</td><td>0.289</td><td>0.366</td><td>0.314</td><td>0.330</td><td>0.282</td><td>0.362</td><td>0.284</td><td>0.340</td><td>0.290</td><td>0.453</td><td>0.368</td></tr></table>

## 4.4 PRE-TRAINED TIME-SERIES TRANSFORMERS

Setups Pre-training enriches time-series Transformers with generalizable forecasting capabilities. The outcome large time series model can cope with widespread challenges of few-shot and zero-shot forecasting. In this section, we conduct univariate pre-training on UTSD (Liu et al., 2024c) and LOTSA (Woo et al., 2024) and evaluate zero-shot performance on benchmarks from Wu et al. (2022). We further conduct large-scale multivariate pre-training on our ERA5-Large dataset, which spans 40 years and encompasses 4920 stations. Subsequently, we evaluate three types of generalization results comparing PatchTST (encoder-only Transformer) and Timer-XL (decoder-only Transformer): pre-training on 80% stations and 80% time span and then forecast on the remaining stations (variable generalization), remaining time span (temporal generalization), and remaining split of time span and stations (variable and temporal generalization). To evaluate the benefit of pre-training with longer context, we compare the zero-shot performance of Timer (2024c) and Timer-XL, where the context length of pre-training is increased from 1440 to 2880. 

Results We compare generalization performance on ERA5-Large in the middle of Figure 5 (a). Timer-XL achieves better results than PatchTST in all cases, revealing that decoder-only architecture has stronger generalization capability. Figure 5 (b) compares zero-shot performance of two pretrained Transformers with different context lengths, where Timer-XL outperforms previous Timer on all benchmark datasets, validating that long-context pre-training enhances large time series models. In Table 7, we provide a comprehensive zero-shot evaluation under a comparable pre-training scale and model size, where Timer-XL achieves notable performance with better sample efficiency. The versatility and scalability make it a promising backbone of foundation models. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/04fe536027ada9ca1a9ca86577fab6831ee1a896de19f5c76ceb4668cbbe410f.jpg)



(a) ERA5 (Pred-96, 4920 Stations)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/c6f0c104b42e8b7eec057b29f5b90e16de8491485b7089e12760b04ec6aa0cd1.jpg)



(b) TSLib (Pred-96, UTSD Pre-Trained)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/9323a981d92b49445ea087a86d49bd0c8b6a375bf9760117c4cf113de07981d1.jpg)



Figure 5: Illustration of one-for-all generalization (left). Based on the contextual flexibility, Timer-XL can predict heterogeneous time series, indicating three directions of generalization shown on the left. We compare performance when generalizing across the time and variables (middle), and zero-shot results across datasets (right), emphasizing the benefit of long-context pre-training.



Table 7: Averaged results of zero-shot forecasting. A lower MSE or MAE indicates a better prediction. Corresponding prediction lengths include 96, 192, 336, 720 . Full results of all prediction lengths are provided in Table 13. 1<sup>st</sup> Count represents the number of wins achieved by a model under all prediction lengths and datasets. The detailed configuration of Timer- ${ \bf \delta } { \bf . } { \bf X } { \bf L } _ { B a s e }$ is provided in Table 11.


<table><tr><td>Models</td><td colspan="2">Timer-XL<eq>_{Base}</eq>(Ours)</td><td colspan="2">Time-MoE<eq>_{Base}</eq>(2024)</td><td colspan="2">Time-MoELarge(2024)</td><td colspan="2">Time-MoE<eq>_{Ultra}</eq>(2024)</td><td colspan="2">Moirai<eq>_{Small}</eq>(2024)</td><td colspan="2">Moirai<eq>_{Base}</eq>(2024)</td><td colspan="2">Moirai<eq>_{Large}</eq>(2024)</td><td colspan="2">TimesFM(2023)</td><td colspan="2">MOMENT(2024)</td><td colspan="2">Chronos<eq>_{Base}</eq>(2024)</td><td colspan="2">Chronos<eq>_{Large}</eq>(2024)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.373</td><td>0.392</td><td>0.394</td><td>0.415</td><td>0.376</td><td>0.405</td><td>0.356</td><td>0.391</td><td>0.436</td><td>0.410</td><td>0.406</td><td>0.385</td><td>0.422</td><td>0.391</td><td>0.433</td><td>0.418</td><td>0.670</td><td>0.536</td><td>0.645</td><td>0.500</td><td>0.555</td><td>0.465</td></tr><tr><td>ETTm2</td><td>0.273</td><td>0.336</td><td>0.317</td><td>0.365</td><td>0.316</td><td>0.361</td><td>0.288</td><td>0.344</td><td>0.307</td><td>0.347</td><td>0.311</td><td>0.337</td><td>0.329</td><td>0.343</td><td>0.328</td><td>0.346</td><td>0.316</td><td>0.365</td><td>0.310</td><td>0.350</td><td>0.295</td><td>0.338</td></tr><tr><td>ETTh1</td><td>0.404</td><td>0.417</td><td>0.400</td><td>0.424</td><td>0.394</td><td>0.419</td><td>0.412</td><td>0.426</td><td>0.428</td><td>0.427</td><td>0.417</td><td>0.419</td><td>0.480</td><td>0.439</td><td>0.473</td><td>0.443</td><td>0.683</td><td>0.566</td><td>0.591</td><td>0.468</td><td>0.588</td><td>0.466</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.388</td><td>0.366</td><td>0.404</td><td>0.405</td><td>0.415</td><td>0.371</td><td>0.399</td><td>0.361</td><td>0.384</td><td>0.362</td><td>0.382</td><td>0.367</td><td>0.377</td><td>0.392</td><td>0.406</td><td>0.361</td><td>0.409</td><td>0.405</td><td>0.410</td><td>0.455</td><td>0.427</td></tr><tr><td>ECL</td><td>0.174</td><td>0.278</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.218</td><td>0.303</td><td>0.187</td><td>0.274</td><td>0.186</td><td>0.270</td><td>-</td><td>-</td><td>0.765</td><td>0.686</td><td>0.214</td><td>0.278</td><td>0.204</td><td>0.273</td></tr><tr><td>Weather</td><td>0.256</td><td>0.294</td><td>0.265</td><td>0.297</td><td>0.270</td><td>0.300</td><td>0.256</td><td>0.288</td><td>0.275</td><td>0.286</td><td>0.287</td><td>0.281</td><td>0.264</td><td>0.273</td><td>-</td><td>-</td><td>0.294</td><td>0.326</td><td>0.292</td><td>0.315</td><td>0.279</td><td>0.306</td></tr><tr><td><eq>1^{st}Count</eq></td><td>15</td><td>10</td><td>2</td><td>1</td><td>3</td><td>0</td><td>10</td><td>7</td><td>0</td><td>0</td><td>0</td><td>5</td><td>1</td><td>10</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td></tr></table>


∗ Dataset for pre-training is not evaluated on corresponding models, which is denoted by a dash (−). 



∗ Traffic from (PEMS) is generally used during the pre-training of large models and thus not evaluated here. 



∗ Our model checkpoint is available at https://huggingface.co/thuml/timer-base-84m. 


## 4.5 MODEL ANALYSIS

Model Efficiency To evaluate the model efficiency of Timer-XL with respect to the context length, it is essential to recognize the distinct characteristics of time series data compared to 1D sequences. Unlike natural language, the time series modality is characterized by the variable number N and the input length. We adopt two representative multivariate datasets with different N, and provide the memory footprint and training speed under gradually prolonged input. We evaluate typical approaches to handle multivariate series: (1) Timer-XL and Moiria that adopt channel dependence; (2) Timer that adopts channel independence. Intuitively, the complexity of the first type is $\mathcal { O } ( N ^ { 2 } T ^ { 2 } )$ while the complexity of self-attention under channel independence is $\mathcal { O } ( N T ^ { 2 } )$ . However, results shown in Figure 6 reveal that measured overheads of Timer-XL is much less than N times of Timer. 

Since the previous analysis of model efficiency on time-series Transformer predominantly focuses on self-attention on 1D time series, we initially present a theoretical derivation of the computational complexity of Transformers on 2D time series, including the parameter counts, memory footprint, and FLOPs in Table 8. We find that other parts of Transformers, such as feed-forward network, have a complexity of $\mathcal { O } ( N T )$ no matter which approach is adopted to handle multivariate time series. They also account for dominant overheads in existing benchmarks since the context length is not large enough, confirming our empirical results. Further, we introduce FlashAttention (Dao et al., 2022) to improve the model efficiency, which is computationally equivalent and reduces the overall memory footprint of Timer-XL to (NT) without affecting performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/e9335bb956d2d4fe5d90aff779d23cb24156fb6bbb1530578435f38c72cd5831.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/262b2143968c065c0b37a0fce8ab7866612f9c640d9886b6fd24d0a9ed8a94f0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/7429256c9f641b6d0b89e0b5864bf5692aa6efe04df7b6f57a6a12e4dd55877b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/7dfe037b47fc459659f80d52deb85c12df1040cdc5daf8efb72e01e5256c8570.jpg)



Figure 6: Efficiency analysis. We compare representative time-series Transformers on multivariate datasets with variable numbers ranging from ten to hundred and increase the lookback length.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/c09a13a16b96ad4933ca7c5be49faed5fbd94da71843a9478cda98d0dc751486.jpg)



Figure 7: Visualization of TimeAttention. It is from the first sample of a length 672 in the test split of Traffic. We visualize the last 10 variables with each contains 7 tokens. We present auto-correlation function plot. Auto-correlation can be reflected by the distribution of attention scores (bottom right). We average TimeAttention across sub-blocks, which indicates Pearson correlations (upper right).


Representation Analysis In addition to the enhanced performance, fine-grained token dependencies offer improved interpretability. We present a showcase visualization from Traffic in Figure 7. It is observed that sub-matrices along the diagonal generally receive greater attention, which reasonably reveals predominant dependencies within the endogenous variable. By zooming in a sub-block that corresponds to Variable-3, we observe that the attention distribution of the last row can indicate certain strong dependencies among patch tokens. This observation is also supported by the auto-correlation function plot (ACF), which reveals auto-correlations with certain lags and thus the model pays special attention to these tokens. Furthermore, we average each sub-matrix into one scalar. The outcome matrix can also illustrate Pearson correlations presented in the raw data. 

## 5 CONCLUSION AND FUTURE WORK

In this paper, we emphasize the efficacy of causal Transformers in the forecasting of long-context time series. To facilitate long-context Transformers on diverse tasks, we propose multivariate next token prediction, a novel paradigm to predict multidimensional series with covariates. We present Timer-XL enhanced by TimeAttention as an extra-long version of pre-trained time-series Transformers. It simultaneously captures temporal dynamics and variable correlations by enhanced self-attention. In addition to achieving state-of-the-art performance on extensive benchmarks, we establish challenging benchmarks for long-context forecasting. By pre-training on large-scale heterogeneous time series, Timer-XL demonstrates notable zero-shot performance as a large time-series model. In the future, we will improve computational efficiency and build large domain-specific models with Timer-XL. 

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (U2342217 and 62021002), the BNRist Project, and the National Engineering Research Center for Big Data Software. 

## REFERENCES



Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024. 





Yoshua Bengio, Rejean Ducharme, and Pascal Vincent. A neural probabilistic language model.´ Advances in neural information processing systems, 13, 2000. 





George Box. Box and jenkins: time series analysis, forecasting and control. In A Very British Affair: Six Britons and the Development of Time Series Analysis During the 20th Century, pp. 161–215. Springer, 2013. 





Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, et al. Spectral temporal graph neural network for multivariate time-series forecasting. Advances in neural information processing systems, 33:17766–17778, 2020. 





Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-´ efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 





Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023. 





Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, and Ying Shan. Unireplknet: A universal perception large-kernel convnet for audio video point cloud time-series and image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5513–5524, 2024. 





Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. Moment: A family of open time-series foundation models. arXiv preprint arXiv:2402.03885, 2024. 





Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, Andras Hor ´ anyi, Joaqu ´ ´ın Munoz-Sabater,˜ Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, et al. The era5 global reanalysis. Quarterly Journal ofthe Royal Meteorological Society, 146(730):1999–2049, 2020. 





RJ Hyndman. Forecasting: principles and practice. OTexts, 2018. 





Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2021. 





Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023. 





Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, and Rafał Weron. Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy, 293:116983, 2021. 





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 95–104, 2018. 





Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. Advances in neural information processing systems, 32, 2019. 





Bryan Lim, Sercan O Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for<sup>¨</sup> interpretable multi-horizon time series forecasting. International Journal ofForecasting, 37(4): 1748–1764, 2021. 





Juncheng Liu, Chenghao Liu, Gerald Woo, Yiwei Wang, Bryan Hooi, Caiming Xiong, and Doyen Sahoo. Unitst: Effectively modeling inter-series and intra-series dependencies for multivariate time series forecasting. arXiv preprint arXiv:2406.04975, 2024a. 





Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: Time series modeling and forecasting with sample convolution and interaction. Advances in Neural Information Processing Systems, 35:5816–5828, 2022a. 





Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International conference on learning representations, 2021. 





Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. Advances in Neural Information Processing Systems, 35: 9881–9893, 2022b. 





Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625, 2023. 





Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Autotimes: Autoregressive time series forecasters via large language models. arXiv preprint arXiv:2402.02370, 2024b. 





Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer: Generative pre-trained transformers are large time series models. In Forty-first International Conference on Machine Learning, 2024c. 





Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m4 competition: 100,000 time series and 61 forecasting methods. International Journal ofForecasting, 36(1):54–74, 2020. 





Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730, 2022. 





R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2:13, 2023. 





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437, 2019. 





Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 





PEMS. Traffic Dataset. http://pems.dot.ca.gov/. 





Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551, 2020. 





Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Bilos, Hena Ghonia, Nadhir Vincent Hassen, Anderson Schnei-ˇ der, et al. Lag-llama: Towards foundation models for time series forecasting. arXiv preprint arXiv:2310.08278, 2023. 





David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International journal of forecasting, 36(3): 1181–1191, 2020. 





Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Timemoe: Billion-scale time series foundation models with mixture of experts. arXiv preprint arXiv:2409.16040, 2024. 





Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 





Huihui Sun and Xiaofeng Zhang. Study on coded permutation entropy of finite length gaussian white noise time series. Chinese Journal ofElectronics, 33(1):185–194, 2024. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and ` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 





Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language models. arXiv preprint arXiv:2402.02244, 2024a. 





Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Yong Liu, Yunzhong Qiu, Haoran Zhang, Jianmin Wang, and Mingsheng Long. Timexer: Empowering transformers for time series forecasting with exogenous variables. arXiv preprint arXiv:2402.19072, 2024b. 





Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592, 2024. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021. 





Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186, 2022. 





Haixu Wu, Hang Zhou, Mingsheng Long, and Jianmin Wang. Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 5(6):602–611, 2023. 





Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023. 





Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems, 30, 2017. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023. 





Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The Eleventh International Conference on Learning Representations, 2022. 



Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. 

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, pp. 11106–11115, 2021. 

## A PROOF OF MODEL EFFICIENCY

## A.1 SETUPS

Given an input univariate time series divided into T tokens according to the patch size P, which is fed into the vanilla Transformer. The training objective is to predict the next token of P time points. We will generalize the derivation from 1D sequences to 2D time series based on different approaches to handle multivariate data with the variable number N. We adopt the same denotations as before: Transformer consists of L blocks with model dimension D. The multi-head attention mechanism has H heads, each with a dimension of $d _ { k }$ for query, key, and value, and $\begin{array} { r } { d _ { k } = \frac { D } { H } } \end{array}$ . The intermediate dimension of feed-forward network is set as $D _ { \mathrm { f f } } = \alpha D$ . The results are summarized in Table 8, we provide the detailed proof in the following sections. 


Table 8: Parameters count and computational complexity of Transformers for multivariate time series.


<table><tr><td>Metric</td><td>Type</td><td>Count</td><td>Complexity</td></tr><tr><td rowspan="2">FLOPs(Training Speed)</td><td>Channel Independence</td><td><eq>12(PDNT + L(D + H)NT^{2} + (2 + \alpha)LD^{2}NT)</eq></td><td><eq>\mathcal{O}(LDNT(D + T))</eq></td></tr><tr><td>Channel Dependence</td><td><eq>12(PDNT + L(D + H)N^{2}T^{2} + (2 + \alpha)LD^{2}NT)</eq></td><td><eq>\mathcal{O}(LDNT(D + NT))</eq></td></tr><tr><td rowspan="2">Parameters</td><td>Encoder-Only</td><td><eq>(4 + 2\alpha)LD^{2} + 4LD + (1 + T)PD</eq></td><td><eq>\mathcal{O}(LD^{2})</eq></td></tr><tr><td>Decoder-Only</td><td><eq>(4 + 2\alpha)LD^{2} + 4LD + 2PD</eq></td><td><eq>\mathcal{O}(LD^{2})</eq></td></tr><tr><td rowspan="2">MemoryFootprint</td><td>Self-Attention</td><td><eq>4(D + P)NT + (32 + 8\alpha)LDNT + 4LHN^{2}T^{2}</eq></td><td><eq>\mathcal{O}(LHN^{2}T^{2})</eq></td></tr><tr><td>FlashAttention</td><td><eq>4(D + P)NT + (32 + 8\alpha)LDNT</eq></td><td><eq>\mathcal{O}(LDNT)</eq></td></tr></table>


∗ L is the block number of Transformers. D is the dimension of embeddings (the hidden dimension of FFN D is set as αD). H is the head number and the dimension of query, key, and value $d _ { k } = D / H$ . The overhead is to train on a multivariate time series $( N \cdot$ -variables and TP time points) with patch token length P and context length T. Set $N = 1$ for training on univariate time series. 


## A.2 FLOPS

As a preliminary, the multiplication between matrix $\mathbf { A } \in \mathbb { R } ^ { n \times m }$ and matrix $\mathbf { C } \in \mathbb { R } ^ { m \times p }$ requires mnp multiplications and mnp additions, resulting in 2mnp floating-point operations. Given batched matrices $\overset { \bullet } { \mathbf { A } } \in \mathbb { R } ^ { B \times n \times m }$ and $\mathbf { \bar { C } } \in \mathbb { R } ^ { B \times m \times p }$ , B times matrix multiplications will be performed. It is evident that the batch size is a linear multiplier. Thus, we first omit B to calculate the operations of dealing with one univariate series, and then we will reintroduce it to analyze channel independence. 

The computational cost of Transformers can be primarily categorized into two types: (1) multi-head attention calculation and (2) linear transformations. In contrast, the operations of layer normalization, residual connection, activation functions, and position embedding with the complexity of $\mathcal { O } ( T D )$ are less significant. Therefore, we derive the computational complexity mainly with respect to the above two types by delving into the forwarding process of one univariate series. 

Patch Embedding The tokenized time series $\{ \mathbf { x } _ { i } \} \in \mathbb { R } ^ { T \times P }$ is mapped into the embedding space through the patch-wise embedding $\mathbf { W } _ { e } \in \mathbb { R } ^ { D \times P }$ , resulting in $2 P D \bar { T }$ operations. 

Self-Attention The calculation of self-attention begins with the computation of query, key and value by multiplying the patch embeddings with matrices $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } \stackrel { \bullet } { \in } \mathbb { R } ^ { D \times d _ { k } }$ respectively in H heads, which incurs a computational cost of $6 H D d _ { k } T = 6 D ^ { 2 } T$ and yields $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in \mathbb { R } ^ { H \times T \times d _ { k } }$ Next, the dot product $\mathbf { Q } \mathbf { K } ^ { \bar { \top } } \in \mathbb { R } ^ { H \times T \times T }$ is conducted in each head, leading to $2 H d _ { k } T ^ { 2 } = 2 D T ^ { 2 }$ operations. Following this, the Pre-Softmax map is divided by $\sqrt { d _ { k } }$ and processed through Softmax, which includes exponentiation, summation, and normalization of each element, resulting in $4 H T ^ { 2 }$ operations. The subsequent multiplication with V incurs $2 H d _ { k } T ^ { 2 } = 2 D T ^ { 2 }$ operations. Finally, multiple heads are concatenated and multiplied by $\mathbf { W } _ { o } \in \mathbb { R } ^ { D \times D }$ , contributing $2 \dot { D } ^ { 2 } T$ operations. 

Feed-Forward Network It first projects the token representations into the dimension of $D _ { f f }$ and subsequently projects it back to the dimension D, resulting in a total operations of 4α $D ^ { 2 } T$ 

Patch Projection For encoder-only models, all token representations are flattened and mapped directly to P time points by $\mathbf { W } _ { d } \in \mathbf { \bar { \mathbb { R } } } ^ { T D \times P }$ . In contrast, token-wise projector $\mathbf { W } _ { d } \in \mathbb { R } ^ { D \times \pmb { P } }$ in decoder-only models independently map each token to the predicted next token. In both cases, the number of operations is $2 \bar { P } D T$ , but the token-wise projector will result in a smaller parameter count. 

The forwarding operations in L-layers Transformer is $4 P D T + 4 L ( D + H ) T ^ { 2 } + ( 8 + 4 \alpha ) L D ^ { 2 } T$ in sum. Considering that the majority of operations in Transformers are binary operations $( \mathrm { e . g . }$ , matrix multiplications), the gradients for both matrices are computed separately. As a result, the number of operations in backpropagation is the twice of forwarding. Therefore, the total operations of training a Transformer on a univariate series consisting of T patches, each of length $P ,$ is derived as: 

$$
f (T) = 1 2 P D T + 1 2 L (D + H) T ^ {2} + (2 4 + 1 2 \alpha) L D ^ {2} T.
$$

We plug typical hyperparameters in the current time-series Transformers and forecasting benchmarks: $D \stackrel { - } { = } 5 \bar { 1 } 2 , \bar { H } = 8 , \bar { L } \stackrel { - } { = } 4 , \alpha = 4 , T = 7 .$ , and $P = 9 6$ , we obtain that: 

$$
f (T) = 2 4 9 6 0 T ^ {2} + 7 6 0 8 7 2 9 6 T \propto 3. 2 8 * 1 0 ^ {- 4} T ^ {2} + T.
$$

Due to the prevalence of short contexts in the time series field, where $T \ll D$ leads to a significant coefficient in $\mathcal { O } ( T )$ , we find the primary computational burden of time-series Transformer lies in linear transformations with $\mathcal { O } ( T )$ , rather than in multi-head self-attention with the $\mathcal { O } ( T ^ { 2 } )$ complexity. 

For multivariate series with N variables, FLOPs is influenced by the handling of multivariate data. When adopting channel independence (Timer and PatchTST), N can be regarded as the batch size B: 

$$
N f (T) = 1 2 P D N T + 1 2 L (D + H) N T ^ {2} + (2 4 + 1 2 \alpha) L D ^ {2} N T.\tag{9}
$$

For models that capture fine-grained intra- and inter-series dependencies (Timer-XL and UniTST) in multivariate series, N is reflected as the enlarged number of tokens: 

$$
f (N T) = 1 2 P D N T + 1 2 L (D + H) N ^ {2} T ^ {2} + (2 4 + 1 2 \alpha) L D ^ {2} N T.\tag{10}
$$

Notably, FLOPs is not entirely equivalent to actual runtime. While FlashAttention increases the overall FLOPs due to its recomputation process, it reduces the number of memory reads and writes. Given that on GPUs, computation is significantly faster than memory access, using FlashAttention can actually lead to further improvements in runtime performance. 

## A.3 PARAMETER COUNT

From the above analysis, we observe that the parameter count of Transformers includes the following: 

Patch Embedding $\mathbf { W } _ { e } \in \mathbb { R } ^ { D \times P }$ to obtain patch embeddings. 

Self-Attention $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } \in \mathbb { R } ^ { D \times d _ { k } }$ of H heads and $\mathbf { W } _ { o } \in \mathbb { R } ^ { D \times D }$ for all heads. 

Feed-Forward Network $\mathbf { W } _ { \mathrm { f f n 1 } } , \mathbf { W } _ { \mathrm { f f n 2 } } \in \mathbb { R } ^ { D \times D _ { \mathrm { f f } } }$ in feed-forward network. 

Layer Normalization It contains the weight $\mathbf { W } \in \mathbb { R } ^ { D }$ and the bias $\mathbf { b } \in \mathbb { R } ^ { D }$ . Every Transformer block includes two normalizations after multi-head attention and feed-forward network respectively. 

Patch Projection $\mathbf { W } _ { d } \in \mathbb { R } ^ { T D \times P }$ in flatten head and $\mathbf { W } _ { d } \in \mathbb { R } ^ { D \times P }$ in token-wise projection. 

In sum, the total count of parameters in time-series Transformers can be expressed as: 

$\left\{ \begin{array} { l } { { ( 4 + 2 \alpha ) L D ^ { 2 } + 4 L D + ( 1 + T ) P D } } \\ { { ( 4 + 2 \alpha ) L D ^ { 2 } + 4 L D + 2 P D , } } \end{array} \right.$ , using flatten head, Parameter Count = using token-wise projection. 

(11) 

## A.4 MEMORY FOOTPRINT

The memory footprint during training can be primarily categorized into three parts: activation values stored for backpropagation, model parameters, and optimizer parameters. 

Regardless of other precision types (e.g., FP16), model parameters and gradients are typically stored as 32-bit floating-point numbers, with each parameter occupying 4 bytes of memory. For time-series Transformers, memory footprint of activation values is given as follows: 

Patch Embedding Gradient computation for W<sub>e</sub> preserves its input $\{ \mathbf { x } _ { i } \} \in \mathbb { R } ^ { T \times P }$ of $4 P T$ bytes. 

Self-Attention Gradient calculation for $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } \in \mathbb { R } ^ { D \times d _ { k } }$ requires their inputs $\mathbf { H } \in \mathbb { R } ^ { T \times D }$ amounting to a total of 4DT bytes. The dot product for attention map also needs to store $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in$ $\mathbb { R } ^ { H \times T \times d _ { k } ^ { \smile } }$ , which collectively require a total of 12DT bytes of memory. Gradient computation of $\mathbf { W } _ { o } \in \mathbb { R } ^ { D \times D }$ necessitates the concatenated multi-head attention representations $\mathbf { H } \in \mathbb { R } ^ { \pmb { T } \times \pmb { D } }$ , which occupies 4DT bytes. If memory-efficient attention mechanisms like FlashAttention (Dao et al., 2022) is not applied, the outcome $\mathbf { Q } \dot { \mathbf { K } } ^ { \top }$ will be stored and occupy $4 H T ^ { 2 }$ bytes. Instead, if FlashAttention is adopted, the storage overhead can be avoided. 

Feed-Forward Network ReLU activation function is typically employed in this module. The input $\mathbf { H } \in \mathbb { R } ^ { T \times D }$ must be retained, requiring a total of 4DT bytes. Additionally, the product $\mathbf { W } _ { \mathrm { f f n 1 } } \mathbf { H }$ also needs to be stored, amounting to $\dot { 4 } D _ { \mathrm { f f } } \bar { T }$ bytes. Similarly, the output activations of ReLU, which serve as the input for subsequent linear transformations, necessitate another 4 $D _ { \mathrm { f f } } T$ bytes. 

Layer Normalization Each block of Transformer encompasses two layer normalizations, with each normalization retaining its input, resulting in the memory requirement of 8DT bytes. 

Patch Projection To perform backpropagation for $W _ { d } \in \mathbb { R } ^ { D \times P }$ , it is necessary to retain its input $\mathbf { H } \in \mathbb { R } ^ { T \times \pmb { D } }$ , resulting in a total memory requirement of 4DT bytes. 

The formula for the total activation values of the entire model occupying GPU memory is as follows: 

$$
\text { Memory   Footprint } = \left\{ \begin{array}{l l} 4 (D + P) T + (3 2 + 8 \alpha) L D T + 4 L H T ^ {2}, & \text { w / o   FlashAttention }, \\ 4 (D + P) T + (3 2 + 8 \alpha) L D T, & \text { with   FlashAttention }. \end{array} \right.\tag{12}
$$

The derived occupancy of activation values increases proportionally with the batch size B. For multivariate series, N can be used as a multiplier in channel independence. For channel independence models, we can substitute T with NT as before. The total memory footprint is the sum of activation values and parameters of model and optimizer, which are proportional to the parameter count derived in Equation 11. Due to the limited model size in the time series field, the memory consumption of parameters is minimal and can be considered negligible in practice. Therefore, the overall memory footprint can be predominantly determined by the occupied memory of activation values. 

## B EXPERIMENTAL DETAILS

## B.1 DATASETS

We conduct experiments on well-acknowledged benchmarks to evaluate performance of the proposed Timer-XL, which includes (1) ETT (Zhou et al., 2021) contains 7 factors of electricity transformers from July 2016 to July 2018, which is recorded every hour or 15 minutes. (2) Weather (Wu et al., 2021) includes 21 meteorological factors collected every 10 minutes from the Max Planck Biogeochemistry Institute Weather Station in 2020. (3) ECL (Wu et al., 2021) records the hourly electricity consumption data of 321 clients. (4) Traffic (Wu et al., 2021) collects hourly road occupancy rates measured by 862 sensors on the San Francisco Bay area highways from January 2015 to December 2016. (5) Solar-Energy (Lai et al., 2018) records the solar power production of 137 PV plants in 2006, which are sampled every 10 minutes. (7) PEMS (Liu et al., 2022a) contains records from the public traffic network in California collected in 5-minute time windows. (8) EPF (Lago et al., 2021) includes five subsets that span six years. Each contains the electricity price as the endogenous variable to be predicted and two exogenous variables of the day-ahead electricity markets. (9) GTWSF (Wu et al., 

2023) is a dataset collected from the National Centers for Environmental Information (NCEI). This large-scale collection contains hourly averaged wind speed and temperature data from 3850 stations with different geographical scales and densities each, spanning from 2019 to 2021. (10) UTSD (Liu et al., 2024c) is a multi-domain time series dataset, which includes seven domains with a hierarchy of four volumes. We adopt the largest volume that encompasses 1 billion time points for pre-training. 

We further establish challenging forecasting benchmarks based on the ECMWF Reanalysis v5 (ERA5) dataset (Hersbach et al., 2020) to prevent potential overfitting and performance saturation of deep forecasters in existing benchmarks. Concretely, ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1940 to the present, which provides hourly estimates of a large number of atmospheric, land, and oceanic climate variables, and includes information about uncertainties for all variables at reduced spatial and temporal resolutions. Due to its pattern sufficiency of temporal dynamics and variable correlations, we could establish practical benchmarks to thoroughly evaluate the performance for univariate and multivariate forecasting, as well as adopt it for large-scale pre-training to develop domain-specific large time series models. 

Our datasets are constructed as follows: 

• ERA5-S: To establish a realistic univariate forecasting benchmark, we start from the basic principle of forecastability and make the prediction on sufficient lookback lengths. Instead of the short time span of training in previous benchmarks (generally no more than 2 years), we curated a three-hour frequency dataset spanning 40 years (January 1979 to December 2018) from ERA5, encompassing 116880 time points. In order to prevent overfitting on a single time series, we selected worldwide stations to form seven subsets. 

• ERA5-MS: Each univariate series of ERA5-S provides partial observations governed by the spatio–temporal global weather system. Since discovering the global spatio-temporal correlations presents a fundamental challenge in meteorology, we convert ERA5-S into ERA5-MS by using seven subsets as a challenging multivariate forecasting benchmark. Based on the average results in Tables 2 and 5, we can validate the existence of multi-station correlations among selected stations, which have enhanced the average prediction accuracy. 

• ERA5-Large: To explore the pure data-driven approach to build domain-specific large time series models, we further expanded the number of stations as ERA5-Large, a dataset that evenly covers meteorological 4920 worldwide stations and spans 40 years. We establish the dataset for pre-training, which is expected to generalize across the time (train on the past observations and generalize to the future) and across stations (train on partial stations and generalize to other unseen stations). The total number of time points is around half a billion. 

We follow the same data processing and train-validation-test split protocol used in TimesNet (Wu et al., 2022), where the train, validation, and test datasets are divided according to chronological order to prevent data leakage. Detailed dataset descriptions and prediction settings are provided in Table 9. 

## B.2 BASELINE MODELS

We aim to present Timer-XL as a foundation model for unified time series forecasting. We thoroughly include well-acknowledged and advanced models in each forecasting task. For univariate time series forecasting, we compare Timer-XL with PatchTST (Nie et al., 2022) under channel independence. For multivariate time series prediction, we report official results from Liu et al. (2023; 2024b); Ding et al. (2024), including UniRepLKNet (2024), iTransformer (2023), Corrformer (2023), DLinear (2023), TimesNet (2022), Non-stationary Transformer (2022b), Pyraformer (2021), Autoformer (2021) , StemGNN (2020), DeepAR (2020), and N-BEATS (2019). We further reproduce the performance of related Transformers: Timer (2024c) and UniTST (2024a) based on their official repositories. For covariate-informed time series forecasting, we report the official results of TimeXer (2024b). For zero-shot forecasting, we follow Liu et al. (2024c) that predicts future length-96 windows in well-acknowledged datasets. Totally, more than 20 baselines are included for a complete comparison. 

## B.3 IMPLEMENTATION DETAILS

All the experiments are implemented by PyTorch (Paszke et al., 2019) on NVIDIA A100 Tensor Core GPUs. We employ the Adam optimizer (Kingma & Ba, 2014) and MSE loss for model optimization. 


Table 9: Dataset descriptions. Dim. denotes the number of variables (For univariate forecasting, we adopt channel independence (Nie et al., 2022) or train separate models on each variable). Dataset Length denotes the number of time points in the (train, validation, test) splits.


<table><tr><td>Tasks</td><td>Dataset</td><td>Dim.</td><td>Training Setting</td><td>Dataset Length</td><td>Information (Frequency)</td></tr><tr><td rowspan="5">Univariate Forecasting</td><td>ETTh1</td><td>7</td><td>{24, 96, 168, 672, 2880}→96</td><td>(8545, 2881, 2881)</td><td>Electricity (Hourly)</td></tr><tr><td>ECL</td><td>321</td><td>{24, 96, 168, 672, 2880, 8832}→96</td><td>(18317, 2633, 5261)</td><td>Electricity (Hourly)</td></tr><tr><td>Traffic</td><td>862</td><td>{24, 96, 168, 672, 2880, 8832}→96</td><td>(12185, 1757, 3509)</td><td>Transportation (Hourly)</td></tr><tr><td>PEMS03</td><td>358</td><td>{96, 288, 1152, 2016, 8064}→96</td><td>(15617, 5135, 5135)</td><td>Transportation (5 mins)</td></tr><tr><td>ERA5-S</td><td>7</td><td>3072→96</td><td>(81816, 11688, 23376)</td><td>Climate (3 Hours)</td></tr><tr><td rowspan="8">Multivariate Forecasting</td><td>ETTh1, ETTh2</td><td>7</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Electricity (Hourly)</td></tr><tr><td>ETTm1, ETTm2</td><td>7</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>Electricity (15 mins)</td></tr><tr><td>ECL</td><td>321</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Electricity (Hourly)</td></tr><tr><td>Traffic</td><td>862</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(12185, 1757, 3509)</td><td>Transportation (Hourly)</td></tr><tr><td>Weather</td><td>21</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>Climate (10 mins)</td></tr><tr><td>Solar-Energy</td><td>137</td><td>{96, 672}→{96, 192, 336, 720}</td><td>(36601, 5161, 10417)</td><td>Energy (10 mins)</td></tr><tr><td>ERA5-MS</td><td>7</td><td>3072→96</td><td>(81816, 11688, 23376)</td><td>Climate (3 Hours)</td></tr><tr><td>GTWSF</td><td>3850</td><td>48→24</td><td>(12280, 1755, 3509)</td><td>Wu et al. (2023)</td></tr><tr><td rowspan="5">Forecasting with Covariates</td><td>NP</td><td>1+2</td><td>168→24</td><td>(36500, 5219, 10460)</td><td>Electricity (Hourly)</td></tr><tr><td>PJM</td><td>1+2</td><td>168→24</td><td>(36500, 5219, 10460)</td><td>Electricity (Hourly)</td></tr><tr><td>BE</td><td>1+2</td><td>168→24</td><td>(36500, 5219, 10460)</td><td>Electricity (Hourly)</td></tr><tr><td>FR</td><td>1+2</td><td>168→24</td><td>(36500, 5219, 10460)</td><td>Electricity (Hourly)</td></tr><tr><td>DE</td><td>1+2</td><td>168→24</td><td>(36500, 5219, 10460)</td><td>Electricity (Hourly)</td></tr><tr><td rowspan="3">Pre-training</td><td>ERA5-Large</td><td>4920</td><td>3072→96</td><td>(81816, 11688, 23376)</td><td>Climate (3 Hours)</td></tr><tr><td>UTSD</td><td>-</td><td>2880→96</td><td>(868778970, 96530996, -)</td><td>Liu et al. (2024c)</td></tr><tr><td>LOTSA</td><td>-</td><td>2880→96</td><td>(231082956489, -, -)</td><td>Woo et al. (2024)</td></tr></table>


Table 10: Performance robustness of Timer-XL. The prediction settings and results keep the same with Table 12. The standard deviation is obtained from three random seeds.


<table><tr><td rowspan="2">Dataset Horizon</td><td colspan="2">ECL</td><td colspan="2">ETTh1</td><td colspan="2">Traffic</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td>0.127±0.001</td><td>0.219±0.001</td><td>0.364±0.002</td><td>0.397±0.001</td><td>0.340±0.002</td><td>0.238±0.001</td></tr><tr><td>192</td><td>0.145±0.001</td><td>0.236±0.001</td><td>0.405±0.002</td><td>0.424±0.001</td><td>0.360±0.001</td><td>0.247±0.001</td></tr><tr><td>336</td><td>0.159±0.001</td><td>0.252±0.001</td><td>0.427±0.003</td><td>0.439±0.002</td><td>0.377±0.002</td><td>0.256±0.002</td></tr><tr><td>720</td><td>0.187±0.003</td><td>0.277±0.003</td><td>0.439±0.002</td><td>0.459±0.004</td><td>0.418±0.003</td><td>0.279±0.002</td></tr><tr><td rowspan="2">Dataset Horizon</td><td colspan="2">Solar-Energy</td><td colspan="2">Weather</td><td colspan="2">ERA5-MS</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>96</td><td>0.162±0.003</td><td>0.221±0.002</td><td>0.157±0.002</td><td>0.205±0.001</td><td>0.164±0.001</td><td>0.307±0.000</td></tr><tr><td>192</td><td>0.187±0.003</td><td>0.239±0.002</td><td>0.206±0.003</td><td>0.250±0.002</td><td></td><td></td></tr><tr><td>336</td><td>0.205±0.003</td><td>0.255±0.002</td><td>0.259±0.003</td><td>0.291±0.003</td><td></td><td></td></tr><tr><td>720</td><td>0.238±0.003</td><td>0.279±0.003</td><td>0.337±0.002</td><td>0.344±0.002</td><td></td><td></td></tr></table>

We adopt channel independence from Nie et al. (2022) in univariate time series forecasting. Based on the prevalence of patch-level tokenization in the time series field, we reproduce typical Transformers: PatchTST (2022), Timer (2024c), and UniTST (2024a) based on their official repositories, and keep their model hyperparameters and training configurations the same to evaluate the inherent capability of base models. The results of other baselines are based on the benchmark provided by Liu et al. (2023; 2024b); Ding et al. (2024); Wang et al. (2024b), which is fairly built on the configurations provided by their original paper. Detailed experimental configurations are provided in Table 11. We also report the standard deviations under three runs with different random seeds in Table 10, which exhibits that the performance of Timer-XL is stable. 

For the metrics, we adopt the symmetric mean absolute percentage error (SMAPE), a metric that is independent of the numerical range, to evaluate one-for-all generalization performance on ERA5- Large. For other experiments, we adopt the root mean square error (MSE) and mean absolute error (MAE) that follows previous work. These metrics can be calculated as follows: 

$$
\mathrm{SMAPE} = \frac {2 0 0}{T} \sum_ {i = 1} ^ {T} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} | + | \widehat {\mathbf {X}} _ {i} |}, \mathrm{MSE} = \sum_ {i = 1} ^ {T} | \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} | ^ {2}, \mathrm{MAE} = \sum_ {i = 1} ^ {T} | \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |.
$$

Here $\mathbf { X } \in \mathbb { R } ^ { T }$ is a univariate time series and $\widehat { \mathbf X }$ is the corresponding prediction. For multivariate time bseries, we further calculate the mean metric in the variable dimension. 


Table 11: Experimental configurations of Timer-XL and other baseline Transformers. All the experi ments adopt the ADAM (2014) optimizer with the default hyperparameter $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 \dot { 9 } 9 )$


<table><tr><td rowspan="2">Experiment</td><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="5">Configuration</td><td colspan="4">Training Process</td></tr><tr><td>L</td><td>D</td><td><eq>d_k</eq></td><td>H</td><td>P</td><td>LR</td><td>Loss</td><td>Batch Size</td><td>Epochs</td></tr><tr><td rowspan="5">Univariate Forecasting</td><td></td><td>ECL</td><td>3</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>2048</td><td>10</td></tr><tr><td>Timer-XL</td><td>Traffic</td><td>3</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.001</td><td>MSE</td><td>2048</td><td>10</td></tr><tr><td>PatchTST</td><td>ETTh1</td><td>1</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>256</td><td>10</td></tr><tr><td></td><td>PEMS03</td><td>3</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>2048</td><td>10</td></tr><tr><td></td><td>ERA5-S</td><td>1</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>2048</td><td>10</td></tr><tr><td rowspan="8">Multivariate Forecasting</td><td></td><td>Global Temp.</td><td>3</td><td>1024</td><td>128</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>8</td><td>10</td></tr><tr><td></td><td>Global Wind</td><td>3</td><td>1024</td><td>128</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>8</td><td>10</td></tr><tr><td>Timer-XL</td><td>ECL</td><td>5</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>4</td><td>10</td></tr><tr><td>UniTST</td><td>Traffic</td><td>4</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>4</td><td>10</td></tr><tr><td>Timer</td><td>ETTh1</td><td>1</td><td>1024</td><td>128</td><td>8</td><td>96</td><td>0.0001</td><td>MSE</td><td>32</td><td>10</td></tr><tr><td>PatchTST</td><td>Weather</td><td>4</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0005</td><td>MSE</td><td>32</td><td>10</td></tr><tr><td></td><td>Solar.</td><td>6</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0001</td><td>MSE</td><td>16</td><td>10</td></tr><tr><td></td><td>ERA5-MS</td><td>3</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0001</td><td>MSE</td><td>256</td><td>10</td></tr><tr><td rowspan="5">Forecasting with Covariates</td><td>Timer-XL</td><td>NP</td><td>3</td><td>512</td><td>64</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>4</td><td>10</td></tr><tr><td>TimeXer</td><td>PJM</td><td>2</td><td>512</td><td>64</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>16</td><td>10</td></tr><tr><td>Timer</td><td>BE</td><td>2</td><td>512</td><td>64</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>16</td><td>10</td></tr><tr><td>PatchTST</td><td>FR</td><td>2</td><td>512</td><td>64</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>16</td><td>10</td></tr><tr><td></td><td>DE</td><td>2</td><td>512</td><td>64</td><td>8</td><td>24</td><td>0.0001</td><td>MSE</td><td>16</td><td>10</td></tr><tr><td rowspan="8">Pre-training</td><td>Timer-XL</td><td rowspan="2">ERA5-Large</td><td>4</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0001</td><td>MSE</td><td>40960</td><td>10</td></tr><tr><td>PatchTST</td><td>4</td><td>512</td><td>64</td><td>8</td><td>96</td><td>0.0001</td><td>MSE</td><td>40960</td><td>10</td></tr><tr><td>Timer-XL</td><td>UTSD</td><td>8</td><td>1024</td><td>128</td><td>8</td><td>96</td><td>0.00005</td><td>MSE</td><td>16384</td><td>10</td></tr><tr><td>Timer</td><td>(Liu et al., 2024c)</td><td>8</td><td>1024</td><td>128</td><td>8</td><td>96</td><td>0.00005</td><td>MSE</td><td>16384</td><td>10</td></tr><tr><td>Timer-XL</td><td></td><td>8</td><td>1024</td><td>128</td><td>8</td><td>96</td><td>0.001</td><td>MSE</td><td>32768</td><td>-</td></tr><tr><td>MoiraiSmall</td><td>LOTSA</td><td>6</td><td>384</td><td>64</td><td>6</td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td>MoiraiBase</td><td>(Woo et al., 2024)</td><td>12</td><td>768</td><td>64</td><td>12</td><td>-</td><td></td><td colspan="3">Woo et al. (2024)</td></tr><tr><td>MoiraiLarge</td><td></td><td>24</td><td>1024</td><td>64</td><td>16</td><td>-</td><td></td><td></td><td></td><td></td></tr></table>


∗ L is the layer number of Transformers, D is the dimension of token embedding (the hidden dimension of FFN is set as $4 D ) , d _ { k }$ is the dimension of query, key, and value, H is the multi-head number, P is the patch size, and LR is the initial learning rate. 


## C HYPERPARAMETER SENSITIVITY

We evaluate the hyperparameter sensitivity of Timer-XL on the ERA5-MS benchmark, as illustrated in Figure 8, concerning the following factors: the number of layers $L ,$ the patch size $P ,$ and the lookback length during inference. Our findings indicate that performance of Timer-XL generally improves with increases with $L ,$ suggesting that Timer-XL is a scalable deep forecaster. Furthermore, our analysis of the influence of $\bar { P }$ reveals that the optimal patch size is generally close to the predicted length, since it avoid multi-step error accumulations. Toward better long-term forecasting performance, it leaves a future improvement to adopt different patch sizes of input and output tokens. Finally, we investigate the impact of input length during inference. We discover that the optimal lookback length of during is not necessarily the length during training. Given that decoder-only Transformers can accommodate inference inputs shorter than those used during training, this finding is noteworthy and indicates the potential to improve the performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/5f7267f7834927f6ebc244cbba344b58840251c5279be649905a2e952be76970.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/327d4fcf93118dd76cb13595795a336719ca740b4af29519ba92f5554c573ea5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/51512f041dbfa9525a42353c21fc0ca260d902b11aa5eb63e5c6a93ee2cbbac4.jpg)



Figure 8: Hyperparameter sensitivity of Timer-XL (input-3072-pred-96 on ERA5-MS), including the number of Transformer blocks L, the patch size P, and the input lookback length during inference.


## D SHOWCASES

To facilitate a clear comparison among various models, we present additional prediction visualization from diverse datasets in Figure 9 and 10. Showcases are randomly selected from Timer-XL and the following time-series Transformers: PatchTST (2022), Timer (2024c), and UniTST (2024a). Among them, Timer-XL presents the most accurate predictions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/80e239f34cbc27e792a68e83340e6b26bdac2813d0dd0f369c1a80dc063e76db.jpg)



Figure 9: Visualization results on univariate time series dataset. We adopt the forecasting setting of 2880-pred-96 on ECL, ETTh1 and Traffic, and 2016-pred-96 on PEMS.


## E SUPPLEMENTARY RESULTS

## E.1 FULL RESULT OF MULTIVARIATE FORECASTING

Table 12 provides the complete results of the one-for-all multivariate forecasting benchmark across well-acknowledged datasets. We evaluate Timer-XL and baseline models by rolling forecasting: each model is trained with input length 672 and output length 96, and the predicted values are integrated as part of the input in the next iteration until reaching the desired forecast length in 96, 192, 336, 720 . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/5d1f4e3aeb04a53af44b45bdcb56b4a2d6d88af6969c06909795e1aceb54cebd.jpg)



Figure 10: Visualization results on multivariate time series dataset. We adopt the forecasting setting of 672-pred-96 on ETTh1 (7 Variables) and Traffic (862 Variables).


We highlight that this benchmark evaluates the fundamental model versatility of deep forecasters, which aims to break the awkward situation of extensive training and model storage in pursuit of better practice for real-world forecasting requirements. On this benchmark, time-series Transformers significantly stand out from other baseline models, and our proposed Timer-XL can achieve state-ofthe-art performance, making it a nice fundamental backbone of a one-for-all forecaster. 

## E.2 FULL RESULT OF ZERO-SHOT FORECASTING

Table 13 provides the full results of zero-shot forecasting on the benchmark from Wu et al. (2022). We build Timer-XL based on the configuration in Table 11, which is pre-trained on the aggregated datasets of UTSD (Liu et al., 2024c) and LOTSA (Woo et al., 2024). The patch size of Timer-XL is set as 96 and we conduct rolling forecast to obtain the desired forecast length in 96, 192, 336, 720 . 

We evaluate most advanced large models based on their official model checkpoints, including Time-MoE (Shi et al., 2024), Moirai (Woo et al., 2024), TimesFM (Das et al., 2023), MOMENT Goswami et al. (2024), and Chronos (Ansari et al., 2024). We conduct zero-shot evaluations on datasets that are not included during the pre-training of corresponding models. For each of the evaluated model, we use their maximum input length during inference. The metric (MSE/MAE) is averaged from all predicted windows in the test split. 

## E.3 ABLATION STUDY OF TIMEATTENTION

We conduct evaluations on TimeAttention to validate the effectiveness of position embeddings. As for variable embedding, the distinction between endogenous and exogenous variables can improve performance. Based on our observation of the learned $u > v ,$ we find that the token reasonably pays more attention to tokens of the endogenous variable. It leaves a prior to mask out minor dependencies that focuses less on exogenous variables. For the temporal dimension, other position embeddings are inferior to RoPE, since it uses the affine transformation, while others are additive, and thereby less confused with the same additive embedding for variables. 

## E.4 SUPPLEMENTARY RESULTS OF LONG-CONTEXT FORECASTING

Long context is a basic indicator of foundation models, which can support emergence capabilities such as prompting, in-context learning, retrieval-augmented generation, etc. However, the long-context forecasting paradigm receives less attention in the current community, which can be due to the lack of benchmarks. In the meteorological ERA5, it is necessary to support the context of more than years to contain a specific cycle (such as El Nino). In Table 15, the performance of Timer-XL and DLinear generally improves with the increased context length. By contrast, it reveals the performance degradation of PatchTST. Similar to the observations in Figure 3, the encoder-only architecture produces inferior predictions after thousands of time points, which can be concealed due to the short context adopted in previous benchmarks. Although PatchTST has conducted an initial exploration in the context of hundreds of time points, it inappropriately works in ever-long contexts. Therefore, we believe that context bottlenecks deserve further exploration in this community. 


Table 12: Full multivariate forecasting results: we conduct rolling forecast with a single model trained on each dataset (lookback length is 672) and accomplish four forecast lengths in 96, 192, 336, 720 .


<table><tr><td colspan="2">Models</td><td colspan="2">Timer-XL(Ours)</td><td colspan="2">Timer(2024c)</td><td colspan="2">UniTST(2024a)</td><td colspan="2">iTransformer(2023)</td><td colspan="2">DLinear(2023)</td><td colspan="2">PatchTST(2022)</td><td colspan="2">TimesNet(2022)</td><td colspan="2">Stationary(2022b)</td><td colspan="2">Autoformer(2021)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.364</td><td>0.397</td><td>0.371</td><td>0.404</td><td>0.379</td><td>0.415</td><td>0.387</td><td>0.418</td><td>0.369</td><td>0.400</td><td>0.373</td><td>0.403</td><td>0.452</td><td>0.463</td><td>0.452</td><td>0.478</td><td>0.467</td><td>0.499</td></tr><tr><td>192</td><td>0.405</td><td>0.424</td><td>0.407</td><td>0.429</td><td>0.415</td><td>0.438</td><td>0.416</td><td>0.437</td><td>0.405</td><td>0.422</td><td>0.405</td><td>0.425</td><td>0.474</td><td>0.477</td><td>0.484</td><td>0.510</td><td>0.492</td><td>0.523</td></tr><tr><td>336</td><td>0.427</td><td>0.439</td><td>0.434</td><td>0.445</td><td>0.440</td><td>0.454</td><td>0.434</td><td>0.450</td><td>0.435</td><td>0.445</td><td>0.423</td><td>0.440</td><td>0.493</td><td>0.489</td><td>0.511</td><td>0.522</td><td>0.519</td><td>0.531</td></tr><tr><td>720</td><td>0.439</td><td>0.459</td><td>0.461</td><td>0.466</td><td>0.482</td><td>0.482</td><td>0.447</td><td>0.473</td><td>0.493</td><td>0.508</td><td>0.445</td><td>0.471</td><td>0.560</td><td>0.534</td><td>0.571</td><td>0.543</td><td>0.589</td><td>0.560</td></tr><tr><td>Avg</td><td>0.409</td><td>0.430</td><td>0.418</td><td>0.436</td><td>0.429</td><td>0.447</td><td>0.421</td><td>0.445</td><td>0.426</td><td>0.444</td><td>0.412</td><td>0.435</td><td>0.495</td><td>0.491</td><td>0.505</td><td>0.513</td><td>0.517</td><td>0.528</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.277</td><td>0.343</td><td>0.285</td><td>0.344</td><td>0.343</td><td>0.398</td><td>0.304</td><td>0.362</td><td>0.305</td><td>0.371</td><td>0.289</td><td>0.347</td><td>0.340</td><td>0.374</td><td>0.348</td><td>0.403</td><td>0.358</td><td>0.397</td></tr><tr><td>192</td><td>0.348</td><td>0.391</td><td>0.365</td><td>0.400</td><td>0.376</td><td>0.420</td><td>0.372</td><td>0.407</td><td>0.412</td><td>0.439</td><td>0.360</td><td>0.393</td><td>0.402</td><td>0.414</td><td>0.408</td><td>0.448</td><td>0.435</td><td>0.451</td></tr><tr><td>336</td><td>0.375</td><td>0.418</td><td>0.412</td><td>0.440</td><td>0.399</td><td>0.435</td><td>0.418</td><td>0.440</td><td>0.527</td><td>0.508</td><td>0.389</td><td>0.420</td><td>0.452</td><td>0.452</td><td>0.424</td><td>0.457</td><td>0.454</td><td>0.475</td></tr><tr><td>720</td><td>0.409</td><td>0.458</td><td>0.468</td><td>0.487</td><td>0.419</td><td>0.457</td><td>0.463</td><td>0.476</td><td>0.830</td><td>0.653</td><td>0.398</td><td>0.440</td><td>0.462</td><td>0.468</td><td>0.448</td><td>0.476</td><td>0.479</td><td>0.492</td></tr><tr><td>Avg</td><td>0.352</td><td>0.402</td><td>0.382</td><td>0.418</td><td>0.384</td><td>0.428</td><td>0.389</td><td>0.421</td><td>0.518</td><td>0.493</td><td>0.359</td><td>0.400</td><td>0.414</td><td>0.427</td><td>0.407</td><td>0.446</td><td>0.431</td><td>0.454</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.290</td><td>0.341</td><td>0.281</td><td>0.338</td><td>0.289</td><td>0.348</td><td>0.311</td><td>0.365</td><td>0.307</td><td>0.350</td><td>0.285</td><td>0.346</td><td>0.338</td><td>0.375</td><td>0.414</td><td>0.414</td><td>0.466</td><td>0.466</td></tr><tr><td>192</td><td>0.337</td><td>0.369</td><td>0.330</td><td>0.368</td><td>0.332</td><td>0.375</td><td>0.353</td><td>0.390</td><td>0.337</td><td>0.368</td><td>0.329</td><td>0.372</td><td>0.371</td><td>0.387</td><td>0.524</td><td>0.482</td><td>0.504</td><td>0.496</td></tr><tr><td>336</td><td>0.374</td><td>0.392</td><td>0.367</td><td>0.393</td><td>0.365</td><td>0.397</td><td>0.387</td><td>0.411</td><td>0.366</td><td>0.387</td><td>0.363</td><td>0.394</td><td>0.410</td><td>0.411</td><td>0.541</td><td>0.497</td><td>0.574</td><td>0.530</td></tr><tr><td>720</td><td>0.437</td><td>0.428</td><td>0.432</td><td>0.433</td><td>0.421</td><td>0.431</td><td>0.452</td><td>0.445</td><td>0.419</td><td>0.419</td><td>0.421</td><td>0.426</td><td>0.478</td><td>0.450</td><td>0.578</td><td>0.509</td><td>0.596</td><td>0.558</td></tr><tr><td>Avg</td><td>0.359</td><td>0.382</td><td>0.352</td><td>0.383</td><td>0.352</td><td>0.388</td><td>0.376</td><td>0.403</td><td>0.357</td><td>0.381</td><td>0.349</td><td>0.385</td><td>0.399</td><td>0.406</td><td>0.514</td><td>0.475</td><td>0.535</td><td>0.512</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.175</td><td>0.257</td><td>0.175</td><td>0.257</td><td>0.171</td><td>0.260</td><td>0.183</td><td>0.272</td><td>0.167</td><td>0.263</td><td>0.172</td><td>0.259</td><td>0.187</td><td>0.267</td><td>0.237</td><td>0.306</td><td>0.255</td><td>0.339</td></tr><tr><td>192</td><td>0.242</td><td>0.301</td><td>0.239</td><td>0.301</td><td>0.228</td><td>0.230</td><td>0.250</td><td>0.315</td><td>0.230</td><td>0.311</td><td>0.233</td><td>0.299</td><td>0.249</td><td>0.309</td><td>0.330</td><td>0.387</td><td>0.279</td><td>0.335</td></tr><tr><td>336</td><td>0.293</td><td>0.337</td><td>0.293</td><td>0.342</td><td>0.282</td><td>0.336</td><td>0.311</td><td>0.356</td><td>0.298</td><td>0.361</td><td>0.280</td><td>0.331</td><td>0.321</td><td>0.351</td><td>0.404</td><td>0.424</td><td>0.331</td><td>0.374</td></tr><tr><td>720</td><td>0.376</td><td>0.390</td><td>0.392</td><td>0.407</td><td>0.380</td><td>0.398</td><td>0.417</td><td>0.419</td><td>0.432</td><td>0.446</td><td>0.357</td><td>0.382</td><td>0.497</td><td>0.403</td><td>0.525</td><td>0.486</td><td>0.413</td><td>0.450</td></tr><tr><td>Avg</td><td>0.271</td><td>0.322</td><td>0.275</td><td>0.327</td><td>0.265</td><td>0.306</td><td>0.290</td><td>0.340</td><td>0.282</td><td>0.345</td><td>0.261</td><td>0.318</td><td>0.314</td><td>0.333</td><td>0.374</td><td>0.401</td><td>0.320</td><td>0.374</td></tr><tr><td rowspan="5">ECL</td><td>96</td><td>0.127</td><td>0.219</td><td>0.129</td><td>0.221</td><td>0.130</td><td>0.225</td><td>0.133</td><td>0.229</td><td>0.138</td><td>0.238</td><td>0.132</td><td>0.232</td><td>0.184</td><td>0.288</td><td>0.185</td><td>0.287</td><td>0.256</td><td>0.357</td></tr><tr><td>192</td><td>0.145</td><td>0.236</td><td>0.148</td><td>0.239</td><td>0.150</td><td>0.244</td><td>0.158</td><td>0.258</td><td>0.152</td><td>0.251</td><td>0.151</td><td>0.250</td><td>0.192</td><td>0.295</td><td>0.282</td><td>0.368</td><td>0.291</td><td>0.376</td></tr><tr><td>336</td><td>0.159</td><td>0.252</td><td>0.164</td><td>0.256</td><td>0.166</td><td>0.262</td><td>0.168</td><td>0.262</td><td>0.167</td><td>0.268</td><td>0.171</td><td>0.272</td><td>0.200</td><td>0.303</td><td>0.289</td><td>0.377</td><td>0.290</td><td>0.379</td></tr><tr><td>720</td><td>0.187</td><td>0.277</td><td>0.201</td><td>0.289</td><td>0.206</td><td>0.297</td><td>0.205</td><td>0.294</td><td>0.203</td><td>0.302</td><td>0.222</td><td>0.318</td><td>0.228</td><td>0.325</td><td>0.305</td><td>0.399</td><td>0.320</td><td>0.403</td></tr><tr><td>Avg</td><td>0.155</td><td>0.246</td><td>0.161</td><td>0.251</td><td>0.163</td><td>0.257</td><td>0.164</td><td>0.258</td><td>0.165</td><td>0.265</td><td>0.169</td><td>0.268</td><td>0.201</td><td>0.303</td><td>0.265</td><td>0.358</td><td>0.289</td><td>0.379</td></tr><tr><td rowspan="5">Traffic</td><td>96</td><td>0.340</td><td>0.238</td><td>0.348</td><td>0.240</td><td>0.359</td><td>0.250</td><td>0.353</td><td>0.259</td><td>0.399</td><td>0.285</td><td>0.359</td><td>0.255</td><td>0.593</td><td>0.315</td><td>0.610</td><td>0.322</td><td>0.675</td><td>0.412</td></tr><tr><td>192</td><td>0.360</td><td>0.247</td><td>0.369</td><td>0.250</td><td>0.373</td><td>0.257</td><td>0.373</td><td>0.267</td><td>0.409</td><td>0.290</td><td>0.377</td><td>0.265</td><td>0.596</td><td>0.317</td><td>0.626</td><td>0.346</td><td>0.679</td><td>0.423</td></tr><tr><td>336</td><td>0.377</td><td>0.256</td><td>0.388</td><td>0.260</td><td>0.386</td><td>0.265</td><td>0.386</td><td>0.275</td><td>0.422</td><td>0.297</td><td>0.393</td><td>0.276</td><td>0.600</td><td>0.319</td><td>0.633</td><td>0.352</td><td>0.688</td><td>0.440</td></tr><tr><td>720</td><td>0.418</td><td>0.279</td><td>0.431</td><td>0.285</td><td>0.421</td><td>0.286</td><td>0.425</td><td>0.296</td><td>0.461</td><td>0.319</td><td>0.436</td><td>0.305</td><td>0.619</td><td>0.335</td><td>0.651</td><td>0.366</td><td>0.693</td><td>0.457</td></tr><tr><td>Avg</td><td>0.374</td><td>0.255</td><td>0.384</td><td>0.259</td><td>0.385</td><td>0.265</td><td>0.384</td><td>0.274</td><td>0.423</td><td>0.298</td><td>0.391</td><td>0.275</td><td>0.602</td><td>0.322</td><td>0.630</td><td>0.347</td><td>0.684</td><td>0.433</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.157</td><td>0.205</td><td>0.151</td><td>0.202</td><td>0.152</td><td>0.206</td><td>0.174</td><td>0.225</td><td>0.169</td><td>0.229</td><td>0.149</td><td>0.202</td><td>0.169</td><td>0.228</td><td>0.185</td><td>0.241</td><td>0.355</td><td>0.409</td></tr><tr><td>192</td><td>0.206</td><td>0.250</td><td>0.196</td><td>0.245</td><td>0.198</td><td>0.249</td><td>0.227</td><td>0.268</td><td>0.211</td><td>0.268</td><td>0.194</td><td>0.245</td><td>0.222</td><td>0.269</td><td>0.286</td><td>0.325</td><td>0.421</td><td>0.450</td></tr><tr><td>336</td><td>0.259</td><td>0.291</td><td>0.249</td><td>0.288</td><td>0.251</td><td>0.291</td><td>0.290</td><td>0.309</td><td>0.258</td><td>0.306</td><td>0.244</td><td>0.285</td><td>0.290</td><td>0.310</td><td>0.323</td><td>0.347</td><td>0.452</td><td>0.465</td></tr><tr><td>720</td><td>0.337</td><td>0.344</td><td>0.330</td><td>0.344</td><td>0.322</td><td>0.340</td><td>0.374</td><td>0.360</td><td>0.320</td><td>0.362</td><td>0.317</td><td>0.338</td><td>0.376</td><td>0.364</td><td>0.436</td><td>0.401</td><td>0.513</td><td>0.496</td></tr><tr><td>Avg</td><td>0.240</td><td>0.273</td><td>0.232</td><td>0.270</td><td>0.231</td><td>0.272</td><td>0.266</td><td>0.291</td><td>0.239</td><td>0.291</td><td>0.226</td><td>0.268</td><td>0.264</td><td>0.293</td><td>0.308</td><td>0.329</td><td>0.435</td><td>0.455</td></tr><tr><td rowspan="5">Solar-Energy</td><td>96</td><td>0.162</td><td>0.221</td><td>0.212</td><td>0.230</td><td>0.190</td><td>0.240</td><td>0.183</td><td>0.265</td><td>0.193</td><td>0.258</td><td>0.168</td><td>0.237</td><td>0.180</td><td>0.272</td><td>0.199</td><td>0.290</td><td>0.206</td><td>0.296</td></tr><tr><td>192</td><td>0.187</td><td>0.239</td><td>0.232</td><td>0.246</td><td>0.223</td><td>0.264</td><td>0.205</td><td>0.283</td><td>0.214</td><td>0.274</td><td>0.189</td><td>0.257</td><td>0.199</td><td>0.286</td><td>0.243</td><td>0.307</td><td>0.254</td><td>0.328</td></tr><tr><td>336</td><td>0.205</td><td>0.255</td><td>0.237</td><td>0.253</td><td>0.250</td><td>0.283</td><td>0.224</td><td>0.299</td><td>0.233</td><td>0.291</td><td>0.212</td><td>0.277</td><td>0.220</td><td>0.301</td><td>0.264</td><td>0.322</td><td>0.272</td><td>0.330</td></tr><tr><td>720</td><td>0.238</td><td>0.279</td><td>0.252</td><td>0.266</td><td>0.292</td><td>0.311</td><td>0.239</td><td>0.316</td><td>0.246</td><td>0.307</td><td>0.240</td><td>0.305</td><td>0.251</td><td>0.321</td><td>0.310</td><td>0.339</td><td>0.326</td><td>0.347</td></tr><tr><td>Avg</td><td>0.198</td><td>0.249</td><td>0.233</td><td>0.249</td><td>0.241</td><td>0.275</td><td>0.213</td><td>0.291</td><td>0.222</td><td>0.283</td><td>0.202</td><td>0.269</td><td>0.213</td><td>0.295</td><td>0.254</td><td>0.315</td><td>0.265</td><td>0.325</td></tr></table>


Table 13: Full results of zero-shot forecasting. A lower MSE or MAE indicates a better prediction. 1<sup>st</sup> Count represents the number of wins achieved by a model under all prediction lengths and datasets.


<table><tr><td colspan="2">Models</td><td colspan="2">Timer-XLBase(Ours)</td><td colspan="2">Time-MoEBase(2024)</td><td colspan="2">Time-MoELarge(2024)</td><td colspan="2">Time-MoEUltra(2024)</td><td colspan="2">MoiraiSmall(2024)</td><td colspan="2">MoiraiBase(2024)</td><td colspan="2">MoiraiLarge(2024)</td><td colspan="2">TimesFM(2023)</td><td colspan="2">MOMENT(2024)</td><td colspan="2">ChronosBase(2024)</td><td colspan="2">ChronosLarge(2024)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.317</td><td>0.356</td><td>0.338</td><td>0.368</td><td>0.309</td><td>0.357</td><td>0.281</td><td>0.341</td><td>0.418</td><td>0.392</td><td>0.363</td><td>0.356</td><td>0.380</td><td>0.361</td><td>0.361</td><td>0.370</td><td>0.654</td><td>0.527</td><td>0.454</td><td>0.408</td><td>0.457</td><td>0.403</td></tr><tr><td>192</td><td>0.358</td><td>0.381</td><td>0.353</td><td>0.388</td><td>0.346</td><td>0.381</td><td>0.305</td><td>0.358</td><td>0.431</td><td>0.405</td><td>0.388</td><td>0.375</td><td>0.412</td><td>0.383</td><td>0.414</td><td>0.405</td><td>0.662</td><td>0.532</td><td>0.567</td><td>0.477</td><td>0.530</td><td>0.450</td></tr><tr><td>336</td><td>0.386</td><td>0.401</td><td>0.381</td><td>0.413</td><td>0.373</td><td>0.408</td><td>0.369</td><td>0.395</td><td>0.433</td><td>0.412</td><td>0.416</td><td>0.392</td><td>0.436</td><td>0.400</td><td>0.445</td><td>0.429</td><td>0.672</td><td>0.537</td><td>0.662</td><td>0.525</td><td>0.577</td><td>0.481</td></tr><tr><td>720</td><td>0.430</td><td>0.431</td><td>0.504</td><td>0.493</td><td>0.475</td><td>0.477</td><td>0.469</td><td>0.472</td><td>0.462</td><td>0.432</td><td>0.460</td><td>0.418</td><td>0.462</td><td>0.420</td><td>0.512</td><td>0.471</td><td>0.692</td><td>0.551</td><td>0.900</td><td>0.591</td><td>0.660</td><td>0.526</td></tr><tr><td></td><td>Avg</td><td>0.373</td><td>0.392</td><td>0.394</td><td>0.415</td><td>0.376</td><td>0.405</td><td>0.356</td><td>0.391</td><td>0.436</td><td>0.410</td><td>0.406</td><td>0.385</td><td>0.422</td><td>0.391</td><td>0.433</td><td>0.418</td><td>0.670</td><td>0.536</td><td>0.645</td><td>0.500</td><td>0.555</td><td>0.465</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.189</td><td>0.277</td><td>0.201</td><td>0.291</td><td>0.197</td><td>0.286</td><td>0.198</td><td>0.288</td><td>0.214</td><td>0.288</td><td>0.205</td><td>0.273</td><td>0.211</td><td>0.274</td><td>0.202</td><td>0.270</td><td>0.260</td><td>0.335</td><td>0.199</td><td>0.274</td><td>0.197</td><td>0.271</td></tr><tr><td>192</td><td>0.241</td><td>0.315</td><td>0.258</td><td>0.334</td><td>0.250</td><td>0.322</td><td>0.235</td><td>0.312</td><td>0.284</td><td>0.332</td><td>0.275</td><td>0.316</td><td>0.281</td><td>0.318</td><td>0.289</td><td>0.321</td><td>0.289</td><td>0.350</td><td>0.261</td><td>0.322</td><td>0.254</td><td>0.314</td></tr><tr><td>336</td><td>0.286</td><td>0.348</td><td>0.324</td><td>0.373</td><td>0.337</td><td>0.375</td><td>0.293</td><td>0.348</td><td>0.331</td><td>0.362</td><td>0.329</td><td>0.350</td><td>0.341</td><td>0.355</td><td>0.360</td><td>0.366</td><td>0.324</td><td>0.369</td><td>0.326</td><td>0.366</td><td>0.313</td><td>0.353</td></tr><tr><td>720</td><td>0.375</td><td>0.402</td><td>0.488</td><td>0.464</td><td>0.480</td><td>0.461</td><td>0.427</td><td>0.428</td><td>0.402</td><td>0.408</td><td>0.437</td><td>0.411</td><td>0.485</td><td>0.428</td><td>0.462</td><td>0.430</td><td>0.394</td><td>0.409</td><td>0.455</td><td>0.439</td><td>0.416</td><td>0.415</td></tr><tr><td></td><td>Avg</td><td>0.273</td><td>0.336</td><td>0.317</td><td>0.365</td><td>0.316</td><td>0.361</td><td>0.288</td><td>0.344</td><td>0.307</td><td>0.347</td><td>0.311</td><td>0.337</td><td>0.329</td><td>0.343</td><td>0.328</td><td>0.346</td><td>0.316</td><td>0.365</td><td>0.310</td><td>0.350</td><td>0.295</td><td>0.338</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.369</td><td>0.391</td><td>0.357</td><td>0.381</td><td>0.350</td><td>0.382</td><td>0.349</td><td>0.379</td><td>0.401</td><td>0.402</td><td>0.376</td><td>0.392</td><td>0.381</td><td>0.388</td><td>0.414</td><td>0.404</td><td>0.688</td><td>0.557</td><td>0.440</td><td>0.393</td><td>0.441</td><td>0.390</td></tr><tr><td>192</td><td>0.405</td><td>0.413</td><td>0.384</td><td>0.404</td><td>0.388</td><td>0.412</td><td>0.395</td><td>0.413</td><td>0.435</td><td>0.421</td><td>0.412</td><td>0.413</td><td>0.434</td><td>0.415</td><td>0.465</td><td>0.434</td><td>0.688</td><td>0.560</td><td>0.492</td><td>0.426</td><td>0.502</td><td>0.524</td></tr><tr><td>336</td><td>0.418</td><td>0.423</td><td>0.411</td><td>0.434</td><td>0.411</td><td>0.430</td><td>0.447</td><td>0.453</td><td>0.438</td><td>0.434</td><td>0.433</td><td>0.428</td><td>0.485</td><td>0.445</td><td>0.503</td><td>0.456</td><td>0.675</td><td>0.563</td><td>0.550</td><td>0.462</td><td>0.576</td><td>0.467</td></tr><tr><td>720</td><td>0.423</td><td>0.441</td><td>0.449</td><td>0.477</td><td>0.427</td><td>0.455</td><td>0.457</td><td>0.462</td><td>0.439</td><td>0.454</td><td>0.447</td><td>0.444</td><td>0.611</td><td>0.510</td><td>0.511</td><td>0.481</td><td>0.683</td><td>0.585</td><td>0.882</td><td>0.591</td><td>0.835</td><td>0.583</td></tr><tr><td></td><td>Avg</td><td>0.404</td><td>0.417</td><td>0.400</td><td>0.424</td><td>0.394</td><td>0.419</td><td>0.412</td><td>0.426</td><td>0.428</td><td>0.427</td><td>0.417</td><td>0.419</td><td>0.480</td><td>0.439</td><td>0.473</td><td>0.443</td><td>0.683</td><td>0.566</td><td>0.591</td><td>0.468</td><td>0.588</td><td>0.466</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.283</td><td>0.342</td><td>0.305</td><td>0.359</td><td>0.302</td><td>0.354</td><td>0.292</td><td>0.352</td><td>0.297</td><td>0.336</td><td>0.294</td><td>0.330</td><td>0.296</td><td>0.330</td><td>0.315</td><td>0.349</td><td>0.342</td><td>0.396</td><td>0.308</td><td>0.343</td><td>0.320</td><td>0.345</td></tr><tr><td>192</td><td>0.340</td><td>0.379</td><td>0.351</td><td>0.386</td><td>0.364</td><td>0.385</td><td>0.347</td><td>0.379</td><td>0.368</td><td>0.381</td><td>0.365</td><td>0.375</td><td>0.361</td><td>0.371</td><td>0.388</td><td>0.395</td><td>0.354</td><td>0.402</td><td>0.384</td><td>0.392</td><td>0.406</td><td>0.399</td></tr><tr><td>336</td><td>0.366</td><td>0.400</td><td>0.391</td><td>0.418</td><td>0.417</td><td>0.425</td><td>0.406</td><td>0.419</td><td>0.370</td><td>0.393</td><td>0.376</td><td>0.390</td><td>0.390</td><td>0.390</td><td>0.422</td><td>0.427</td><td>0.356</td><td>0.407</td><td>0.429</td><td>0.430</td><td>0.492</td><td>0.453</td></tr><tr><td>720</td><td>0.397</td><td>0.431</td><td>0.419</td><td>0.454</td><td>0.537</td><td>0.496</td><td>0.439</td><td>0.447</td><td>0.411</td><td>0.426</td><td>0.416</td><td>0.433</td><td>0.423</td><td>0.418</td><td>0.443</td><td>0.454</td><td>0.395</td><td>0.434</td><td>0.501</td><td>0.477</td><td>0.603</td><td>0.511</td></tr><tr><td></td><td>Avg</td><td>0.347</td><td>0.388</td><td>0.366</td><td>0.404</td><td>0.405</td><td>0.415</td><td>0.371</td><td>0.399</td><td>0.361</td><td>0.384</td><td>0.362</td><td>0.382</td><td>0.367</td><td>0.377</td><td>0.392</td><td>0.406</td><td>0.361</td><td>0.409</td><td>0.405</td><td>0.410</td><td>0.455</td><td>0.427</td></tr><tr><td rowspan="4">ECL</td><td>96</td><td>0.141</td><td>0.237</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.189</td><td>0.280</td><td>0.160</td><td>0.250</td><td>0.153</td><td>0.241</td><td>-</td><td>-</td><td>0.745</td><td>0.680</td><td>0.154</td><td>0.231</td><td>0.152</td><td>0.229</td></tr><tr><td>192</td><td>0.159</td><td>0.254</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.205</td><td>0.292</td><td>0.175</td><td>0.263</td><td>0.169</td><td>0.255</td><td>-</td><td>-</td><td>0.755</td><td>0.683</td><td>0.179</td><td>0.254</td><td>0.172</td><td>0.250</td></tr><tr><td>336</td><td>0.177</td><td>0.272</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.221</td><td>0.307</td><td>0.187</td><td>0.277</td><td>0.187</td><td>0.273</td><td>-</td><td>-</td><td>0.766</td><td>0.687</td><td>0.214</td><td>0.284</td><td>0.203</td><td>0.276</td></tr><tr><td>720</td><td>0.219</td><td>0.308</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.258</td><td>0.335</td><td>0.228</td><td>0.309</td><td>0.237</td><td>0.313</td><td>-</td><td>-</td><td>0.794</td><td>0.696</td><td>0.311</td><td>0.346</td><td>0.289</td><td>0.337</td></tr><tr><td></td><td>Avg</td><td>0.174</td><td>0.278</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.218</td><td>0.303</td><td>0.187</td><td>0.274</td><td>0.186</td><td>0.270</td><td>-</td><td>-</td><td>0.765</td><td>0.686</td><td>0.214</td><td>0.278</td><td>0.204</td><td>0.273</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.171</td><td>0.225</td><td>0.160</td><td>0.214</td><td>0.159</td><td>0.213</td><td>0.157</td><td>0.211</td><td>0.198</td><td>0.222</td><td>0.220</td><td>0.217</td><td>0.199</td><td>0.211</td><td>-</td><td>-</td><td>0.243</td><td>0.255</td><td>0.203</td><td>0.238</td><td>0.194</td><td>0.235</td></tr><tr><td>192</td><td>0.221</td><td>0.271</td><td>0.210</td><td>0.260</td><td>0.215</td><td>0.266</td><td>0.208</td><td>0.256</td><td>0.247</td><td>0.265</td><td>0.271</td><td>0.259</td><td>0.246</td><td>0.251</td><td>-</td><td>-</td><td>0.278</td><td>0.329</td><td>0.256</td><td>0.290</td><td>0.249</td><td>0.285</td></tr><tr><td>336</td><td>0.274</td><td>0.311</td><td>0.274</td><td>0.309</td><td>0.291</td><td>0.322</td><td>0.255</td><td>0.290</td><td>0.283</td><td>0.303</td><td>0.286</td><td>0.297</td><td>0.274</td><td>0.291</td><td>-</td><td>-</td><td>0.306</td><td>0.346</td><td>0.314</td><td>0.336</td><td>0.302</td><td>0.327</td></tr><tr><td>720</td><td>0.356</td><td>0.370</td><td>0.418</td><td>0.405</td><td>0.415</td><td>0.400</td><td>0.405</td><td>0.397</td><td>0.373</td><td>0.354</td><td>0.373</td><td>0.354</td><td>0.337</td><td>0.340</td><td>-</td><td>-</td><td>0.350</td><td>0.374</td><td>0.397</td><td>0.396</td><td>0.372</td><td>0.378</td></tr><tr><td></td><td>Avg</td><td>0.256</td><td>0.294</td><td>0.265</td><td>0.297</td><td>0.270</td><td>0.300</td><td>0.256</td><td>0.288</td><td>0.275</td><td>0.286</td><td>0.287</td><td>0.281</td><td>0.264</td><td>0.273</td><td>-</td><td>-</td><td>0.294</td><td>0.326</td><td>0.292</td><td>0.315</td><td>0.279</td><td>0.306</td></tr><tr><td colspan="2"><eq>1^{st}Count</eq></td><td>15</td><td>10</td><td>2</td><td>1</td><td>3</td><td>0</td><td>10</td><td>7</td><td>0</td><td>0</td><td>0</td><td>5</td><td>1</td><td>10</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>2</td><td></td></tr></table>


∗ Dataset for pre-training is not evaluated on corresponding models, which is denoted by a dash (−). 



∗ Traffic from (PEMS) is generally used during the pre-training of large models and thus not evaluated here. 



∗ Our model checkpoint is available at https://huggingface.co/thuml/timer-base-84m. 



Table 14: Embedding ablation in TimeAttention. For the temporal dimension, we compare prevalent relative and absolute position embeddings. As for the variable dimension, we explore the effectiveness of the variable embedding that distinguishes endogenous and exogenous variables.


<table><tr><td rowspan="2">Design</td><td rowspan="2">Temporal</td><td rowspan="2">Variable</td><td colspan="2">Traffic</td><td colspan="2">Weather</td><td colspan="2">Solar-Energy</td><td colspan="2">ERA5-MS</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Timer-XL</td><td>RoPE (2024)</td><td>with</td><td>0.340</td><td>0.238</td><td>0.157</td><td>0.205</td><td>0.162</td><td>0.221</td><td>0.164</td><td>0.307</td></tr><tr><td rowspan="3">Replace</td><td>ALiBi (2021)</td><td>with</td><td>0.351</td><td>0.246</td><td>0.162</td><td>0.212</td><td>0.188</td><td>0.210</td><td>0.167</td><td>0.308</td></tr><tr><td>Relative (2020)</td><td>with</td><td>0.361</td><td>0.250</td><td>0.163</td><td>0.214</td><td>0.197</td><td>0.215</td><td>0.168</td><td>0.309</td></tr><tr><td>Absolute (2017)</td><td>with</td><td>0.381</td><td>0.270</td><td>0.159</td><td>0.207</td><td>0.171</td><td>0.204</td><td>0.165</td><td>0.306</td></tr><tr><td rowspan="2">w/o</td><td>RoPE (2024)</td><td>w/o</td><td>0.361</td><td>0.254</td><td>0.171</td><td>0.217</td><td>0.181</td><td>0.221</td><td>0.235</td><td>0.373</td></tr><tr><td>w/o</td><td>w/o</td><td>0.363</td><td>0.253</td><td>0.164</td><td>0.215</td><td>0.194</td><td>0.215</td><td>0.167</td><td>0.309</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/7c0bc9769eb59da5b74690fc1f460405103770c78d7c54fbdf8a399b3d60bbab.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/69636d411d99eb99e7f0bc8f42457183e3263c62626fb4281539d9cfc128eef6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/068588a5d9ed88f12585e65d3b622d2f04442840f40c56f62477d22117d79015.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/1545bb3339879f441059a002e6140fc17d7a739dd875568fb5223a90959fdeb3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/7cd05253c2dd5a443e46716d0da56420db83c0694489db46123a4208d93038d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/47fbc0afdcdbff67ee362de0cfd2e47f306b992e37597c75dc20c810dedc358e.jpg)



Figure 11: Case studies of learned attention in encoder-/decoder-only Transformers.



Table 15: Performance on ERA5 (pred-1day). Lookback lengths vary from daily to yearly contexts.


<table><tr><td>Models</td><td colspan="2">Timer-XL</td><td colspan="2">PatchTST</td><td colspan="2">DLinear</td></tr><tr><td>Metric</td><td colspan="2">MSE | MAE</td><td colspan="2">MSE | MAE</td><td colspan="2">MSE | MAE</td></tr><tr><td>Lookback-8 (1 Day)</td><td colspan="2">0.0847 | 0.2100</td><td colspan="2">0.0897 | 0.2196</td><td colspan="2">0.0970 | 0.2276</td></tr><tr><td>Lookback-32 (4 Day)</td><td colspan="2">0.0713 | 0.1928</td><td colspan="2">0.0778 | 0.2080</td><td colspan="2">0.0841 | 0.2113</td></tr><tr><td>Lookback-56 (1 Week)</td><td colspan="2">0.0688 | 0.1891</td><td colspan="2">0.0785 | 0.2082</td><td colspan="2">0.0814 | 0.2081</td></tr><tr><td>Lookback-224 (1 Month)</td><td colspan="2">0.0675 | 0.1868</td><td colspan="2">0.0745 | 0.2042</td><td colspan="2">0.0788 | 0.2048</td></tr><tr><td>Lookback-960 (4 Month)</td><td colspan="2">0.0667 | 0.1863</td><td colspan="2">0.1194 | 0.2696</td><td colspan="2">0.0773 | 0.2031</td></tr><tr><td>Lookback-2944 (1 Year)</td><td colspan="2">0.0663 | 0.1857</td><td colspan="2">0.1109 | 0.2638</td><td colspan="2">0.0763 | 0.2024</td></tr></table>

Representation Analysis We further delve into long-context modeling from the perspective of learned representations. As shown in Figure 11, the decoder-only model can selectively focus on the previous context while PatchTST wrongly focuses on noisy parts. Since causality is the basis of forecasting, using causal masks leads to coherent token embeddings, while the unmasked attention mechanism may break the causality and prevent the model from telling each tokens. 

Normalization Section 4.1 has discussed instance normalization (Kim et al., 2021). It generally improves the performance of the previous encoder-only Transformers but leads to special problems in decoder-only Transformers (e.g., unmatched statistics in multi-step autoregression). However, it is indicative that Timer-XL without ReVIN can achieve competitive performance on well-acknowledged benchmarks in Table 16, while the performance of PatchTST may heavily rely on this normalization. 

## E.5 ILLUSTRATION OF TIMEATTENTION

Although the formulation to generalize from 1D sequences to multivariate time series is straightforward, Timer-XL is built on a decoder-only Transformer, an underexploited backbone among current time series models. As shown in Figure 12, challenges lie in capturing fine-grained dependencies between all variables in the patch level, while maintaining temporal causality in multiple sequences. 


Table 16: Evaluations (672-pred-96) on the effect of ReVIN (Kim et al., 2021) on Transformers.


<table><tr><td>Models</td><td colspan="2">Timer-XL with ReVIN</td><td colspan="2">Timer-XL w/o ReVIN</td><td colspan="2">PatchTST with ReVIN</td><td colspan="2">PatchTST w/o ReVIN</td></tr><tr><td>Metric</td><td colspan="2">MSE | MAE</td><td colspan="2">MSE | MAE</td><td colspan="2">MSE | MAE</td><td colspan="2">MSE | MAE</td></tr><tr><td>ETTh1</td><td colspan="2">0.364 | 0.397</td><td colspan="2">0.370 | 0.401</td><td colspan="2">0.370 | 0.399</td><td colspan="2">0.421 | 0.448</td></tr><tr><td>Weather</td><td colspan="2">0.157 | 0.205</td><td colspan="2">0.151 | 0.205</td><td colspan="2">0.149 | 0.198</td><td colspan="2">0.173 | 0.242</td></tr><tr><td>ECL</td><td colspan="2">0.127 | 0.219</td><td colspan="2">0.130 | 0.225</td><td colspan="2">0.129 | 0.222</td><td colspan="2">0.138 | 0.244</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e7661869-19e9-41a2-9a7c-6233ce9e80ff/394dd2d4e039b109b73a5ead58a570e2928789c9139bc613c50640afb3ae3db0.jpg)



Figure 12: Illustration of TimeAttention for modeling univariate and multivariate time series.


Technically, we introduce the masking formulation, whose key lies in the grouped causality of flattened 2D sequences. We derive it based on the Kronecker Product, which disentangles the large attention map into formalizable temporal and variable dependencies. It can be naturally extended to covariates or pre-defined variable dependencies, which may inspire a lot of future explorations. 

## F LIMITATIONS

Timer-XL is a unified model for time series forecasting. It can be used for task-specific training or scalable pre-training, handling varying-length and multivariate time series. As an autoregressive model, Timer-XL necessitates iterative generation for long-term forecasting, which may lead to error accumulation and inflexibility in the output length. In the future, we plan to incorporate multiresolution patches for input and output series. Furthermore, given that Timer-XL explicitly captures fine-grained token dependencies, there remains significant potential to reduce the complexity of TimeAttention, particularly in high-dimensional and lengthy time series. Finally, we will investigate the factors contributing to the stagnation of Transformer performance in extremely long contexts, and seek insights in the time series modality to improve context efficiency. 