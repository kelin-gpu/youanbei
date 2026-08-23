# Are Transformers Effective for Time Series Forecasting?

Ailing Zeng<sup>1,2</sup>*, Muxi Chen<sup>1</sup>*, Lei Zhang<sup>2</sup>, Qiang Xu<sup>1</sup> 

<sup>1</sup>The Chinese University of Hong Kong 

<sup>2</sup>International Digital Economy Academy 

{zengailing, leizhang}@idea.edu.cn,{mxchen21, qxu}@cse.cuhk.edu.hk 

## Abstract

Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Specifically, Transformers is arguably the most successful solution to extract the semantic correlations among the elements in a long sequence. However, in time series modeling, we are to extract the temporal relations in an ordered set of continuous points. While employing positional encoding and using tokens to embed sub-series in Transformers facilitate preserving some ordering information, the nature of the permutationinvariant self-attention mechanism inevitably results in temporal information loss. 

To validate our claim, we introduce a set of embarrassingly simple one-layer linear models named LTSF-Linear for comparison. Experimental results on nine real-life datasets show that LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models in all cases, and often by a large margin. Moreover, we conduct comprehensive empirical studies to explore the impacts of various design elements of LTSF models on their temporal relation extraction capability. We hope this surprising finding opens up new research directions for the LTSF task. We also advocate revisiting the validity of Transformer-based solutions for other time series analysis tasks (e.g., anomaly detection) in the future. 

## Introduction

Time series are ubiquitous in today’s data-driven world. Given historical data, time series forecasting (TSF) is a long-standing task that has a wide range of applications, including but not limited to traffic flow estimation, energy management, and financial investment. Over the past several decades, TSF solutions have undergone a progression from traditional statistical methods (e.g., ARIMA (Ariyo, Adewumi, and Ayo 2014)) and machine learning techniques (e.g., GBRT (Friedman 2001)) to deep learning-based solutions, e.g., (Bai, Kolter, and Koltun 2018; Liu et al. 2022). 

Transformer (Vaswani et al. 2017) is arguably the most successful sequence modeling architecture, demonstrating unparalleled performances in various applications, such as natural language processing (NLP) (Devlin et al. 2018), speech recognition (Dong, Xu, and Xu 2018), and computer vision (Liu et al. 2021b). Recently, there has also been a surge of Transformer-based solutions for time series analysis, as surveyed in (Wen et al. 2022). Most notable models, which focus on the less explored and challenging longterm time series forecasting (LTSF) problem, include Log-Trans (Li et al. 2019) (NeurIPS 2019), Informer (Zhou et al. 2021) (AAAI 2021 Best paper), Autoformer (Xu et al. 2021) (NeurIPS 2021), Pyraformer (Liu et al. 2021a) (ICLR 2022 Oral), Triformer (Cirstea et al. 2022) (IJCAI 2022) and the recent FEDformer (Zhou et al. 2022) (ICML 2022). 

The main working power of Transformers is from its multi-head self-attention mechanism, which has a remarkable capability of extracting semantic correlations among elements in a long sequence (e.g., words in texts or 2D patches in images). However, self-attention is permutation-invariant and “anti-order” to some extent. While using various types of positional encoding techniques can preserve some ordering information, it is still inevitable to have temporal information loss after applying self-attention on top of them. This is usually not a serious concern for semantic-rich applications such as NLP, e.g., the semantic meaning of a sentence is largely preserved even if we reorder some words in it. However, when analyzing time series data, there is usually a lack of semantics in the numerical data itself, and we are mainly interested in modeling the temporal changes among a continuous set of points. That is, the order itself plays the most crucial role. Consequently, we pose the following intriguing question: Are Transformers really effective for long-term time seriesforecasting? 

Moreover, while existing Transformer-based LTSF solutions have demonstrated considerable prediction accuracy improvements over traditional methods, in their experiments, all the compared (non-Transformer) baselines perform autoregressive or iterated multi-step (IMS) forecasting (Ariyo, Adewumi, and Ayo 2014; Salinas, Flunkert, and Gasthaus 2017; Bahdanau, Cho, and Bengio 2014; Taylor and Letham 2017), which are known to suffer from significant error accumulation effects for the LTSF problem. Therefore, in this work, we challenge Transformer-based LTSF solutions with direct multi-step (DMS) forecasting strategies to validate their real performance. 

Not all time series are predictable, let alone long-term forecasting (e.g., for chaotic systems). We hypothesize that long-term forecasting is only feasible for those time series with a relatively clear trend and periodicity. As linear models can already extract such information, we introduce a set of embarrassingly simple models named LTSF-Linear as a new baseline for comparison. LTSF-Linear regresses historical time series with a one-layer linear model to forecast future time series directly. We conduct extensive experiments on nine widely-used benchmark datasets that cover various real-life applications: traffic, energy, economics, weather, and disease predictions. Surprisingly, our results show that LTSF-Linear outperforms existing complex Transformerbased models in all cases, and often by a large margin (20% ∼ 50%). Moreover, we find that, in contrast to the claims in existing Transformers, most of them fail to extract temporal relations from long sequences, i.e., the forecasting errors are not reduced (sometimes even increased) with the increase of look-back window sizes. Finally, we conduct various ablation studies on existing Transformer-based TSF solutions to study the impact of various design elements in them. 

To sum up, the contributions of this work include: 

• To the best of our knowledge, this is the first work to challenge the effectiveness of the booming Transformers for the long-term time series forecasting task. 

• To validate our claims, we introduce a set of embarrassingly simple one-layer linear models, named LTSF-Linear, and compare them with existing Transformerbased LTSF solutions on nine benchmarks. LTSF-Linear can be a new baseline for the LTSF problem. 

• We conduct comprehensive empirical studies on various aspects of existing Transformer-based solutions, including the capability of modeling long inputs, the sensitivity to time series order, the impact of positional encoding and sub-series embedding, and efficiency comparisons. Our findings would benefit future research in this area. 

With the above, we conclude that the temporal modeling capabilities ofTransformersfor time series are exaggerated, at leastfor the existing LTSF benchmarks. At the same time, while LTSF-Linear achieves a better prediction accuracy compared to existing works, it merely serves as a simple baseline for future research on the challenging long-term TSF problem. With our findings, we also advocate revisiting the validity of Transformer-based solutions for other time series analysis tasks (e.g., anomaly detection) in the future. 

## Preliminaries: TSF Problem Formulation

For time series containing C variates, given historical data $\mathcal { X } = \{ X _ { 1 } ^ { t } , . . . , X _ { C } ^ { t } \} _ { t = 1 } ^ { L }$ , wherein L is the look-back window size and ${ \bar { X } } _ { i } ^ { t }$ is the value of the $i _ { t h }$ variate at the $t _ { t h }$ time step. The time series forecasting task is to predict the values $\hat { \mathcal X } =$ $\{ \hat { X } _ { 1 } ^ { t } , . . . , \hat { X } _ { C } ^ { t } \} _ { t = L + 1 } ^ { L + T }$ at the T future time steps. When $T > 1$ iterated multi-step (IMS) forecasting (Taieb, Hyndman et al. 2012) learns a single-step forecaster and iteratively applies it to obtain multi-step predictions. Alternatively, direct multistep (DMS) forecasting (Chevillon 2007) directly optimizes the multi-step forecasting objective at once. 

Compared to DMS forecasting results, IMS predictions have smaller variance thanks to the autoregressive estimation procedure, but they inevitably suffer from error accumulation effects. Consequently, IMS forecasting is preferable when there is a highly-accurate single-step forecaster, and T is relatively small. In contrast, DMS forecasting generates more accurate predictions when it is hard to obtain an unbiased single-step forecasting model, or T is large. 

## Transformer-Based LTSF Solutions

Transformer-based models (Vaswani et al. 2017) have achieved unparalleled performances in many long-standing AI tasks in natural language processing and computer vision fields, thanks to the effectiveness of the multi-head self-attention mechanism. This has also triggered lots of research interest in Transformer-based time series modeling techniques (Wen et al. 2022). In particular, a large amount of research works are dedicated to the LTSF task (e.g., (Li et al. 2019; Liu et al. 2021a; Xu et al. 2021; Zhou et al. 2021, 2022)). Considering the ability to capture long-range dependencies with Transformer models, most of them focus on the less-explored long-term forecasting problem $( T \gg 1 ) ^ { 1 }$ 

When applying the vanilla Transformer model to the LTSF problem, it has some limitations, including the quadratic time/memory complexity with the original selfattention scheme and error accumulation caused by the autoregressive decoder design. Informer (Zhou et al. 2021) addresses these issues and proposes a novel Transformer architecture with reduced complexity and a DMS forecasting strategy. Later, more Transformer variants introduce various time series features into their models for performance or efficiency improvements (Liu et al. 2021a; Xu et al. 2021; Zhou et al. 2022). We summarize the design elements of existing Transformer-based LTSF solutions as follows (see Figure 1). Time series decomposition: For data preprocessing, normalization with zero-mean is common in TSF. Besides, Autoformer (Xu et al. 2021) first applies seasonal-trend decomposition behind each neural block, which is a standard method in time series analysis to make raw data more predictable (Cleveland 1990; Hamilton 2020). Specifically, they use a moving average kernel on the input sequence to extract the trend-cyclical component of the time series. The difference between the original sequence and the trend component is regarded as the seasonal component. On top of the decomposition scheme of Autoformer, FEDformer (Zhou et al. 2022) further proposes the mixture of experts’ strategies to mix the trend components extracted by moving average kernels with various kernel sizes. 

Input embedding strategies: The self-attention layer in the Transformer architecture cannot preserve the positional information of the time series. However, local positional information, i.e. the ordering of time series, is important. Besides, global temporal information, such as hierarchical timestamps (week, month, year) and agnostic timestamps (holidays and events), is also informative (Zhou et al. 2021). To enhance the temporal context of time-series inputs, a practical design in the SOTA Transformer-based methods is injecting several embeddings, like a fixed positional encoding, a channel projection embedding, and learnable temporal embeddings into the input sequence. Moreover, temporal embeddings with a temporal convolution layer (Li et al. 2019) or learnable timestamps (Xu et al. 2021) are introduced. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/5a370506-7274-4949-9a14-7e26e4e30cd2/7497ef7665ef42f350eb2b5caf0a839774f01175edac79a6febe0d9321f466cf.jpg)



Figure 1: The pipeline of existing Transformer-based TSF solutions. In (a) and (b), the solid boxes are essential operations, and the dotted boxes are applied optionally. (c) and (d) are distinct for different methods (Li et al. 2019; Zhou et al. 2021; Xu et al. 2021; Liu et al. 2021a; Zhou et al. 2022).


Self-attention schemes: Transformers rely on the selfattention mechanism to extract the semantic dependencies between paired elements. Motivated by reducing the O  L<sup>2</sup> time and memory complexity of the vanilla Transformer, recent works propose two strategies for efficiency. On the one hand, LogTrans and Pyraformer explicitly introduce a sparsity bias into the self-attention scheme. Specifically, Log-Trans uses a Logsparse mask to reduce the computational complexity to O (LlogL) while Pyraformer adopts pyramidal attention that captures hierarchically multi-scale temporal dependencies with an O (L) time and memory complexity. On the other hand, Informer and FEDformer use the low-rank property in the self-attention matrix. Informer proposes a ProbSparse self-attention mechanism and a selfattention distilling operation to decrease the complexity to O (LlogL), and FEDformer designs a Fourier enhanced block and a wavelet enhanced block with random selection to obtain O (L) complexity. Lastly, Autoformer designs a series-wise auto-correlation mechanism to replace the original self-attention layer. 

Decoders: The vanilla Transformer decoder outputs sequences in an autoregressive manner, resulting in a slow inference speed and error accumulation effects, especially for long-term predictions. Informer designs a generative-style decoder for DMS forecasting. Other Transformer variants employ similar DMS strategies. For instance, Pyraformer uses a fully-connected layer concatenating Spatio-temporal axes as the decoder. Autoformer sums up two refined decomposed features from trend-cyclical components and the stacked auto-correlation mechanism for seasonal components to get the final prediction. FEDformer also uses a decomposition scheme with the proposed frequency attention block to decode the final results. 

The premise of Transformer models is the semantic correlations between paired elements, while the self-attention mechanism itself is permutation-invariant, and its capability of modeling temporal relations largely depends on positional encodings associated with input tokens. Considering the raw numerical data in time series (e.g., stock prices or electricity values), there are hardly any point-wise semantic correlations between them. In time series modeling, we are mainly interested in the temporal relations among a continuous set of points, and the order of these elements instead of the paired relationship plays the most crucial role. While employing positional encoding and using tokens to embed sub-series facilitate preserving some ordering information, the nature of the permutation-invariant self-attention mechanism inevitably results in temporal information loss. Due to the above observations, we are interested in revisiting the effectiveness of Transformer-based LTSF solutions. 

## An Embarrassingly Simple Baseline for LTSF

In the experiments of existing Transformer-based LTSF solutions (T ≫ 1), all the compared (non-Transformer) baselines are IMS forecasting techniques, which are known to suffer from significant error accumulation effects. We hypothesize that the performance improvements in these works are largely due to the DMS strategy used in them. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/5a370506-7274-4949-9a14-7e26e4e30cd2/a3c112c914d6ebfc0cbc0b87d98efeaa1fa101ed87ad9b067ab49dd423da1f49.jpg)



Figure 2: Illustration of the basic linear model.


To validate this hypothesis, we present the simplest DMS model via a temporal linear layer, named LTSF-Linear, as a baseline for comparison. The basic formulation of LTSF-Linear directly regresses historical time series for future prediction via a weighted sum operation (as illustrated in Figure 2). The mathematical expression is ${ \hat { X } } _ { i } = W X _ { i }$ , where 

<table><tr><td>Datasets</td><td>ETTh1&amp;ETTh2</td><td>ETTm1 &amp;ETTm2</td><td>Traffic</td><td>Electricity</td><td>Exchange-Rate</td><td>Weather</td><td>ILI</td></tr><tr><td>Variates</td><td>7</td><td>7</td><td>862</td><td>321</td><td>8</td><td>21</td><td>7</td></tr><tr><td>Timesteps</td><td>17,420</td><td>69,680</td><td>17,544</td><td>26,304</td><td>7,588</td><td>52,696</td><td>966</td></tr><tr><td>Granularity</td><td>1hour</td><td>5min</td><td>1hour</td><td>1hour</td><td>1day</td><td>10min</td><td>1week</td></tr></table>


Table 1: The statistics of the nine popular datasets for the LTSF problem.


<table><tr><td colspan="2">Methods</td><td>IMP.</td><td colspan="2">Linear*</td><td colspan="2">NLinear*</td><td colspan="2">DLinear*</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Pyraformer*</td><td colspan="2">Repeat*</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>27%</td><td>0.140</td><td>0.237</td><td>0.141</td><td>0.237</td><td>0.140</td><td>0.237</td><td>0.193</td><td>0.308</td><td>0.201</td><td>0.317</td><td>0.274</td><td>0.368</td><td>0.386</td><td>0.449</td><td>1.588</td><td>0.946</td></tr><tr><td>192</td><td>24%</td><td>0.153</td><td>0.250</td><td>0.154</td><td>0.248</td><td>0.153</td><td>0.249</td><td>0.201</td><td>0.315</td><td>0.222</td><td>0.334</td><td>0.296</td><td>0.386</td><td>0.386</td><td>0.443</td><td>1.595</td><td>0.950</td></tr><tr><td>336</td><td>21%</td><td>0.169</td><td>0.268</td><td>0.171</td><td>0.265</td><td>0.169</td><td>0.267</td><td>0.214</td><td>0.329</td><td>0.231</td><td>0.338</td><td>0.300</td><td>0.394</td><td>0.378</td><td>0.443</td><td>1.617</td><td>0.961</td></tr><tr><td>720</td><td>17%</td><td>0.203</td><td>0.301</td><td>0.210</td><td>0.297</td><td>0.203</td><td>0.301</td><td>0.246</td><td>0.355</td><td>0.254</td><td>0.361</td><td>0.373</td><td>0.439</td><td>0.376</td><td>0.445</td><td>1.647</td><td>0.975</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>45%</td><td>0.082</td><td>0.207</td><td>0.089</td><td>0.208</td><td>0.081</td><td>0.203</td><td>0.148</td><td>0.278</td><td>0.197</td><td>0.323</td><td>0.847</td><td>0.752</td><td>0.376</td><td>1.105</td><td>0.081</td><td>0.196</td></tr><tr><td>192</td><td>42%</td><td>0.167</td><td>0.304</td><td>0.180</td><td>0.300</td><td>0.157</td><td>0.293</td><td>0.271</td><td>0.380</td><td>0.300</td><td>0.369</td><td>1.204</td><td>0.895</td><td>1.748</td><td>1.151</td><td>0.167</td><td>0.289</td></tr><tr><td>336</td><td>34%</td><td>0.328</td><td>0.432</td><td>0.331</td><td>0.415</td><td>0.305</td><td>0.414</td><td>0.460</td><td>0.500</td><td>0.509</td><td>0.524</td><td>1.672</td><td>1.036</td><td>1.874</td><td>1.172</td><td>0.305</td><td>0.396</td></tr><tr><td>720</td><td>46%</td><td>0.964</td><td>0.750</td><td>1.033</td><td>0.780</td><td>0.643</td><td>0.601</td><td>1.195</td><td>0.841</td><td>1.447</td><td>0.941</td><td>2.478</td><td>1.310</td><td>1.943</td><td>1.206</td><td>0.823</td><td>0.681</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>30%</td><td>0.410</td><td>0.282</td><td>0.410</td><td>0.279</td><td>0.410</td><td>0.282</td><td>0.587</td><td>0.366</td><td>0.613</td><td>0.388</td><td>0.719</td><td>0.391</td><td>2.085</td><td>0.468</td><td>2.723</td><td>1.079</td></tr><tr><td>192</td><td>30%</td><td>0.423</td><td>0.287</td><td>0.423</td><td>0.284</td><td>0.423</td><td>0.287</td><td>0.604</td><td>0.373</td><td>0.616</td><td>0.382</td><td>0.696</td><td>0.379</td><td>0.867</td><td>0.467</td><td>2.756</td><td>1.087</td></tr><tr><td>336</td><td>30%</td><td>0.436</td><td>0.295</td><td>0.435</td><td>0.290</td><td>0.436</td><td>0.296</td><td>0.621</td><td>0.383</td><td>0.622</td><td>0.337</td><td>0.777</td><td>0.420</td><td>0.869</td><td>0.469</td><td>2.791</td><td>1.095</td></tr><tr><td>720</td><td>26%</td><td>0.466</td><td>0.315</td><td>0.464</td><td>0.307</td><td>0.466</td><td>0.315</td><td>0.626</td><td>0.382</td><td>0.660</td><td>0.408</td><td>0.864</td><td>0.472</td><td>0.881</td><td>0.473</td><td>2.811</td><td>1.097</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>19%</td><td>0.176</td><td>0.236</td><td>0.182</td><td>0.232</td><td>0.176</td><td>0.237</td><td>0.217</td><td>0.296</td><td>0.266</td><td>0.336</td><td>0.300</td><td>0.384</td><td>0.896</td><td>0.556</td><td>0.259</td><td>0.254</td></tr><tr><td>192</td><td>21%</td><td>0.218</td><td>0.276</td><td>0.225</td><td>0.269</td><td>0.220</td><td>0.282</td><td>0.276</td><td>0.336</td><td>0.307</td><td>0.367</td><td>0.598</td><td>0.544</td><td>0.622</td><td>0.624</td><td>0.309</td><td>0.292</td></tr><tr><td>336</td><td>23%</td><td>0.262</td><td>0.312</td><td>0.271</td><td>0.301</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.380</td><td>0.359</td><td>0.395</td><td>0.578</td><td>0.523</td><td>0.739</td><td>0.753</td><td>0.377</td><td>0.338</td></tr><tr><td>720</td><td>20%</td><td>0.326</td><td>0.365</td><td>0.338</td><td>0.348</td><td>0.323</td><td>0.362</td><td>0.403</td><td>0.428</td><td>0.419</td><td>0.428</td><td>1.059</td><td>0.741</td><td>1.004</td><td>0.934</td><td>0.465</td><td>0.394</td></tr><tr><td rowspan="4">ILI</td><td>24</td><td>48%</td><td>1.947</td><td>0.985</td><td>1.683</td><td>0.858</td><td>2.215</td><td>1.081</td><td>3.228</td><td>1.260</td><td>3.483</td><td>1.287</td><td>5.764</td><td>1.677</td><td>1.420</td><td>2.012</td><td>6.587</td><td>1.701</td></tr><tr><td>36</td><td>36%</td><td>2.182</td><td>1.036</td><td>1.703</td><td>0.859</td><td>1.963</td><td>0.963</td><td>2.679</td><td>1.080</td><td>3.103</td><td>1.148</td><td>4.755</td><td>1.467</td><td>7.394</td><td>2.031</td><td>7.130</td><td>1.884</td></tr><tr><td>48</td><td>34%</td><td>2.256</td><td>1.060</td><td>1.719</td><td>0.884</td><td>2.130</td><td>1.024</td><td>2.622</td><td>1.078</td><td>2.669</td><td>1.085</td><td>4.763</td><td>1.469</td><td>7.551</td><td>2.057</td><td>6.575</td><td>1.798</td></tr><tr><td>60</td><td>34%</td><td>2.390</td><td>1.104</td><td>1.819</td><td>0.917</td><td>2.368</td><td>1.096</td><td>2.857</td><td>1.157</td><td>2.770</td><td>1.125</td><td>5.264</td><td>1.564</td><td>7.662</td><td>2.100</td><td>5.893</td><td>1.677</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>1%</td><td>0.375</td><td>0.397</td><td>0.374</td><td>0.394</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.419</td><td>0.449</td><td>0.459</td><td>0.865</td><td>0.713</td><td>0.664</td><td>0.612</td><td>1.295</td><td>0.713</td></tr><tr><td>192</td><td>4%</td><td>0.418</td><td>0.429</td><td>0.408</td><td>0.415</td><td>0.405</td><td>0.416</td><td>0.420</td><td>0.448</td><td>0.500</td><td>0.482</td><td>1.008</td><td>0.792</td><td>0.790</td><td>0.681</td><td>1.325</td><td>0.733</td></tr><tr><td>336</td><td>7%</td><td>0.479</td><td>0.476</td><td>0.429</td><td>0.427</td><td>0.439</td><td>0.443</td><td>0.459</td><td>0.465</td><td>0.521</td><td>0.496</td><td>1.107</td><td>0.809</td><td>0.891</td><td>0.738</td><td>1.323</td><td>0.744</td></tr><tr><td>720</td><td>13%</td><td>0.624</td><td>0.592</td><td>0.440</td><td>0.453</td><td>0.472</td><td>0.490</td><td>0.506</td><td>0.507</td><td>0.514</td><td>0.512</td><td>1.181</td><td>0.865</td><td>0.963</td><td>0.782</td><td>1.339</td><td>0.756</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>20%</td><td>0.288</td><td>0.352</td><td>0.277</td><td>0.338</td><td>0.289</td><td>0.353</td><td>0.346</td><td>0.388</td><td>0.358</td><td>0.397</td><td>3.755</td><td>1.525</td><td>0.645</td><td>0.597</td><td>0.432</td><td>0.422</td></tr><tr><td>192</td><td>20%</td><td>0.377</td><td>0.413</td><td>0.344</td><td>0.381</td><td>0.383</td><td>0.418</td><td>0.429</td><td>0.439</td><td>0.456</td><td>0.452</td><td>5.602</td><td>1.931</td><td>0.788</td><td>0.683</td><td>0.534</td><td>0.473</td></tr><tr><td>336</td><td>26%</td><td>0.452</td><td>0.461</td><td>0.357</td><td>0.400</td><td>0.448</td><td>0.465</td><td>0.496</td><td>0.487</td><td>0.482</td><td>0.486</td><td>4.721</td><td>1.835</td><td>0.907</td><td>0.747</td><td>0.591</td><td>0.508</td></tr><tr><td>720</td><td>14%</td><td>0.698</td><td>0.595</td><td>0.394</td><td>0.436</td><td>0.605</td><td>0.551</td><td>0.463</td><td>0.474</td><td>0.515</td><td>0.511</td><td>3.647</td><td>1.625</td><td>0.963</td><td>0.783</td><td>0.588</td><td>0.517</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>21%</td><td>0.308</td><td>0.352</td><td>0.306</td><td>0.348</td><td>0.299</td><td>0.343</td><td>0.379</td><td>0.419</td><td>0.505</td><td>0.475</td><td>0.672</td><td>0.571</td><td>0.543</td><td>0.510</td><td>1.214</td><td>0.665</td></tr><tr><td>192</td><td>21%</td><td>0.340</td><td>0.369</td><td>0.349</td><td>0.375</td><td>0.335</td><td>0.365</td><td>0.426</td><td>0.441</td><td>0.553</td><td>0.496</td><td>0.795</td><td>0.669</td><td>0.557</td><td>0.537</td><td>1.261</td><td>0.690</td></tr><tr><td>336</td><td>17%</td><td>0.376</td><td>0.393</td><td>0.375</td><td>0.388</td><td>0.369</td><td>0.386</td><td>0.445</td><td>0.459</td><td>0.621</td><td>0.537</td><td>1.212</td><td>0.871</td><td>0.754</td><td>0.655</td><td>1.283</td><td>0.707</td></tr><tr><td>720</td><td>22%</td><td>0.440</td><td>0.435</td><td>0.433</td><td>0.422</td><td>0.425</td><td>0.421</td><td>0.543</td><td>0.490</td><td>0.671</td><td>0.561</td><td>1.166</td><td>0.823</td><td>0.908</td><td>0.724</td><td>1.319</td><td>0.729</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>18%</td><td>0.168</td><td>0.262</td><td>0.167</td><td>0.255</td><td>0.167</td><td>0.260</td><td>0.203</td><td>0.287</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.435</td><td>0.507</td><td>0.266</td><td>0.328</td></tr><tr><td>192</td><td>18%</td><td>0.232</td><td>0.308</td><td>0.221</td><td>0.293</td><td>0.224</td><td>0.303</td><td>0.269</td><td>0.328</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td><td>0.730</td><td>0.673</td><td>0.340</td><td>0.371</td></tr><tr><td>336</td><td>16%</td><td>0.320</td><td>0.373</td><td>0.274</td><td>0.327</td><td>0.281</td><td>0.342</td><td>0.325</td><td>0.366</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td><td>1.201</td><td>0.845</td><td>0.412</td><td>0.410</td></tr><tr><td>720</td><td>13%</td><td>0.413</td><td>0.435</td><td>0.368</td><td>0.384</td><td>0.397</td><td>0.421</td><td>0.421</td><td>0.415</td><td>0.433</td><td>0.432</td><td>3.379</td><td>1.338</td><td>3.625</td><td>1.451</td><td>0.521</td><td>0.465</td></tr></table>


- Methods* are implemented by us; Other results are from FEDformer (Zhou et al. 2022).



Table 2: Multivariate long-term forecasting errors in terms of MSE and MAE, the lower the better. Among them, ILI dataset is with forecasting horizon $\mathsf { \bar { T } } \in \{ 2 4 , 3 6 , 4 8 , \bar { 6 } 0 \}$ . For the others, $T \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . The best results are highlighted in bold and the best results of Transformers are highlighted with an underline. IMP. is the best result of linear models compared to the results of Transformer-based solutions.


$W \in \mathbb { R } ^ { T \times L }$ is a linear layer along the temporal axis. ${ \hat { X } } _ { i }$ and $X _ { i }$ are the prediction and input for each $i _ { t h }$ variate. Note that LTSF-Linear shares weights across different variates and does not model any spatial correlations. 

LTSF-Linear is a set of linear models. Vanilla Linear is a one-layer linear model. To handle time series across different domains (e.g., finance, traffic, and energy domains), we further introduce two variants with two preprocessing methods, named DLinear and NLinear. 

• Specifically, DLinear is a combination of a Decomposition scheme used in Autoformer and FEDformer with linear layers. It first decomposes a raw data input into a trend component by a moving average kernel and a remainder (seasonal) component. Then, two one-layer linear layers are applied to each component, and we sum up the two features to get the final prediction. By explicitly handling trend, DLinear enhances the performance of a vanilla linear when there is a clear trend in the data. 

• Meanwhile, to boost the performance of LTSF-Linear when there is a distribution shift in the dataset, NLinear first subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction. The subtraction and addition in NLinear are a simple normalization for the input sequence. 

## Experiments

## Experimental Settings

Dataset. We conduct extensive experiments on nine widely-used real-world datasets, including ETT (Electricity Transformer Temperature) (Zhou et al. 2021) (ETTh1, ETTh2, ETTm1, ETTm2), Traffic, Electricity, Weather, ILI, Exchange-Rate (Lai et al. 2017). All of them are multivariate time series. We leave data descriptions in the Appendix. 

Evaluation metric. Following previous works (Zhou et al. 2021; Xu et al. 2021; Zhou et al. 2022), we use Mean Squared Error (MSE) and Mean Absolute Error (MAE). 

Compared methods. We include four recent Transformer-based methods: FEDformer (Fourier) (Zhou et al. 2022), Autoformer (Xu et al. 2021), Informer (Zhou et al. 2021), Pyraformer (Liu et al. 2021a). Besides, we include a naive DMS method: Closest Repeat (Repeat), which repeats the last value in the look-back window. 

## Comparison with Transformers

Quantitative results. In Table 2, we extensively evaluate all mentioned Transformers on nine benchmarks, following the experimental setting of previous work (Xu et al. 2021; Zhou et al. 2022, 2021). Surprisingly, the performance of LTSF-Linear surpasses the SOTA FEDformer in most cases by 20% ∼ 50% improvements on the multivariate forecasting, where LTSF-Linear even does not model correlations among variates. For different time series benchmarks, NLinear and DLinear show the superiority to handle the distribution shift and trend-seasonality features. We also provide results for univariate forecasting of ETT datasets in the Appendix, where LTSF-Linear still consistently outperforms Transformer-based LTSF solutions by a large margin. In general, these results reveal that existing complex Transformer-based LTSF solutions are not seemingly effective on the existing nine benchmarks while LTSF-Linear can be a powerful baseline. Another interesting observation is that even though the naive Repeat method shows worse results when predicting long-term seasonal data (e.g., Electricity ), it surprisingly outperforms all Transformers on Exchange-Rate (around 45%). This is mainly caused by the wrong prediction of trends in Transformer-based solutions, which may overfit toward sudden change noises in the training data, resulting in significant accuracy degradation. 

Qualitative results. As shown in Figure 3, we plot the prediction results on three selected time series datasets with Transformer-based solutions and LTSF-Linear: Electricity (Sequence 1951, Variate 36) , where it has different temporal patterns. When the input length is 96 steps, and the output horizon is 336 steps, Transformers fail to capture the scale and bias of the future data . Moreover, they can hardly predict a proper trend on aperiodic data. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/5a370506-7274-4949-9a14-7e26e4e30cd2/3cd15bce82b5bd7fdeba00ce8c00fe129d512d0399ea189df7edeb78ff9f947b.jpg)



Figure 3: Illustration of the long-term forecasting outputs (Y-axis) of five models with an input length L=96 and output length T=192 (X-axis) on Electricity.


## More Analyses on Transformer-Based Solutions

Can existing LTSF-Transformers extract temporal relations well from longer input sequences? The size of the look-back window greatly impacts forecasting accuracy as it determines how much we can learn from historical data. Generally speaking, a powerful TSF model with a strong temporal relation extraction capability should be able to achieve better results with larger look-back window sizes. 

To study the impact of input look-back window sizes, we conduct experiments with $\bar { L } \in \{ 2 4 , . . . , 7 2 0 \}$ for long-term forecasting (T=720). Similar to the observations from previous studies (Zhou et al. 2021; Wen et al. 2022), existing Transformers’ performance deteriorates or stays stable when the look-back window size increases. In contrast, the performances of all LTSF-Linear are significantly boosted with the increase of look-back window size. Thus, existing solutions tend to overfit temporal noises instead of extracting temporal information if given a longer sequence, and the input size 96 is exactly suitable for most Transformers. 

What can be learned for long-term forecasting? While the temporal dynamics in the look-back window significantly impact the forecasting accuracy of short-term time series forecasting, we hypothesize that long-term forecasting depends on whether models can capture the trend and periodicity well only. That is, the farther the forecasting horizon, the less impact the look-back window itself has. 

<table><tr><td>Methods</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td></tr><tr><td>Input</td><td>Close</td><td>Far</td><td>Close</td><td>Far</td></tr><tr><td>Electricity</td><td>0.251</td><td>0.265</td><td>0.255</td><td>0.287</td></tr><tr><td>Traffic</td><td>0.631</td><td>0.645</td><td>0.677</td><td>0.675</td></tr></table>


Table 3: The MSE comparisons of different input sequences.


To validate the above hypothesis, in Table 3, we compare the forecasting accuracy for the same future 720 time steps with data from two different look-back windows: (i). the original input L=96 setting (called Close) and (ii). the far input L=96 setting (called Far) that is before the original 96 time steps. The performance of the SOTA Transformers drops slightly, indicating these models only capture similar temporal information from the adjacent time series sequence. Since capturing the intrinsic characteristics of the dataset generally does not require a large number of parameters, i,e. one parameter can represent the periodicity. Using too many parameters will even cause overfitting. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/5a370506-7274-4949-9a14-7e26e4e30cd2/f81028d5e0743c0591212b4cc3a7bc2ce04379a2509f39a070c8bc6bef9e83c8.jpg)



Figure 4: The MSE results (Y-axis) of models with different look-back window sizes (X-axis) of long-term forecasting (T=720) on Electricity.


Are the self-attention scheme effective for LTSF? We verify whether these complex designs in the existing Transformer (e.g., Informer) are essential. In Table 4, we gradually transform Informer to Linear. First, we replace each self-attention layer with a linear layer, called Att.-Linear, since a self-attention layer can be regarded as a fullyconnected layer where weights are dynamically changed. Furthermore, we discard other auxiliary designs (e.g., FFN) in Informer to leave embedding layers and linear layers, named Embed + Linear. Finally, we simplify the model to one linear layer. As can be observed, the performance of Informer grows with the gradual simplification, thereby challenging the necessity of these modules. 

<table><tr><td colspan="2">Methods</td><td>Informer</td><td>Att.-Linear</td><td>Embed + Linear</td><td>Linear</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.847</td><td>1.003</td><td>0.173</td><td>0.084</td></tr><tr><td>192</td><td>1.204</td><td>0.979</td><td>0.443</td><td>0.155</td></tr><tr><td>336</td><td>1.672</td><td>1.498</td><td>1.288</td><td>0.301</td></tr><tr><td>720</td><td>2.478</td><td>2.102</td><td>2.026</td><td>0.763</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.865</td><td>0.613</td><td>0.454</td><td>0.400</td></tr><tr><td>192</td><td>1.008</td><td>0.759</td><td>0.686</td><td>0.438</td></tr><tr><td>336</td><td>1.107</td><td>0.921</td><td>0.821</td><td>0.479</td></tr><tr><td>720</td><td>1.181</td><td>0.902</td><td>1.051</td><td>0.515</td></tr></table>


Table 4: The MSE comparisons of gradually transforming Informer to a Linear from the left to right columns.


Can existing LTSF-Transformers preserve temporal order well? Self-attention is inherently permutation-invariant, i.e., regardless of the order. However, in time-series forecasting, the sequence order often plays a crucial role. We argue that even with positional and temporal embeddings, existing Transformer-based methods still suffer from temporal information loss. In Table 5, we shuffle the raw input before the embedding strategies. Two shuffling strategies are presented: Shuf. randomly shuffles the whole input sequences and Half-Ex. exchanges the first half of the input sequence with the second half. Interestingly, compared with the original setting (Ori.) on the Exchange Rate, the performance of all Transformer-based methods does not fluctuate even when the input sequence is randomly shuffled. By contrary, the performance of LTSF-Linear is damaged significantly. These indicate that LTSF-Transformers with different positional and temporal embeddings preserve quite limited temporal relations and are prone to overfit on noisy financial data, while the simple LTSF-Linear can model the order naturally and avoid overfitting with fewer parameters. 

For the ETTh1 dataset, FEDformer and Autoformer introduce time series inductive bias into their models, making them can extract certain temporal information when the dataset has more clear temporal patterns (e.g., periodicity) than the Exchange Rate. Therefore, the average drops of the two Transformers are 73.28% and 56.91% under the Shuf. setting, where it loses the whole order information. Moreover, Informer still suffers less from both Shuf. and Half-Ex. settings due to its no such temporal inductive bias. Overall, the average drops of LTSF-Linear are larger than Transformer-based methods for all cases, indicating the existing Transformers do not preserve temporal order well. 

How effective are different embedding strategies? In Table 6, the forecasting errors of Informer largely increase without positional embeddings (wo/Pos.). Without timestamp embeddings (wo/Temp.) will gradually damage the performance of Informer as the forecasting lengths increase. Since Informer uses a single time step for each token, it is necessary to introduce temporal information in tokens. 

Rather than using a single time step in each token, FEDformer and Autoformer input a sequence of timestamps to embed the temporal information. Hence, they can achieve comparable or even better performance without fixed positional embeddings. However, without timestamp embeddings, the performance of Autoformer declines rapidly because of the loss of global temporal information. Instead, thanks to the frequency-enhanced module proposed in FEDformer to introduce temporal inductive bias, it suffers less from removing any position/timestamp embeddings. 

Is training data size a limitingfactorfor existing LTSF-Transformers? Some may argue that the poor performance of Transformer-based solutions is due to the small sizes of the benchmark datasets. Unlike computer vision or natural language processing tasks, TSF is performed on collected time series, and it is difficult to scale up the training data size. In fact, the size of the training data would indeed have a significant impact on the model performance. Accordingly, we conduct experiments on Traffic, comparing the performance of the model trained on a full dataset (17,544*0.7 hours), named Ori., with that training on a shortened dataset (8,760 hours, i.e., 1 year), called Short. Unexpectedly, Table 7 presents that the prediction errors with reduced training data are usually lower. This might be because the wholeyear data maintain clearer temporal features than a longer but incomplete data size. While we cannot conclude that we should use fewer data for training, it demonstrates that the training data scale is not the limiting reason. 

<table><tr><td colspan="2">Methods</td><td colspan="3">Linear</td><td colspan="3">FEDformer</td><td colspan="3">Autoformer</td><td colspan="3">Informer</td></tr><tr><td colspan="2">Predict Length</td><td>Ori.</td><td>Shuf.</td><td>Half-Ex.</td><td>Ori.</td><td>Shuf.</td><td>Half-Ex.</td><td>Ori.</td><td>Shuf.</td><td>Half-Ex.</td><td>Ori.</td><td>Shuf.</td><td>Half-Ex.</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.080</td><td>0.133</td><td>0.169</td><td>0.161</td><td>0.160</td><td>0.162</td><td>0.152</td><td>0.158</td><td>0.160</td><td>0.952</td><td>1.004</td><td>0.959</td></tr><tr><td>192</td><td>0.162</td><td>0.208</td><td>0.243</td><td>0.274</td><td>0.275</td><td>0.275</td><td>0.278</td><td>0.271</td><td>0.277</td><td>1.012</td><td>1.023</td><td>1.014</td></tr><tr><td>336</td><td>0.286</td><td>0.320</td><td>0.345</td><td>0.439</td><td>0.439</td><td>0.439</td><td>0.435</td><td>0.430</td><td>0.435</td><td>1.177</td><td>1.181</td><td>1.177</td></tr><tr><td>720</td><td>0.806</td><td>0.819</td><td>0.836</td><td>1.122</td><td>1.122</td><td>1.122</td><td>1.113</td><td>1.113</td><td>1.113</td><td>1.198</td><td>1.210</td><td>1.196</td></tr><tr><td></td><td>Average Drop</td><td>N/A</td><td>27.26%</td><td>46.81%</td><td>N/A</td><td>-0.09%</td><td>0.20%</td><td>N/A</td><td>0.09%</td><td>1.12%</td><td>N/A</td><td>-0.12%</td><td>-0.18%</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.395</td><td>0.824</td><td>0.431</td><td>0.376</td><td>0.753</td><td>0.405</td><td>0.455</td><td>0.838</td><td>0.458</td><td>0.974</td><td>0.971</td><td>0.971</td></tr><tr><td>192</td><td>0.447</td><td>0.824</td><td>0.471</td><td>0.419</td><td>0.730</td><td>0.436</td><td>0.486</td><td>0.774</td><td>0.491</td><td>1.233</td><td>1.232</td><td>1.231</td></tr><tr><td>336</td><td>0.490</td><td>0.825</td><td>0.505</td><td>0.447</td><td>0.736</td><td>0.453</td><td>0.496</td><td>0.752</td><td>0.497</td><td>1.693</td><td>1.693</td><td>1.691</td></tr><tr><td>720</td><td>0.520</td><td>0.846</td><td>0.528</td><td>0.468</td><td>0.720</td><td>0.470</td><td>0.525</td><td>0.696</td><td>0.524</td><td>2.720</td><td>2.716</td><td>2.715</td></tr><tr><td></td><td>Average Drop</td><td>N/A</td><td>81.06%</td><td>4.78%</td><td>N/A</td><td>73.28%</td><td>3.44%</td><td>N/A</td><td>56.91%</td><td>0.46%</td><td>N/A</td><td>1.98%</td><td>0.18%</td></tr></table>


Table 5: The MSE comparisons of models when shuffling the raw input sequence. Shuf. randomly shuffles the input sequence. Half-EX. randomly exchanges the first half of the input sequences with the second half. We run five times.


<table><tr><td rowspan="2">Methods</td><td rowspan="2">Embedding</td><td colspan="4">Traffic</td></tr><tr><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td rowspan="4">FEDformer</td><td>All</td><td>0.597</td><td>0.606</td><td>0.627</td><td>0.649</td></tr><tr><td>wo/Pos.</td><td>0.587</td><td>0.604</td><td>0.621</td><td>0.626</td></tr><tr><td>wo/Temp.</td><td>0.613</td><td>0.623</td><td>0.650</td><td>0.677</td></tr><tr><td>wo/Pos.-Temp.</td><td>0.613</td><td>0.622</td><td>0.648</td><td>0.663</td></tr><tr><td rowspan="4">Autoformer</td><td>All</td><td>0.629</td><td>0.647</td><td>0.676</td><td>0.638</td></tr><tr><td>wo/Pos.</td><td>0.613</td><td>0.616</td><td>0.622</td><td>0.660</td></tr><tr><td>wo/Temp.</td><td>0.681</td><td>0.665</td><td>0.908</td><td>0.769</td></tr><tr><td>wo/Pos.-Temp.</td><td>0.672</td><td>0.811</td><td>1.133</td><td>1.300</td></tr><tr><td rowspan="4">Informer</td><td>All</td><td>0.719</td><td>0.696</td><td>0.777</td><td>0.864</td></tr><tr><td>wo/Pos.</td><td>1.035</td><td>1.186</td><td>1.307</td><td>1.472</td></tr><tr><td>wo/Temp.</td><td>0.754</td><td>0.780</td><td>0.903</td><td>1.259</td></tr><tr><td>wo/Pos.-Temp.</td><td>1.038</td><td>1.351</td><td>1.491</td><td>1.512</td></tr></table>


Table 6: The MSE comparisons of different embedding strategies on Transformer-based methods with look-back window size 96 and forecasting lengths {96, 192, 336, 720}.


<table><tr><td>Methods</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td></tr><tr><td>Dataset</td><td>Ori.</td><td>Short</td><td>Ori.</td><td>Short</td></tr><tr><td>96</td><td>0.587</td><td>0.568</td><td>0.613</td><td>0.594</td></tr><tr><td>192</td><td>0.604</td><td>0.584</td><td>0.616</td><td>0.621</td></tr><tr><td>336</td><td>0.621</td><td>0.601</td><td>0.622</td><td>0.621</td></tr><tr><td>720</td><td>0.626</td><td>0.608</td><td>0.660</td><td>0.650</td></tr></table>


Table 7: The MSE comparisons of two training data sizes.


Is efficiency really a top-level priority? Existing LTSF-Transformers claim that the $O \bar { ( } L ^ { 2 } )$ complexity of the vanilla Transformer is unaffordable for the LTSF problem. Although they prove to be able to improve the theoretical time and memory complexity from $\stackrel { \bullet } { O ^ { } } ( L ^ { 2 } )$ to O (L), it is unclear whether 1) the actual inference time and memory cost on devices are improved, and 2) the memory issue is unacceptable and urgent for today’s GPU (e.g., an NVIDIA Titan XP here). In Table 8, we compare the average prac-


tical efficiencies with 5 runs. Interestingly, compared with the vanilla Transformer (with the same DMS decoder), most Transformer variants incur similar or worse inference time and parameters in practice. These follow-ups introduce more additional design elements to make practical costs high.


<table><tr><td>Method</td><td>MACs</td><td>Parameter</td><td>Time</td><td>Memory</td></tr><tr><td>DLinear</td><td>0.04G</td><td>139.7K</td><td>0.4ms</td><td>687MiB</td></tr><tr><td>Transformer×</td><td>4.03G</td><td>13.61M</td><td>26.8ms</td><td>6091MiB</td></tr><tr><td>Informer</td><td>3.93G</td><td>14.39M</td><td>49.3ms</td><td>3869MiB</td></tr><tr><td>Autoformer</td><td>4.41G</td><td>14.91M</td><td>164.1ms</td><td>7607MiB</td></tr><tr><td>Pyraformer</td><td>0.80G</td><td>241.4M</td><td>3.4ms</td><td>7017MiB</td></tr><tr><td>FEDformer</td><td>4.41G</td><td>20.68M</td><td>40.5ms</td><td>4143MiB</td></tr></table>


× the same one-step decoder. 



Table 8: Comparison of practical efficiency of LTSF-Transformers under L=96 and T=720 on the Electricity. MACs are the number of multiply-accumulate operations. The inference time averages 5 runs.


## Conclusion and Future Work

Conclusion. This work questions the effectiveness of emerging favored Transformer-based solutions for the longterm time series forecasting problem. We use an embarrassingly simple linear model LTSF-Linear as a DMS forecasting baseline to verify our claims. Note that our contributions do not come from proposing a linear model but rather from throwing out an important question, showing surprising comparisons, and demonstrating why LTSF-Transformers are not as effective as claimed in these works through various perspectives. We sincerely hope our comprehensive studies can benefit future work in this area. 

Future work. LTSF-Linear has a limited model capacity, and it merely serves a simple yet competitive baseline with strong interpretability for future research. Consequently, we believe there is great potential for new model designs, data processing, and benchmarks to tackle LTSF. 

## Acknowledgments

This work was supported in part by Alibaba Group Holding Ltd. under Grant No. TA2015393. We thank the anonymous reviewers for their constructive comments and suggestions. 

## References



Ariyo, A. A.; Adewumi, A. O.; and Ayo, C. K. 2014. Stock price prediction using the ARIMA model. In 2014 UKSim-AMSS 16th International Conference on Computer Modelling and Simulation, 106–112. IEEE. 





Bahdanau, D.; Cho, K.; and Bengio, Y. 2014. Neural Machine Translation by Jointly Learning to Align and Translate. arXiv: Computation and Language arXiv:1409.0473. 





Bai, S.; Kolter, J. Z.; and Koltun, V. 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271. 





Chevillon, G. 2007. Direct multi-step estimation and forecasting. Journal ofEconomic Surveys, 21(4): 746–785. 





Cirstea, R.-G.; Guo, C.; Yang, B.; Kieu, T.; Dong, X.; and Pan, S. 2022. Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting–Full Version. arXiv preprint arXiv:2204.13767. 





Cleveland, R. B. 1990. STL : A Seasonal-Trend Decomposition Procedure Based on Loess. Journal ofOffice Statistics. 





Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. 





Dong, L.; Xu, S.; and Xu, B. 2018. Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5884– 5888. IEEE. 





Friedman, J. H. 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189–1232. 





Hamilton, J. D. 2020. Time series analysis. Princeton university press. 





Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2017. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. international acm sigir conference on research and development in information retrieval. 





Li, S.; Jin, X.; Xuan, Y.; Zhou, X.; Chen, W.; Wang, Y.-X.; and Yan, X. 2019. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. Advances in Neural Information Processing Systems, 32. 





Liu, M.; Zeng, A.; Chen, M.; Xu, Z.; Lai, Q.; Ma, L.; and Xu, Q. 2022. SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction. Thirty-sixth Conference on Neural Information Processing Systems. 





Liu, S.; Yu, H.; Liao, C.; Li, J.; Lin, W.; Liu, A. X.; and Dustdar, S. 2021a. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations. 





Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. 2021b. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10012–10022. 





Salinas, D.; Flunkert, V.; and Gasthaus, J. 2017. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. International Journal ofForecasting. 





Taieb, S. B.; Hyndman, R. J.; et al. 2012. Recursive and direct multi-step forecasting: the best of both worlds, volume 19. Citeseer. 





Taylor, S. J.; and Letham, B. 2017. Forecasting at Scale. PeerJ Prepr. 





Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30. 





Wen, Q.; Zhou, T.; Zhang, C.; Chen, W.; Ma, Z.; Yan, J.; and Sun, L. 2022. Transformers in Time Series: A Survey. arXiv preprint arXiv:2202.07125. 





Xu, J.; Wang, J.; Long, M.; et al. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34. 





Zhou, H.; Zhang, S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.; and Zhang, W. 2021. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. In The Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Virtual Conference, volume 35, 11106–11115. AAAI Press. 





Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R. 2022. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning. 

