# Unified Training of Universal Time Series Forecasting Transformers

Gerald Woo <sup>1</sup> <sup>2</sup> Chenghao Liu <sup>1</sup> Akshat Kumar <sup>2</sup> Caiming Xiong <sup>1</sup> Silvio Savarese <sup>1</sup> Doyen Sahoo <sup>1</sup> 

## Abstract

Deep learning for time series forecasting has tradi tionally operated within a one-model-per-dataset framework, limiting its potential to leverage the game-changing impact of large pre-trained models. The concept of universal forecasting, emerging from pre-training on a vast collection of time series datasets, envisions a single Large Time Series Model capable of addressing diverse downstream forecasting tasks. However, constructing such a model poses unique challenges specific to time series data: i) cross-frequency learning, ii) accommodating an arbitrary number of variates for multivariate time series, and iii) addressing the varying distributional properties inherent in large-scale data. To address these challenges, we present novel enhancements to the conventional time series Transformer architecture, resulting in our proposed Masked EncOder-based UnIveRsAl TIme Series Forecasting Transformer (MOIRAI). Trained on our newly introduced Large-scale Open Time Series Archive (LOTSA) featuring over 27B observations across nine domains, MOIRAI achieves competitive or superior performance as a zero-shot forecaster when compared to full-shot models. Code, data, and model weights can be found at https://github. com/SalesforceAIResearch/uni2ts. 

## 1. Introduction

In the era of foundation models (FMs) (Bommasani et al., 2021), the landscape of deep learning for time series forecasting is experiencing a revolution. In contrast to FMs capable of tackling a multitude of downstream tasks, the current deep forecasting paradigm, involving training a model on a single dataset with a fixed context and prediction length, appears increasingly antiquated, lacking the capacity to generalize or adapt to diverse scenarios or datasets. Given the unreasonable effectiveness of large pre-trained models in improving performance and data efficiency via transfer learning in modalities like vision and language (Dosovitskiy et al., 2020; Brown et al., 2020), we are starting to see a push to transition away from the existing paradigm, towards a universalforecasting paradigm (see Figure 1), where a single large pre-trained model is able to handle any time series forecasting problem. However, the road to building a universal time series forecasting model is mired with challenges. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/79e2057e3f98f67dbe64e5612e74a1634caeacc9d72487fea31b54e49d121c3d.jpg)



Figure 1. A universalforecaster is a large pre-trained model capable of handling any time series forecasting problem. It is trained on a large-scale time series dataset spanning multiple domains. Compared to the existing paradigm, universal forecasting faces the three key issues of i) multiple frequencies, ii) any-variate forecasting, and iii) varying distributions.


Unlike the modalities of vision and language which have the unified formats of images and text respectively, time series data is highly heterogeneous. Firstly, the frequency (e.g. minutely, hourly, daily sampling rates) of time series plays an important role in determining the patterns present in the time series. Cross-frequency learning has been shown to be a challenging task due to negative interference (Van Ness et al., 2023), with existing work simply avoiding this problem for multi-frequency datasets by learning one model per frequency (Oreshkin et al., 2020). Secondly, time series data are heterogeneous in terms of dimensionality, whereby multivariate time series can have different number of variates. Furthermore, each variate measures a semantically different quantity across datasets. While considering each variate of a multivariate time series independently (Nie et al., 2023; Ekambaram et al., 2023) can sidestep this problem, we expect a universal model to be sufficiently flexible to consider multivariate interactions and take into account exogenous covariates. Thirdly, probabilistic forecasting is a critical feature often required by practitioners, yet, different datasets have differing support and distributional properties – for example, using a symmetric distribution (e.g. Normal, Student-T) as the predictive distribution is not suitable for positive time series – making standard approaches of pre-defining a simple parametric distribution (Salinas et al., 2020) to be insufficiently flexible to capture a wide variety of datasets. Lastly, a large pre-trained model capable of universal forecasting requires a large-scale dataset from diverse domains. Existing time series datasets are insufficiently large to support the training of such models. 


Table 1. Comparison between pre-trained forecasting models. Further discussion on the notion of a “flexible distribution” can be found in Appendix B.3.


<table><tr><td></td><td>Any-variate (Zero-shot)</td><td>Probabilistic Forecasting</td><td>Flexible Distribution</td><td>Pre-training Data (Size)</td><td>Open-source</td></tr><tr><td>MOIRAI</td><td>√</td><td>√</td><td>√</td><td>LOTSA (&gt; 27B)</td><td>√</td></tr><tr><td>TimeGPT-1</td><td>√</td><td>√</td><td>✗</td><td>Unknown (100B)</td><td>✗</td></tr><tr><td>ForecastPFN</td><td>✗</td><td>✗</td><td>-</td><td>Synthetic Data (60M)</td><td>√</td></tr><tr><td>Lag-Llama</td><td>✗</td><td>√</td><td>✗</td><td>Monash (&lt; 1B)</td><td>√</td></tr><tr><td>TimesFM</td><td>✗</td><td>✗</td><td>-</td><td>Wiki + Trends + Others (&gt; 100B)</td><td>√</td></tr><tr><td>TTM</td><td>✗</td><td>✗</td><td>-</td><td>Monash (&lt; 1B)</td><td>√</td></tr><tr><td>LLMTime</td><td>✗</td><td>√</td><td>√</td><td>Web-scale Text</td><td>√</td></tr></table>

Starting from a masked encoder architecture which has been shown to be a strong candidate architecture for scaling up pre-trained time series forecasting models (Woo et al., 2023), we alleviate the above issues by introducing novel modifications which allows the architecture to handle the heterogeneity of arbitrary time series data. Firstly, we propose to learn multiple input and output projection layers to handle the differing patterns from time series of varying frequencies. Using patch-based projections with larger patch sizes for high-frequency data and vice versa, projection layers are specialized to learn the patterns of that frequency. Secondly, we address the problem of varying dimensionality with our proposed Any-variate Attention, which simultaneously considers both time and variate axes as a single sequence, leveraging Rotary Position Embeddings (RoPE) (Su et al., 2024), and learned binary attention biases (Yang et al., 2022b) to encode time and variate axes respectively. Importantly, Any-variate Attention allows the model to take as input arbitrary number of variates. Thirdly, we overcome the issue of requiring flexible predictive distributions with a mixture of parametric distributions. Furthermore, optimizing the negative log-likelihood of a flexible distribution has the added benefit of being competitive with target metric optimization (Awasthi et al., 2022), a powerful feature for pre-training universal forecasters, given that it can be evaluated with any target metric subsequently. 

To power the training of our Large Time Series Model (LTM), we introduce the Large-scale Open Time Series Archive (LOTSA), the largest collection of open time series datasets with 27B observations across nine domains. We optimize the negative log-likelihood of the mixture distribution, and randomly sample context and prediction lengths during training, allowing for flexible downstream usage of the pre-trained model. We train our proposed method, Masked EncOder-based UnIveRsAl TIme Series Forecasting Transformer $( \mathbf { M O I R A I } ^ { 1 } )$ , in three sizes – MOIRAI<sub>Small</sub>, $\mathbf { M O I R A I _ { B a s e } } .$ , and $\mathbf { M O I R A I _ { L a r g e } } ,$ with 14m, 91m, and 311m parameters respectively. We perform experimental evaluations on both in and out-of-distribution settings, and show that MOIRAI consistently achieves competitive or superior performance compared to state-of-the-art full-shot baselines. Our contributions are summarized as follows: 

1. We introduce a novel Transformer architecture to support the requirements of universal forecasting. Crucially, the components we propose extend beyond masked encoders and are versatile, applicable to a broad range of Transformer variants. 

2. We introduce LOTSA, a new large-scale collection of open time series datasets to empower pre-training of LTMs. LOTSA, the model weights, and our library for unified training of universal time series models, UNI<sup>2</sup>TS, will be fully open sourced. 

3. Trained on LOTSA data, MOIRAI achieves competitive or superior performance as a zero-shot forecaster when compared to full-shot models. 

## 2. Related Work

Pre-training for Zero-shot Forecasting Table 1 provides a summary of the key differences between recent pre-trained models with zero-shot forecasting capabilities, which is a recently emerging field. TimeGPT-1 (Garza & Mergenthaler-Canseco, 2023) first presented a closed-source model, offering zero-shot forecasting capabilities as well as supporting fine-tuning through an API, currently only available to their beta users. ForecastPFN (Dooley et al., 2023) proposes to pre-train on synthetic time series, which can be subsequently be leveraged as a zero-shot forecaster, albeit specialized for data or time limited settings. Lag-llama (Rasul et al., 2023) works towards a foundation model for time series forecasting, leveraging the LLaMA (Touvron et al., 2023) architecture design with lagged time series features, and also presents neural scaling laws for time series forecasting. TimesFM (Das et al., 2023b) is a patch-based decoder-only foundation model for time series forecasting, introducing a larger output patch size for faster decoding. They collected a massive amount of data from Google Trends and Wiki pageviews to pre-train their model in combination with opendata. Tiny Time Mixers (TTMs) (Ekambaram et al., 2024) is a concurrent work leveraging lightweight mixer-style architecture. They perform data augmentation by downsampling high-frequency time series, and support multivariate downstream tasks by fine-tuning an exogenous mixer. leverage Large Language Models (LLMs), pre-trained on web-scale text data, have been leveraged for zero-shot forecasting. Specifically, LLMTime (Gruver et al., 2023) treats time series as strings, applying careful pre-processing based on the specific LLMs’ tokenizer, showing that LLMs have the inherent capability to perform zero-shot forecasting. 

Pre-train + Fine-tune for Time Series Forecasting Pre-training with subsequent fine-tuning on downstream forecasting tasks has predated the recent zero-shot forecasting efforts. Denoising autoencoders (Zerveas et al., 2021) and contrastive learning (Yue et al., 2022; Woo et al., 2022) have been shown to be effective pretext tasks for time series forecasting, but have largely been applied to the existing paradigm of pre-training and fine-tuning on the same dataset, without exploring their generalization capabilities. More recently, Dong et al. (2023) explored combining both reconstruction and contrastive based pre-training approaches, and performed initial explorations into cross-dataset transfer. The topic has been well explored, and we refer readers to more comprehensive surveys (Zhang et al., 2023; Ma et al., 2023). “Reprogramming” is a recent direction which involves fine-tuning the model weights of an LLM which has been pre-trained on text data, for downstream tasks for other modalities. Zhou et al. (2023); Jin et al. (2023) introduce modules and fine-tuning methods to adapt LLMs for time series tasks including forecasting. Liu et al. (2024) has explored leveraging pre-trained LLMs on the cross-dataset setting. 

## 3. Method

Problem Formulation Consider a dataset of N time series $\textit { \textbf { D } } = \{ ( \boldsymbol { Y } ^ { ( i ) } , \boldsymbol { Z } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , where $\begin{array} { r l } { \pmb { Y } ^ { ( i ) } } & { { } = } \end{array}$ $( \pmb { y } _ { 1 } ^ { ( i ) } , \pmb { y } _ { 2 } ^ { ( i ) } , \dots , \pmb { y } _ { T _ { i } } ^ { ( i ) } ) \in \mathbb { R } ^ { d _ { \boldsymbol { y } _ { i } } \times T _ { i } }$ is a target time series of $d _ { y _ { i } }$ variates and $T _ { i } ^ { \phantom { \dagger } }$ time steps. Each time series is associated with a set of covariates ${ \cal Z } ^ { ( i ) } \ = \ ( z _ { 1 } ^ { ( i ) } , z _ { 2 } ^ { ( i ) } , \ldots , z _ { T _ { i } } ^ { ( i ) } ) \ \in$ $\mathbb { R } ^ { d _ { z _ { i } } \times T _ { i } }$ . The goal is to forecast the predictive distribution $p ( Y _ { t : t + h } | \phi )$ by predicting distribution parameters $\phi$ via a learned model $f _ { \theta } ~ : ~ ( { \cal Y } _ { t - l : t } , { \cal Z } _ { t - l : t + h } ) ~ \mapsto ~ \hat { \phi }$ which maximizes the log-likelihood: 

$$
\max_{\boldsymbol{\theta}}\quad \mathop{\mathbb{E}}_{\substack{(\mathbf{Y},\mathbf{Z})\sim p(\mathcal{D})\\ (\mathrm{t},\mathrm{l},\mathrm{h})\sim p(\mathcal{T}|\mathcal{D})}}\log p(\mathbf{Y}_{\mathrm{t:t + h}}|\hat{\boldsymbol{\Phi}}),
$$

$$
\text { s.t. } \hat {\boldsymbol {\phi}} = f _ {\boldsymbol {\theta}} (\mathbf {Y} _ {t - l: t}, \mathbf {Z} _ {t - l: t + h}),\tag{1}
$$

where $p ( \mathcal { D } )$ is the data distribution which samples for a time series, $( Y , z )$ , and $p ( \mathcal T | \mathcal D )$ is the task distribution which defines the lookback window, $Y _ { t - l : t } = ( y _ { t - l } , \dots , y _ { t - 1 } )$ with context length l and forecast horizon, $Y _ { t : t + h } = ( { \pmb y } _ { t } , \dots , { \pmb y } _ { t + h - 1 } )$ with prediction length h. 

## 3.1. Architecture

Illustrated in Figure 2, MOIRAI follows a (non-overlapping) patch-based approach to modeling time series with a masked encoder architecture. One of our proposed modifications to extend the architecture to the any-variate setting is to “flatten” multivariate time series, considering all variates as a single sequence. Patches are subsequently projected into vector representations via a multi patch size input projection layer. The [mask] signifies a learnable embedding which replaces patches falling within the forecast horizon. The output tokens are then decoded via the multi patch size output projection into the parameters of the mixture distribution. While not visualized, (non-learnable) instance normalization (Kim et al., 2022) is applied to inputs/outputs, aligning with the current standard practice for deep forecasting models. 

The core Transformer module is an encoder-only Transformer architecture, leveraging various improvements as proposed by recent state-of-the-art LLM architectures. We use pre-normalization (Xiong et al., 2020) and replace all LayerNorms with RMSNorm (Zhang & Sennrich, 2019), and also apply query-key normalization (Henry et al., 2020). The non-linearity in FFN layers are replaced with SwiGLU (Shazeer, 2020), adjusting the hidden dimension to have equal number of parameters as the original FFN layer. We omit biases in all layers of the Transformer module. 

## 3.1.1. MULTI PATCH SIZE PROJECTION LAYERS

In the context of universal forecasting, a single model should possess the capability to handle time series spanning a wide range of frequencies. Existing patch-based architectures rely on a single patch size hyperparameter, a legacy feature from the prevailing one-model-per-dataset paradigm. Instead, we aim for a more flexible strategy: opting for a larger patch size to handle high-frequency data, thereby lower the burden of the quadratic computation cost of attention while maintaining a long context length. Simultaneously, we advocate for a smaller patch size for low-frequency data to transfer computation to the Transformer layers, rather than relying solely on simple linear embedding layers. To implement this approach, we propose learning multiple input and output embedding layers, each associated with varying patch sizes. The selection of the appropriate patch size for a given time series frequency relies on pre-defined settings (see Appendix B.1). Note that we only learn one set of projection weights per patch size, which is shared amongst frequencies if there is an overlap based on the settings. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/43395d29c311ef96cd330d2a2681c4692c08f1975923133373ac776312e3983c.jpg)



Figure 2. Overall architecture of MOIRAI. Visualized is a 3-variate time series, where variates 0 and 1 are target variables $( \mathrm { i . e . }$ to be forecasted, and variate 2 is a dynamic covariate (values in forecast horizon known). Based on a patch size of 64, each variate is patchified into 3 tokens. The patch embeddings along with sequence and variate id are fed into the Transformer. The shaded patches represent the forecast horizon to be forecasted, whose corresponding output representations are mapped into the mixture distribution parameters.


## 3.1.2. ANY-VARIATE ATTENTION

Universal forecasters must be equipped to handle arbitrary multivariate time series. Existing time series Transformers often rely on an independent variate assumption or are limited to a single dimensionality due to embedding layers mapping $\mathbb { R } ^ { d _ { y } }  \mathbb { R } ^ { d _ { h } }$ , where $\mathbb { R } ^ { \boldsymbol { \dot { d _ { h } } } }$ is the hidden dimension. We overcome this limitation as shown in Figure 2, by flattening a multivariate time series to consider all variates as a single sequence. This introduces a new requirement of having variate encodings to enable the model to disambiguate between different variates in the sequence. Furthermore, we need to ensure that permutation equivariance w.r.t. variate ordering, and permutation invariance w.r.t. variate indices are respected. Conventional approaches like sinusoidal or learned embeddings do not meet these requirements, and are unable to handle an arbitrary number of variates. To address this, we propose Any-variate Attention, leveraging binary attention biases to encode variate indices. 

Dropping layer and attention head indices, and scaling factor for brevity, the attention score between the (i, m)-th query where i represents the time index and m represents the variate index, and the (j, n)-th key, $A _ { i j , m n } \in$ R, is given by: 

$$
E _ {i j, m n} = (\pmb {W} ^ {Q} \pmb {x} _ {i, m}) ^ {T} \pmb {R} _ {i - j} (\pmb {W} ^ {K} \pmb {x} _ {j, n})
$$

$$
+ u ^ {(1)} * \mathbb {1} _ {\{m = n \}} + u ^ {(2)} * \mathbb {1} _ {\{m \neq n \}},\tag{2}
$$

$$
A _ {i j, m n} = \frac {\exp \{E _ {i j , m n} \}}{\sum_ {k , o} \exp \{E _ {i k , m o} \}},\tag{3}
$$

where $\begin{array} { r } { W ^ { Q } \pmb { x } _ { i , m } , \pmb { W } ^ { K } \pmb { x } _ { j , n } \in \mathbb { R } ^ { d _ { h } } } \end{array}$ are the respective query and key vectors, $\pmb { R } _ { i - j } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ is the rotary matrix (Su et al., 2024), $u ^ { ( 1 ) } , u ^ { ( 2 ) } \in$ R are learnable scalars for each head in each layer, and $\mathbb { 1 } _ { \{ \mathrm { c o n d } \} } = \{ { 1 , \mathrm { i f } \mathrm { c o n d } } _ { 0 , \mathrm { o t h e r w i s e } }$ is the indicator function. The binary attention bias component allows for disambiguation between variates via attention scores, fulfills the criteria of permutation equivariance/invariance w.r.t. variate ordering/indices, and can extend to arbitrary number of variates. 

## 3.1.3. MIXTURE DISTRIBUTION

To achieve the goal of having a flexible distribution, yet ensuring that operations of sampling and evaluating the loss function remains simple, we propose to use a mixture of parametric distributions. A mixture distribution of c components has p.d.f.: 

$$
p (\mathbf {Y} _ {t: t + h} | \hat {\boldsymbol {\phi}}) = \sum_ {i = 1} ^ {c} w _ {i} p _ {i} (\mathbf {Y} _ {t: t + h} | \hat {\boldsymbol {\phi}} _ {i}),\tag{4}
$$

where $\hat { \phi } ~ = ~ \{ w _ { 1 } , \hat { \phi } _ { 1 } , . . . , w _ { c } , \hat { \phi } _ { c } \}$ , and $p _ { i }$ is the i-th component’s p.d.f. While the choice of mixture components is flexible and implementing any combination of parametric distributions is straightforward, we specifically propose to use the following mixture components: i) a Student’s t-distribution which has shown to be a robust option for general time series, ii) a negative binomial distribution for positive count data, iii) a log-normal distribution to model right-skewed data commonly across economic and and natural phenomena, and iv) a low variance normal distribution for high confidence predictions. Further details can be found in Appendix B.2. 


Table 2. Key statistics of LOTSA by domain.


<table><tr><td></td><td>Energy</td><td>Transport</td><td>Climate</td><td>CloudOps</td><td>Web</td><td>Sales</td><td>Nature</td><td>Econ/Fin</td><td>Healthcare</td></tr><tr><td># Datasets</td><td>30</td><td>23</td><td>6</td><td>3</td><td>3</td><td>6</td><td>5</td><td>23</td><td>6</td></tr><tr><td># Obs.</td><td>16,358,600,896</td><td>4,900,453,419</td><td>4,188,011,890</td><td>1,518,268,292</td><td>428,082,373</td><td>197,984,339</td><td>28,547,647</td><td>24,919,596</td><td>1,594,281</td></tr><tr><td>%</td><td>59.17%</td><td>17.73%</td><td>15.15%</td><td>5.49%</td><td>1.55%</td><td>0.72%</td><td>0.09%</td><td>0.10%</td><td>0.01%</td></tr></table>


Table 3. Key statistics of LOTSA by frequency.


<table><tr><td></td><td>Yearly</td><td>Quarterly</td><td>Monthly</td><td>Weekly</td><td>Daily</td><td>(Multi) Hourly</td><td>(Multi) Minute-level</td><td>(Multi) Second-level</td></tr><tr><td># Datasets</td><td>4</td><td>5</td><td>10</td><td>7</td><td>21</td><td>31</td><td>25</td><td>2</td></tr><tr><td># Obs.</td><td>873,297</td><td>2,312,027</td><td>11,040,648</td><td>18,481,871</td><td>709,017,118</td><td>19,875,993,973</td><td>7,013,949,430</td><td>14,794,369</td></tr><tr><td>%</td><td>0.003%</td><td>0.008%</td><td>0.040%</td><td>0.067%</td><td>2.565%</td><td>71.893%</td><td>25.370%</td><td>0.054%</td></tr></table>

## 3.2. Unified Training

## 3.2.1. LOTSA DATA

Existing work has predominantly relied on three primary sources of data – the Monash Time Series Forecasting Archive (Godahewa et al., 2021), datasets provided by the GluonTS library (Alexandrov et al., 2020), and datasets from the popular long sequence forecasting benchmark (Lai et al., 2018; Wu et al., 2021). While Monash and GluonTS comprise of datasets from diverse domains, they are constrained in size, with approximately 1B observations combined. In comparison, LLMs are trained on trillions of tokens. Das et al. (2023b) builds a private dataset mainly based on Google Trends and Wiki pageviews, but lacks diversity in terms of the domains these time series originate from. 

The effectiveness of FMs heavily stem from large-scale pretraining data. Given that existing data sources fall short of supporting such a paradigm, attempting to train an LTM on them may result in misleading conclusions. Thus, we tackle this issue head-on by building a large-scale archive of open time series datasets by collating publicly available sources of time series datasets. This effort aims to cover a broad spectrum of domains, consolidating datasets from diverse sources with varying formats. We design a unified storage format using Arrow (Richardson et al., 2023) which is ready for deep learning pipelines. The resulting collection, LOTSA, spans nine domains, with a total of 27, 646, 462, 733 observations, with key statistics in Tables 2 and 3, and in-depth details in Appendix A. 

## 3.2.2. PRE-TRAINING

As introduced in Equation (1), our pre-training task is formulated to optimize the mixture distribution log-likelihood. The design of both the data distribution and task distribution are two critical aspects of the pre-training pipeline. This design imparts versatile capabilities to our LTM, enabling it to adapt to a range of downstream tasks. This flexibility stands in contrast to the prevailing deep forecasting paradigm, where models are typically specialized for specific datasets and settings. 

Data Distribution The data distribution, $( { \bf Y } , { \bf Z } ) \sim p ( { \mathcal D } )$ defines how time series are sampled from the dataset. Trained on LOTSA, which is a dataset of datasets, we introduce the notion of sub-datasets, by decomposing the data distribution into a sub-dataset distribution, and a time series distribution conditioned on a sub-dataset, $p ( \mathcal { D } ) =$ $p ( \mathbf { Y } , \mathbf { Z } | \mathbf { D } ) p ( \mathbf { D } )$ . Thus, we first sample a sub-dataset from $p ( \mathbf { D } )$ , and given that sub-dataset, we sample a time series. For K sub-datasets, where $\scriptstyle D _ { k }$ represents the set of indices of time series belonging to sub-dataset $k ,$ the structure of $\begin{array} { r } { p ( { \pmb Y } ^ { ( i ) } , { \pmb Z } ^ { ( i ) } | { \pmb D } _ { k } ) = \frac { T _ { i } * \mathbb { 1 } _ { \{ i \in { \pmb D } _ { k } \} } } { \sum _ { j \in { \pmb D } _ { k } } T _ { j } } } \end{array}$ , proportionate to the number of observations, is straightforward. 

However, due to data imbalance across domains and frequency, we avoid sampling sub-datasets proportionately, and instead cap the contribution of each sub-dataset at $\epsilon ~ = ~ 0 . 0 0 1$ , before re-normalizing: $\begin{array} { r } { p ( D _ { k } ) ~ = ~ \frac { \omega _ { k } } { \sum _ { i = 1 } ^ { K } \omega _ { i } } } \end{array}$ where $\begin{array} { r } { \omega _ { k } = \operatorname* { m i n } ( \frac { | D _ { k } | } { \sum _ { i } ^ { K } | D _ { i } | } , \epsilon ) } \end{array}$ , and $\begin{array} { r } { | D _ { k } | = \sum _ { i \in D _ { k } } T _ { i } } \end{array}$ 

Task Distribution Different from the existing deep forecasting paradigm, we aim to train a model with forecasting capabilities over varying context and prediction lengths. Rather than defining a fixed context and prediction length, we sample from a task distribution, $( \mathbf { t } , \mathbf { l } , \mathbf { h } ) \ \sim \ p ( \mathcal { T } | \mathcal { D } )$ which defines the lookback window and forecasting horizon, given a time series. In practice, rather than sampling t, l, h, given a time series, we crop a uniformly sampled window, whose length is uniformly sampled from a range. This range is defined by a minimum sequence length per variate of $^ { 2 , }$ and a total maximum sequence length of 512. The window is then split into lookback and horizon segments, where the prediction length is uniformly sampled as a proportion (within the range [0.15, 0.5]) of the window. We further augment training by i) uniformly subsampling multivariate time series in the variate dimension, and ii) constructing multivariate time series from sub-datasets with univariate time series, by randomly concatenating them. The number of variates is sampled from a beta-binomial distribution with parameters $n = 1 2 8 , a = 2 , b = 5$ which supports a maximum of 128 variates, with mean ≈ 37 for efficiency. 


Table 4. Details of MOIRAI model sizes.


<table><tr><td></td><td>Layers</td><td><eq>d_{model}</eq></td><td><eq>d_{ff}</eq></td><td>Heads</td><td><eq>d_{kv}</eq></td><td>Params</td></tr><tr><td><eq>MOIRAI_{Small}</eq></td><td>6</td><td>384</td><td>1536</td><td>6</td><td>64</td><td>14m</td></tr><tr><td><eq>MOIRAI_{Base}</eq></td><td>12</td><td>768</td><td>3072</td><td>12</td><td>64</td><td>91m</td></tr><tr><td><eq>MOIRAI_{Large}</eq></td><td>24</td><td>1024</td><td>4096</td><td>16</td><td>64</td><td>311m</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/427a4238437194cf15a6097d5d486ef9c7751bc7d5556ab341620fe47bd6f648.jpg)



Figure 3. Aggregate results of the Monash Time Series Forecasting Benchmark. The normalized MAE is reported, which normalizes the MAE of each dataset by the naive forecast’s MAE, and aggregated by taking the geometric mean across datasets.


Training We train MOIRAI in three sizes – small, base, and large, with key parameter details listed in Table 4. The small model is trained for 100, 000 steps, while base and large models are trained for 1, 000, 000 steps with a batch size of 256. For optimization, we use the AdamW optimizer with the following hyperparameters, lr = 1e-3, weight decay = 1e-1, $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8$ . We also apply a learning rate scheduler with linear warmup for the first 10, 000 steps, and cosine annealing thereafter. Models are trained on NVIDIA A100-40G GPUs with TF32 precision. We implement sequence packing (Raffel et al., 2020) to avoid large amounts of padding due to the disparity of sequence lengths in the new setting with varying context, prediction, and variate lengths, thereby increasing the effective batch size. 

## 4. Experiments

## 4.1. In-distribution Forecasting

We first perform an in-distribution evaluation using the Monash benchmark, which aim to measure generalization capability across diverse domains. Described in Appendix A, LOTSA includes the Monash Time Series Forecasting Archive as a source of data. For a large portion of these datasets, we only include the train set, holding out the test set which we now use for in-distribution evaluation. In this evaluation, we consider a standard setting with a context length of 1000, and a patch size of 32 for all frequencies, except for quarterly data with a patch size of 8. Figure 3 summarizes the results based on the normalized mean absolute error (MAE), in comparison with the baselines presented in the Monash benchmark. It is worth noting that each baseline in the Monash benchmark is typically trained individually per dataset or per time series within a dataset. In contrast, MOIRAI stands out by being a single model evaluated across various datasets. Full results as well as a comparison with LLMTime (Gruver et al., 2023) can be found in Appendix D.1. 

We observe that MOIRAI outperforms all baselines from the Monash benchmark regardless of model size, displaying the strong in-distribution and cross-domain capabilities arising from our unified training methodology. We highlight that each instance of MOIRAI is a single model evaluated across datasets, compared to baselines for which one model is trained per dataset. Further analysis on computational costs can be found in Appendix D.4. 

## 4.2. Out-of-distribution / Zero-shot Forecasting

Next, we perform an out-of-distribution evaluation on unseen target datasets. Here, MOIRAI is a zero-shot forecaster compared with state-of-the-art full-shot baselines which have been trained on the individual target datasets. While the ideal scenario would be to include other universal forecasters, this proves to be a challenging task. As a nascent field, most universal forecasters currently do not yet have open weights avaiable for evaluation. Furthermore, the problem of comparing zero-shot methods is exacerbated by not having a standard held-out test split, making it challenging to collate a set of datasets which all the models have not been trained on. Thus, we establish the strong zero-shot capabilities of MOIRAI by displaying competitive or stronger results compared with SOTA full-shot methods – datasets used in the following have not been included in LOTSA. 

Probabilistic Forecasting We evaluate on six datasets across energy, transport, climate, and sales domains, following a rolling evaluation setup with stride equal to prediction length. Prediction lengths and number of rolling evaluations are defined for each dataset based on frequency. We report the Continuous Ranked Probability Score (CRPS) and Mean Scaled Interval Score (MSIS) metrics (definitions in Appendix C), comparing against four full-shot baselines – DeepAR (Salinas et al., 2020), PatchTST (Nie et al., 2023), and TiDE (Das et al., 2023a) with Student’s t-distribution prediction heads, and TFT based on quantile prediction (Lim et al., 2021), all implemented with the GluonTS library (Alexandrov et al., 2020), as well as simple baselines AutoARIMA (Garza et al., 2022) and Seasonal Naive (Hyndman & Athanasopoulos, 2018). For each dataset and baseline, we perform hyperparameter tuning on a validation CRPS, and report results averaged over five training runs with different seeds. For MOIRAI, we perform inference time tuning, selecting context length from {1000, 2000, 3000, 4000, 5000} and patch sizes based on frequency, on the validation CRPS. Full details of the evaluation setting can be found in Appendix C. 


Table 5. Probabilistic forecasting results. Best results are highlighted in bold, and second best results are underlined. Baseline results are aggregated over five training runs with different seeds, reporting the mean and standard deviation.


<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Zero-shot</td><td colspan="4">Full-shot</td><td colspan="2">Baseline</td></tr><tr><td><eq>MOIRAI_{Small}</eq></td><td><eq>MOIRAI_{Base}</eq></td><td><eq>MOIRAI_{Large}</eq></td><td>PatchTST</td><td>TiDE</td><td>TFT</td><td>DeepAR</td><td>AutoARIMA</td><td>Seasonal Naive</td></tr><tr><td rowspan="2">Electricity</td><td>CRPS</td><td>0.072</td><td>0.055</td><td>0.050</td><td>0.052±0.00</td><td>0.048±0.00</td><td>0.050±0.00</td><td>0.065±0.01</td><td>0.327</td><td>0.070</td></tr><tr><td>MSIS</td><td>7.999</td><td>6.172</td><td>5.875</td><td>5.744±0.12</td><td>5.672±0.08</td><td>6.278±0.24</td><td>6.893±0.82</td><td>29.412</td><td>35.251</td></tr><tr><td rowspan="2">Solar</td><td>CRPS</td><td>0.471</td><td>0.419</td><td>0.406</td><td>0.518±0.09</td><td>0.420±0.00</td><td>0.446±0.03</td><td>0.431±0.01</td><td>1.055</td><td>0.512</td></tr><tr><td>MSIS</td><td>8.425</td><td>7.011</td><td>6.250</td><td>8.447±1.59</td><td>13.754±0.32</td><td>8.057±3.51</td><td>11.181±0.67</td><td>25.849</td><td>48.130</td></tr><tr><td rowspan="2">Walmart</td><td>CRPS</td><td>0.103</td><td>0.093</td><td>0.098</td><td>0.082±0.01</td><td>0.077±0.00</td><td>0.087±0.00</td><td>0.121±0.00</td><td>0.124</td><td>0.151</td></tr><tr><td>MSIS</td><td>9.371</td><td>8.421</td><td>8.520</td><td>6.005±0.21</td><td>6.258±0.12</td><td>8.718±0.10</td><td>12.502±0.03</td><td>9.888</td><td>49.458</td></tr><tr><td rowspan="2">Weather</td><td>CRPS</td><td>0.049</td><td>0.041</td><td>0.051</td><td>0.059±0.01</td><td>0.054±0.00</td><td>0.043±0.00</td><td>0.132±0.11</td><td>0.252</td><td>0.068</td></tr><tr><td>MSIS</td><td>5.236</td><td>5.136</td><td>4.962</td><td>7.759±0.49</td><td>8.095±1.74</td><td>7.791±0.44</td><td>21.651±17.34</td><td>19.805</td><td>31.293</td></tr><tr><td rowspan="2">Istanbul Traffic</td><td>CRPS</td><td>0.173</td><td>0.116</td><td>0.112</td><td>0.112±0.00</td><td>0.110±0.01</td><td>0.110±0.01</td><td>0.108±0.00</td><td>0.589</td><td>0.257</td></tr><tr><td>MSIS</td><td>5.937</td><td>4.461</td><td>4.277</td><td>3.813±0.09</td><td>4.752±0.17</td><td>4.057±0.44</td><td>4.094±0.31</td><td>16.317</td><td>45.473</td></tr><tr><td rowspan="2">Turkey Power</td><td>CRPS</td><td>0.048</td><td>0.040</td><td>0.036</td><td>0.054±0.01</td><td>0.046±0.01</td><td>0.039±0.00</td><td>0.066±0.02</td><td>0.116</td><td>0.085</td></tr><tr><td>MSIS</td><td>7.127</td><td>6.766</td><td>6.341</td><td>8.978±0.51</td><td>8.579±0.52</td><td>7.943±0.31</td><td>13.520±1.17</td><td>14.863</td><td>36.256</td></tr></table>


Table 6. Long sequence forecasting results. Results are averaged across prediction lengths {96, 192, 336, 720}. Best results are highlighted in bold, and second best results are underlined. Full-shot results are obtained from Liu et al. (2023b).


<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Zero-shot</td><td colspan="8">Full-shot</td></tr><tr><td><eq>MOIRAI_{Small}</eq></td><td><eq>MOIRAI_{Base}</eq></td><td><eq>MOIRAI_{Large}</eq></td><td>iTransformer</td><td>TimesNet</td><td>PatchTST</td><td>Crossformer</td><td>TiDE</td><td>DLinear</td><td>SCINet</td><td>FEDformer</td></tr><tr><td rowspan="2">ETTh1</td><td>MSE</td><td>0.400</td><td>0.434</td><td>0.510</td><td>0.454</td><td>0.458</td><td>0.469</td><td>0.529</td><td>0.541</td><td>0.456</td><td>0.747</td><td>0.44</td></tr><tr><td>MAE</td><td>0.424</td><td>0.438</td><td>0.469</td><td>0.448</td><td>0.450</td><td>0.455</td><td>0.522</td><td>0.507</td><td>0.452</td><td>0.647</td><td>0.46</td></tr><tr><td rowspan="2">ETTh2</td><td>MSE</td><td>0.341</td><td>0.345</td><td>0.354</td><td>0.383</td><td>0.414</td><td>0.387</td><td>0.942</td><td>0.611</td><td>0.559</td><td>0.954</td><td>0.437</td></tr><tr><td>MAE</td><td>0.379</td><td>0.382</td><td>0.376</td><td>0.407</td><td>0.497</td><td>0.407</td><td>0.684</td><td>0.550</td><td>0.515</td><td>0.723</td><td>0.449</td></tr><tr><td rowspan="2">ETTm1</td><td>MSE</td><td>0.448</td><td>0.381</td><td>0.390</td><td>0.407</td><td>0.400</td><td>0.387</td><td>0.513</td><td>0.419</td><td>0.403</td><td>0.486</td><td>0.448</td></tr><tr><td>MAE</td><td>0.409</td><td>0.388</td><td>0.389</td><td>0.410</td><td>0.406</td><td>0.400</td><td>0.495</td><td>0.419</td><td>0.407</td><td>0.481</td><td>0.452</td></tr><tr><td rowspan="2">ETTm2</td><td>MSE</td><td>0.300</td><td>0.272</td><td>0.276</td><td>0.288</td><td>0.291</td><td>0.281</td><td>0.757</td><td>0.358</td><td>0.35</td><td>0.571</td><td>0.305</td></tr><tr><td>MAE</td><td>0.341</td><td>0.321</td><td>0.320</td><td>0.332</td><td>0.333</td><td>0.326</td><td>0.611</td><td>0.404</td><td>0.401</td><td>0.537</td><td>0.349</td></tr><tr><td rowspan="2">Electricity</td><td>MSE</td><td>0.233</td><td>0.188</td><td>0.188</td><td>0.178</td><td>0.193</td><td>0.216</td><td>0.244</td><td>0.252</td><td>0.212</td><td>0.268</td><td>0.214</td></tr><tr><td>MAE</td><td>0.320</td><td>0.274</td><td>0.273</td><td>0.270</td><td>0.295</td><td>0.304</td><td>0.334</td><td>0.344</td><td>0.3</td><td>0.365</td><td>0.327</td></tr><tr><td rowspan="2">Weather</td><td>MSE</td><td>0.242</td><td>0.238</td><td>0.259</td><td>0.258</td><td>0.259</td><td>0.259</td><td>0.259</td><td>0.271</td><td>0.265</td><td>0.292</td><td>0.309</td></tr><tr><td>MAE</td><td>0.267</td><td>0.261</td><td>0.275</td><td>0.278</td><td>0.287</td><td>0.281</td><td>0.315</td><td>0.320</td><td>0.317</td><td>0.363</td><td>0.36</td></tr></table>

Table 5 reports the CRPS and MSIS, with full results including deterministic metrics in Appendix D.2. We observe that $\mathbf { M O I R A I _ { B a s e } }$ and $\mathbf { M O I R A I _ { L a r g e } }$ consistently achieve strong zero-shot performance, obtaining either best or second best results for all datasets except Walmart and Istanbul Traffic. Even for these datasets, performance is still close to the best performance, despite being a single zero-shot model compared to baselines which have been tuned and trained on the train sets. 

Long Sequence Forecasting We evaluate on a subset of the popular long sequence forecasting benchmark (Wu et al., 2021), omitting datasets which have datasets from the same source present in our pre-training data and cannot be considered zero-shot. We report the Mean Squared Error (MSE) and MAE, comparing against six state-of-the-art baselines, iTransformer (Liu et al., 2023b), TimesNet (Wu et al., 2023), PatchTST, Crossformer (Zhang & Yan, 2023), TiDE, DLinear (Zeng et al., 2023), SCINet (Liu et al., 2022), and FEDformer (Zhou et al., 2022b). Point forecasts are obtained from MOIRAI by taking the median from the samples of the predictive distribution. Tuning for MOIRAI was based on the average validation MSE across prediction lengths, further including the options between channel indepedent and channel mixing strategies (Nie et al., 2023) for the low dimension datasets (ETT and Weather). 

Table 6 reports the average performance across prediction lengths, with full results in Appendix D.3. We observe that MOIRAI achieves strong results compared to full-shot baselines. While $\mathbf { M O I R A I _ { B a s e } }$ consistently achieves strong performance across datasets with either best or second-best performance, the large model is less consistent, with slightly weaker but competitive results. The relationship between performance and model size is tenuous in this setting, however, this does not constitute strong evidence against the potential of scaling, since these results are based on models trained on a fixed dataset size and settings. Rather, this calls for more comprehensive neural scaling laws (Kaplan et al., 2020) for LTMs, to build a stronger understanding of their scaling behavior. 

## 4.3. Ablation Study

Architecture We perform a series of ablations in Table 7, starting from the default $\mathbf { M O I R A I } _ { \mathrm { S m a l l } }$ . Firstly, we ablate the multi patch size component, removing the constraints by allowing any frequency to have any patch size during training, and also simply fixing the patch size to 32. In both cases, we observe a deterioration in normalized MAE. Removing Any-variate Attention and using additive learned embeddings (randomizing variate index during training to encourage permutation invariance) instead, leads to suboptimal results, showcasing the strength of Any-variate 


Table 7. Ablation study on Monash benchmark. The aggregated normalized MAE, similarly calculated as in Figure 3 is reported.


<table><tr><td></td><td>Normalized MAE</td></tr><tr><td>MOIRAI<eq>_{Small}</eq></td><td>0.655</td></tr><tr><td>w/o patch size constraints</td><td>0.720</td></tr><tr><td>w/o multi patch size</td><td>1.156</td></tr><tr><td>w/o Any-variate Attention</td><td>0.904</td></tr><tr><td>w/o mixture distribution</td><td>0.740</td></tr><tr><td>w/o LOTSA</td><td>0.809</td></tr><tr><td>w/o packing</td><td>0.785</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/c71b163444e0ef68b62da9db6a853adf2340006364fc40567669a4acbc6e0aad.jpg)



(a) Mixture distribution.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/0a1cd1bfcb4c5f3842099f35a191321f0fd9a536b49a9a3d3af4f258a971a1c0.jpg)



(b) Student’s t-distribution.



Figure 4. Visualization of probabilistic forecasts by two variants of $\mathbf { M O I R A I S m a l l }$ on the Traffic Hourly dataset. Both models forecast peaks, however, the Student’s t-distribution has a symmetric distribution, giving inappropriate prediction intervals for a peak, as highlighted in red.


Attention. We see similar deterioration when replacing the mixture distribution with a Student’s t-distribution, and further visualize the necessity of flexible distributions for probabilistic forecasts in Figure 4. 

Training Methodology We study the impact of a large and diverse dataset by training $\mathbf { M O I R A I } _ { \mathrm { S m a l l } }$ only on the GluonTS and Monash datasets, observing that diversity of data is critical for cross-domain training even on in-distribution evaluation. Finally, given the same batch size and training iterations, we show that packed training significantly boosts performance. This is because packing increases the effective batch size and increases the number of observations the model is trained on, given the same amount of compute. 

## 4.4. Further Analysis

Context Length Our pre-training methodology varies context length defined by the task distribution. We verify that MOIRAI has the capability to take as input arbitrary context lengths by visualizing in Figure 5 the relationship between performance and increasing context lengths over three datasets in the zero-shot setting. Zeng et al. (2023); Liu et al. (2023b) previously observed that the desiderata of continuously improving performance with increasing context length is not present in conventiona Transformer-based forecasters. Here, we observe that MOIRAI indeed achieves this desired property, in fact, capable of handling thousands of time steps. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/6e67e9b552c4b1b08d8be2d1dc3d5b3f3ec33436bd3337887cc287b3e121b39b.jpg)



Figure 5. Plot of performance (MAE) against context length (xaxis in log scale) with prediction length 96 and patch size 32 on the validation set of the ETTm1, Electricity, and Weather datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/e6d8a8e7079db49ace4e26e09118b49fd8840565fede53035b96dca90cb48a33.jpg)



Figure 6. Histogram of sequence length when sampling data from LOTSA based on the proposed task distribution. Sequence length refers to the number of tokens after patching and flattening.


Packing Packing has long been applied in training LLMs and other Transformer-based models, but not for time series Transformers. While we can get away with inefficiencies when dealing with small-scale data, we start to suffer from longer training times as we scale towards the paradigm of FMs and LTMs. This is further exacerbated by our “flattened” setting which increases the disparity in sequence lengths. As evidenced in Section 4.3, keeping compute (batch size, iterations, etc.) constant, packing improves performance by 16%. To understand why this is the case, we visualize sequence length distribution in Figure 6. With a large portion of the data being shorter than the maximum sequence length, padding represents a whopping 61.08% of input tokens without packed training, and only 0.38% with our packed implementation (calculated over 1000 iterations). 

## 5. Conclusion

In this work, we introduced MOIRAI, a masked encoderbased universal time series forecasting Transformer which alleviates the issues faced in the universal forecasting paradigm. We also introduce the LOTSA, the largest collection of open-data for pre-training time series forecasting models. MOIRAI is evaluated on the in-distribution and out-of-distribution settings, and is capable of probabilistic and long sequence forecasting. We show that as a zeroshot forecaster, MOIRAI achieves competitive or superior performance compared to full-shot models. 

Limitations & Future Work While MOIRAI achieves phenomenal in and out-of-distribution performance, this is just a first step in the universal forecasting paradigm. 

Due to resource constraints, little to no hyperparameter tuning was performed – efficient tuning techniques such as $\mu \mathrm { P }$ (Yang et al., 2022a) can be applied. In terms of architecture, our approach to tackling cross-frequency learning with a multi patch size mapping is somewhat heuristic, and future work should design a more flexible and elegant approach. Also, the current architecture has limited support for high-dimensional time series, and efficient methods for extending Transformer input length can alleviate this issue. The masked encoder structure also makes it amenable to exploration of a latent diffusion architecture (Feng et al., 2024). In terms of data, LOTSA can be further enhanced with greater diversity in terms of domain and frequency. Finally, incorporating multi-modality such as tabular or text inputs is an exciting new direction which universal forecasting has unlocked. 

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## References



Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., Maddix, D. C., Rangapuram, S., Salinas, D., Schulz, J., Stella, L., TA<sup>˜</sup> ¼rkmen, A. C., and Wang, Y. Gluonts: Probabilistic and neural time series modeling in python. Journal ofMachine Learning Research, 21(116):1–6, 2020. URL http://jmlr.org/papers/v21/19-820. html. 





Awasthi, P., Das, A., Sen, R., and Suresh, A. T. On the benefits of maximum likelihood estimation for regression and forecasting. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=zrW-LVXj2k1. 





Bergmeir, C., Bui, Q., de Nijs, F., and Stuckey, P. Residential power and battery data, August 2023. URL https: //doi.org/10.5281/zenodo.8219786. 





Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 





Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020. 





CDC. Flu portal dashboard, 2017. URL https://gis.cdc.gov/grasp/fluview/ fluportaldashboard.html. 





Chen, C., Petty, K., Skabardonis, A., Varaiya, P., and Jia, Z. Freeway performance measurement system: mining loop detector data. Transportation Research Record, 1748(1): 96–102, 2001. 





Chen, S. Beijing Multi-Site Air-Quality Data. UCI Machine Learning Repository, 2019. DOI: https://doi.org/10.24432/C5RK5G. 





Das, A., Kong, W., Leach, A., Mathur, S. K., Sen, R., and Yu, R. Long-term forecasting with tiDE: Timeseries dense encoder. Transactions on Machine Learning Research, 2023a. ISSN 2835-8856. URL https: //openreview.net/forum?id=pCbC3aQB5W. 





Das, A., Kong, W., Sen, R., and Zhou, Y. A decoderonly foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023b. 





Dong, J., Wu, H., Zhang, H., Zhang, L., Wang, J., and Long, M. SimMTM: A simple pre-training framework for masked time-series modeling. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=ginTcBUnL8. 





Dooley, S., Khurana, G. S., Mohapatra, C., Naidu, S. V., and White, C. ForecastPFN: Synthetically-trained zeroshot forecasting. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=tScBQRNgjk. 





Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 





Ekambaram, V., Jati, A., Nguyen, N., Sinthong, P., and Kalagnanam, J. Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 459–469, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701030. doi: 10.1145/ 3580305.3599533. URL https://doi.org/10. 1145/3580305.3599533. 





Ekambaram, V., Jati, A., Nguyen, N. H., Dayama, P., Reddy, C., Gifford, W. M., and Kalagnanam, J. Ttms: Fast multi-level tiny time mixers for improved zero-shot and few-shot forecasting of multivariate time series. arXiv preprint arXiv:2401.03955, 2024. 





Emami, P., Sahu, A., and Graf, P. Buildingsbench: A large-scale dataset of 900k buildings and benchmark for short-term load forecasting. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https: //openreview.net/forum?id=c5rqd6PZn6. 





Feng, S., Miao, C., Zhang, Z., and Zhao, P. Latent diffusion transformer for probabilistic time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 11979–11987, 2024. 





Garza, A. and Mergenthaler-Canseco, M. Timegpt-1. arXiv preprint arXiv:2310.03589, 2023. 





Garza, F., Canseco, M. M., Challu, C., and Olivares, K. G.´ StatsForecast: Lightning fast forecasting with statistical and econometric models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/ statsforecast. 





Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American statistical Association, 102(477):359–378, 2007. 





Godahewa, R. W., Bergmeir, C., Webb, G. I., Hyndman, R., and Montero-Manso, P. Monash time series forecasting archive. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview. net/forum?id=wEc1mgAjU-. 





Gruver, N., Finzi, M. A., Qiu, S., and Wilson, A. G. Large language models are zero-shot time series forecasters. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=md68e8iZK1. 





Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y. Query-key normalization for transformers. In Cohn, T., He, Y., and Liu, Y. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4246–4253, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp. 379. URL https://aclanthology.org/2020. findings-emnlp.379. 





Hyndman, R. J. Errors on percentage errors, 4 2014. URL https://robjhyndman.com/hyndsight/ smape/. 





Hyndman, R. J. and Athanasopoulos, G. Forecasting: principles and practice. OTexts, 2018. 





Hyndman, R. J. and Koehler, A. B. Another look at measures of forecast accuracy. International journal offorecasting, 22(4):679–688, 2006. 





Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., et al. Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728, 2023. 





Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 





Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=cGDAkQo1C0p. 





Lai, G., Chang, W.-C., Yang, Y., and Liu, H. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 95–104, 2018. 





Lim, B., Arık, S. O., Loeff, N., and Pfister, T. Temporal<sup>¨</sup> fusion transformers for interpretable multi-horizon time series forecasting. International Journal ofForecasting, 37(4):1748–1764, 2021. 





Liu, M., Zeng, A., Chen, M., Xu, Z., Lai, Q., Ma, L., and Xu, Q. Scinet: Time series modeling and forecasting with sample convolution and interaction. Advances in Neural Information Processing Systems, 35:5816–5828, 2022. 





Liu, X., Xia, Y., Liang, Y., Hu, J., Wang, Y., Bai, L., Huang, C., Liu, Z., Hooi, B., and Zimmermann, R. Largest: A benchmark dataset for large-scale traffic forecasting. arXiv preprint arXiv:2306.08259, 2023a. 





Liu, X., Hu, J., Li, Y., Diao, S., Liang, Y., Hooi, B., and Zimmermann, R. Unitime: A language-empowered unified model for cross-domain time series forecasting. In Proceedings ofthe ACM Web Conference 2024, 2024. 





Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625, 2023b. 





Ma, Q., Liu, Z., Zheng, Z., Huang, Z., Zhu, S., Yu, Z., and Kwok, J. T. A survey on time-series pre-trained models. arXiv preprint arXiv:2305.10716, 2023. 





Makridakis, S., Spiliotis, E., and Assimakopoulos, V. The m4 competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1): 54–74, 2020. 





Mancuso, P., Piccialli, V., and Sudoso, A. M. A machine learning approach for forecasting hierarchical time series. Expert Systems with Applications, 182:115102, 2021. 





Mouatadid, S., Orenstein, P., Flaspohler, G. E., Oprescu, M., Cohen, J., Wang, F., Knight, S. E., Geogdzhayeva, M., Levang, S. J., Fraenkel, E., and Mackey, L. SubseasonalclimateUSA: A dataset for subseasonal forecasting and benchmarking. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview. net/forum?id=pWkrU6raMt. 





Nguyen, T., Jewik, J. K., Bansal, H., Sharma, P., and Grover, A. Climatelearn: Benchmarking machine learning for weather and climate modeling. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https: //openreview.net/forum?id=RZJEkLFlPx. 





Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=Jbdc0vTOcol. 





Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. N-beats: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=r1ecqn4YwB. 





Park, Y., Maddix, D., Aubet, F.-X., Kan, K., Gasthaus, J., and Wang, Y. Learning quantile functions without quantile crossing for distribution-free time series forecasting. In International Conference on Artificial Intelligence and Statistics, pp. 8127–8150. PMLR, 2022. 





Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551, 2020. 





Rasul, K., Ashok, A., Williams, A. R., Khorasani, A., Adamopoulos, G., Bhagwatkar, R., Bilos, M., Ghonia,ˇ H., Hassen, N. V., Schneider, A., Garg, S., Drouin, A., Chapados, N., Nevmyvaka, Y., and Rish, I. Lag-llama: Towards foundation models for time series forecasting, 2023. 





Richardson, N., Cook, I., Crane, N., Dunnington, D., Franc¸ois, R., Keane, J., Moldovan-Grunfeld, D., Ooms,¨ J., Wujciak-Jens, J., and Apache Arrow. arrow: Integration to ’Apache’ ’Arrow’, 2023. URL https: //github.com/apache/arrow/. R package version 14.0.2, https://arrow.apache.org/docs/r/. 





Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal ofForecasting, 36(3):1181–1191, 2020. 





Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 





Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 





Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Trindade, A. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C58C86. 





Van Ness, M., Shen, H., Wang, H., Jin, X., Maddix, D. C., and Gopalswamy, K. Cross-frequency time series metaforecasting. arXiv preprint arXiv:2302.02077, 2023. 





van Panhuis, W. G., Cross, A., and Burke, D. S. Project tycho 2.0: a repository to improve the integration and reuse of data for global population health. Journal of the American Medical Informatics Association, 25:1608– 1617, 2018. 





Walmart Competition Admin, W. C. Walmart recruiting - store sales forecasting, 2014. 





Wang, J., Jiang, J., Jiang, W., Han, C., and Zhao, W. X. Towards efficient and comprehensive urban spatial-temporal prediction: A unified library and performance benchmark. arXiv preprint arXiv:2304.14343, 2023a. 





Wang, Z., Wen, Q., Zhang, C., Sun, L., Von Krannichfeldt, L., and Wang, Y. Benchmarks and custom package for electrical load forecasting. arXiv preprint arXiv:2307.07191, 2023b. 





Wikipedia contributors. Moirai — Wikipedia, the free encyclopedia, 2024. URL https://en.wikipedia. org/wiki/Moirai. [Online; accessed 21-January-2024]. 





Woo, G., Liu, C., Sahoo, D., Kumar, A., and Hoi, S. CoST: Contrastive learning of disentangled seasonaltrend representations for time series forecasting. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=PilZY3omXV2. 





Woo, G., Liu, C., Kumar, A., and Sahoo, D. Pushing the limits of pre-training for time series forecasting in 





the cloudops domain. arXiv preprint arXiv:2310.05063, 2023. 





Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021. 





Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=ju_Uqw384Oq. 





Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pp. 10524– 10533. PMLR, 2020. 





Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022a. 





Yang, J., Gupta, A., Upadhyay, S., He, L., Goel, R., and Paul, S. TableFormer: Robust transformer modeling for table-text encoding. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 528–537, Dublin, Ireland, May 2022b. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.40. URL https:// aclanthology.org/2022.acl-long.40. 





Yu, H.-F., Rao, N., and Dhillon, I. S. Temporal regularized matrix factorization for high-dimensional time series prediction. Advances in neural information processing systems, 29, 2016. 





Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., and Xu, B. Ts2vec: Towards universal representation of time series. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8980–8987, 2022. 





Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023. 





Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., and Eickhoff, C. A transformer-based framework for multivariate time series representation learning. In Proceedings ofthe 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2114–2124, 2021. 





Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 





Zhang, K., Wen, Q., Zhang, C., Cai, R., Jin, M., Liu, Y., Zhang, J., Liang, Y., Pang, G., Song, D., et al. Self-supervised learning for time series analysis: Taxonomy, progress, and prospects. arXiv preprint arXiv:2306.10125, 2023. 





Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The Eleventh International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=vSVLM2j9eie. 





Zheng, Y., Yi, X., Li, M., Li, R., Shan, Z., Chang, E., and Li, T. Forecasting fine-grained air quality based on big data. In Proceedings ofthe 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 2267–2276, 2015. 





Zhou, J., Lu, X., Xiao, Y., Su, J., Lyu, J., Ma, Y., and Dou, D. Sdwpf: A dataset for spatial dynamic wind power forecasting challenge at kdd cup 2022. arXiv preprint arXiv:2208.04360, 2022a. 





Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International Conference on Machine Learning (ICML 2022), 2022b. 





Zhou, T., Niu, P., Wang, X., Sun, L., and Jin, R. One fits all: Power general time series analysis by pretrained LM. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=gMS6FVZvmF. 



## A. Large-scale Open Time Series Archive

LOTSA is a collection of time series datasets curated for pre-training of LTMs. In the following, we describe its constituent datasets and their respective sources, listing any pre-processing and data splitting performed. We further details on the key properties of these datasets, providing the domain, frequency, number of time series, number of target variates, number of past covariates (covariates whose values in the forecast horizon are unknown), and total number of observations in the dataset (defined as $\textstyle \sum _ { i = 1 } ^ { N } T _ { i }$ for a dataset with N time series). Of note, if we consider number of observations to include the number of variates, i.e. $\textstyle \sum _ { i = 1 } ^ { N } T _ { i } * d _ { y _ { i } }$ , LOTSA would have 231,082,956,489 (231B) total observations. 


Table 8. Datasets and key properties from BuildingsBench.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>BDG-2 Panther</td><td>Energy</td><td>H</td><td>105</td><td>1</td><td>0</td><td>919,800</td></tr><tr><td>BDG-2 Fox</td><td>Energy</td><td>H</td><td>135</td><td>1</td><td>0</td><td>2,324,568</td></tr><tr><td>BDG-2 Rat</td><td>Energy</td><td>H</td><td>280</td><td>1</td><td>0</td><td>4,728,288</td></tr><tr><td>BDG-2 Bear</td><td>Energy</td><td>H</td><td>91</td><td>1</td><td>0</td><td>1,482,312</td></tr><tr><td>Low Carbon London</td><td>Energy</td><td>H</td><td>713</td><td>1</td><td>0</td><td>9,543,348</td></tr><tr><td>SMART</td><td>Energy</td><td>H</td><td>5</td><td>1</td><td>0</td><td>95,709</td></tr><tr><td>IDEAL</td><td>Energy</td><td>H</td><td>219</td><td>1</td><td>0</td><td>1,265,672</td></tr><tr><td>Sceaux</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>0</td><td>34,223</td></tr><tr><td>Borealis</td><td>Energy</td><td>H</td><td>15</td><td>1</td><td>0</td><td>83,269</td></tr><tr><td>Buildings900K</td><td>Energy</td><td>H</td><td>1,792,328</td><td>1</td><td>0</td><td>15,702,590,000</td></tr></table>

BuildingsBench BuildingsBench (Emami et al., 2023) (Table 8) provides a collection of datasets for residential and commercial building energy consumption. The BDG-2 datasets, Low Carbon London, SMART, IDEAL, Sceaux, and Borealis are real building energy consumption datasets from various sources. The Electricity dataset (Trindade, 2015) is also included in BuildingsBench but we omit it from LOTSA and use it for out-of-distribution evaluation instead. They further release the Buildings-900K dataset a large-scale dataset of 900K simulated buildings. Emami et al. (2023) introduce a pre-train/test split based on Public Use Microdata Area, but we use include both splits in LOTSA for pre-training. No special pre-processing was applied to these datasets. More information regarding these datasets can be found in Emami et al. (2023). 


Table 9. Datasets and key properties from ClimateLearn.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>CMIP6</td><td>Climate</td><td>6H</td><td>1,351,680</td><td>53</td><td>0</td><td>1,973,453,000</td></tr><tr><td>ERA5</td><td>Climate</td><td>H</td><td>245,760</td><td>45</td><td>0</td><td>2,146,959,000</td></tr></table>

ClimateLearn We include the ERA5 and CMIP6 datasets from the ClimateLearn library (Nguyen et al., 2023) (Table 9). The ERA5 and CMIP6 datasets provide time series of various climate related variables such as temperature, and humidity and various pressure levels, based on a grid structure. We use the 2.8125<sup>◦</sup> resolution which contains time series in a 64 × 128 grid. 


Table 10. Datasets and key properties from CloudOps TSF


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>Azure VM Traces 2017</td><td>CloudOps</td><td>5T</td><td>159,472</td><td>1</td><td>2</td><td>885,522,908</td></tr><tr><td>Borg Cluster Data 2011</td><td>CloudOps</td><td>5T</td><td>143,386</td><td>2</td><td>5</td><td>537,552,854</td></tr><tr><td>Alibaba Cluster Trace 2018</td><td>CloudOps</td><td>5T</td><td>58,409</td><td>2</td><td>6</td><td>95,192,530</td></tr></table>

CloudOps TSF Woo et al. (2023) introduces three large-scale CloudOps time series datasets (Table 10) measuring various variables such as CPU and memory utilization. We follow their pre-train/test split and only include the pre-train time series in LOTSA, holding out the test set. 


Table 11. Datasets and key properties from the GluonTS library.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>Taxi</td><td>Transport</td><td>30T</td><td>67,984</td><td>1</td><td>0</td><td>54,999,060</td></tr><tr><td>Uber TLC Daily</td><td>Transport</td><td>D</td><td>262</td><td>1</td><td>0</td><td>47,087</td></tr><tr><td>Uber TLC Hourly</td><td>Transport</td><td>H</td><td>262</td><td>1</td><td>0</td><td>1,129,444</td></tr><tr><td>Wiki-Rolling</td><td>Web</td><td>D</td><td>47,675</td><td>1</td><td>0</td><td>40,619,100</td></tr><tr><td>M5</td><td>Sales</td><td>D</td><td>30,490</td><td>1</td><td>0</td><td>58,327,370</td></tr></table>


GluonTS The GluonTS library (Alexandrov et al., 2020) provides many datasets popularly used in time series forecasting. We only include the datasets presented in Table 11 as we either hold out the other datasets, or are already included in the Monash repository.



Table 12. Key properties of the LargeST Benchmark Dataset.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>LargeST</td><td>Transport</td><td>5T</td><td>42,333</td><td>1</td><td>0</td><td>4,452,510,528</td></tr></table>

LargeST LargeST (Liu et al., 2023a) (Table 12) collects the largest dataset from the California Department of Transportation Performance Measurement System (PeMS) (Chen et al., 2001) to date. The PeMS is a popular source of data for many traffic forecasting datasets such as PEMS03, PEMS04, PEMS07, PEMS08, and PEMS Bay, as well as the popular Traffic dataset from (Lai et al., 2018). Since we use such a large amount of dataset from the same source, we can no longer consider forecasting on any of these datasets as an out-of-distribution or zero-shot forecasting task anymore, and thus omit the Traffic dataset of the LSF benchmark from our evaluations. 


Table 13. Datasets and key properties from the LibCity library.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>PEMS03</td><td>Transport</td><td>5T</td><td>358</td><td>1</td><td>0</td><td>9,382,464</td></tr><tr><td>PEMS04</td><td>Transport</td><td>5T</td><td>307</td><td>3</td><td>0</td><td>5,216,544</td></tr><tr><td>PEMS07</td><td>Transport</td><td>5T</td><td>883</td><td>1</td><td>0</td><td>24,921,792</td></tr><tr><td>PEMS08</td><td>Transport</td><td>5T</td><td>170</td><td>3</td><td>0</td><td>3,035,520</td></tr><tr><td>PEMS Bay</td><td>Transport</td><td>5T</td><td>325</td><td>1</td><td>0</td><td>16,937,700</td></tr><tr><td>Los-Loop</td><td>Transport</td><td>5T</td><td>207</td><td>1</td><td>0</td><td>7,094,304</td></tr><tr><td>Loop Seattle</td><td>Transport</td><td>5T</td><td>323</td><td>1</td><td>0</td><td>33,953,760</td></tr><tr><td>SZ-Taxi</td><td>Transport</td><td>15T</td><td>156</td><td>1</td><td>0</td><td>464,256</td></tr><tr><td>Beijing Subway</td><td>Transport</td><td>30T</td><td>276</td><td>2</td><td>11</td><td>248,400</td></tr><tr><td>SHMetro</td><td>Transport</td><td>15T</td><td>288</td><td>2</td><td>0</td><td>1,934,208</td></tr><tr><td>HZMetro</td><td>Transport</td><td>15T</td><td>80</td><td>2</td><td>0</td><td>146,000</td></tr><tr><td>Rotterdam</td><td>Transport</td><td>2T</td><td>208</td><td>1</td><td>0</td><td>4,813,536</td></tr><tr><td>Q-Traffic</td><td>Transport</td><td>15T</td><td>45,148</td><td>1</td><td>0</td><td>264,386,688</td></tr></table>

LibCity LibCity (Wang et al., 2023a) (Table 13) provides a collection urban spatio-temporal datasets. We drop the spatial aspect of these datsets and consider them as time series data. 

Monash The Monash Time Series Forecasting Repository (Godahewa et al., 2021) (Table 14) is a large collection of diverse time series datasets, the most popular source for building LTMs. We use Monash for in-distribution evaluation, and thus apart from the exceptions listed below, we only include the train region in LOTSA, by holding out the final forecast horizon as the test set. The final forecast horizon is defined for each dataset by (Godahewa et al., 2021). For the following datasets, we include their entirety in LOTSA without a held-out test set for the following reasons: 


Table 14. Datasets and key properties from the Monash Time Series Forecasting Repository.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>London Smart Meters</td><td>Energy</td><td>30T</td><td>5,520</td><td>1</td><td>0</td><td>166,238,880</td></tr><tr><td>Wind Farms</td><td>Energy</td><td>T</td><td>337</td><td>1</td><td>0</td><td>172,165,370</td></tr><tr><td>Wind Power</td><td>Energy</td><td>4S</td><td>1</td><td>1</td><td>0</td><td>7,397,147</td></tr><tr><td>Solar Power</td><td>Energy</td><td>4S</td><td>1</td><td>1</td><td>0</td><td>7,397,222</td></tr><tr><td>Oikolab Weather</td><td>Climate</td><td>H</td><td>8</td><td>1</td><td>0</td><td>800,456</td></tr><tr><td>Elecdemand</td><td>Energy</td><td>30T</td><td>1</td><td>1</td><td>0</td><td>17,520</td></tr><tr><td>Covid Mobility</td><td>Transport</td><td>D</td><td>362</td><td>1</td><td>0</td><td>148,602</td></tr><tr><td>Kaggle Web Traffic Weekly</td><td>Web</td><td>W</td><td>145,063</td><td>1</td><td>0</td><td>16,537,182</td></tr><tr><td>Extended Web Traffic</td><td>Web</td><td>D</td><td>145,063</td><td>1</td><td>0</td><td>370,926,091</td></tr><tr><td>M1 Yearly</td><td>Econ/Fin</td><td>Y</td><td>106</td><td>1</td><td>0</td><td>3,136</td></tr><tr><td>M1 Quarterly</td><td>Econ/Fin</td><td>Q</td><td>198</td><td>1</td><td>0</td><td>9,854</td></tr><tr><td>M1 Monthly</td><td>Econ/Fin</td><td>M</td><td>617</td><td>1</td><td>0</td><td>44,892</td></tr><tr><td>M3 Yearly</td><td>Econ/Fin</td><td>Y</td><td>645</td><td>1</td><td>0</td><td>18,319</td></tr><tr><td>M3 Quarterly</td><td>Econ/Fin</td><td>Q</td><td>756</td><td>1</td><td>0</td><td>37,004</td></tr><tr><td>M3 Monthly</td><td>Econ/Fin</td><td>M</td><td>1,428</td><td>1</td><td>0</td><td>141,858</td></tr><tr><td>M3 Other</td><td>Econ/Fin</td><td>Q</td><td>174</td><td>1</td><td>0</td><td>11,933</td></tr><tr><td>M4 Yearly</td><td>Econ/Fin</td><td>Y</td><td>22,739</td><td>1</td><td>0</td><td>840,644</td></tr><tr><td>M4 Quarterly</td><td>Econ/Fin</td><td>Q</td><td>24,000</td><td>1</td><td>0</td><td>2,214,108</td></tr><tr><td>M4 Monthly</td><td>Econ/Fin</td><td>M</td><td>48,000</td><td>1</td><td>0</td><td>10,382,411</td></tr><tr><td>M4 Weekly</td><td>Econ/Fin</td><td>W</td><td>359</td><td>1</td><td>0</td><td>366,912</td></tr><tr><td>M4 Hourly</td><td>Econ/Fin</td><td>H</td><td>414</td><td>1</td><td>0</td><td>353,500</td></tr><tr><td>M4 Daily</td><td>Econ/Fin</td><td>D</td><td>4,227</td><td>1</td><td>0</td><td>9,964,658</td></tr><tr><td>NN5 Daily</td><td>Econ/Fin</td><td>D</td><td>111</td><td>1</td><td>0</td><td>81,585</td></tr><tr><td>NN5 Weekly</td><td>Econ/Fin</td><td>W</td><td>111</td><td>1</td><td>0</td><td>11,655</td></tr><tr><td>Tourism Yearly</td><td>Econ/Fin</td><td>Y</td><td>419</td><td>1</td><td>0</td><td>11,198</td></tr><tr><td>Tourism Quarterly</td><td>Econ/Fin</td><td>Q</td><td>427</td><td>1</td><td>0</td><td>39,128</td></tr><tr><td>Tourism Monthly</td><td>Econ/Fin</td><td>M</td><td>366</td><td>1</td><td>0</td><td>100,496</td></tr><tr><td>CIF 2016</td><td>Econ/Fin</td><td>M</td><td>72</td><td>1</td><td>0</td><td>6,334</td></tr><tr><td>Traffic Weekly</td><td>Transport</td><td>W</td><td>862</td><td>1</td><td>0</td><td>82,752</td></tr><tr><td>Traffic Hourly</td><td>Transport</td><td>H</td><td>862</td><td>1</td><td>0</td><td>14,978,112</td></tr><tr><td>Australian Electricity Demand</td><td>Energy</td><td>30T</td><td>5</td><td>1</td><td>0</td><td>1,153,584</td></tr><tr><td>Rideshare</td><td>Transport</td><td>H</td><td>2,304</td><td>1</td><td>0</td><td>859,392</td></tr><tr><td>Saugeen</td><td>Nature</td><td>D</td><td>1</td><td>1</td><td>0</td><td>23,711</td></tr><tr><td>Sunspot</td><td>Nature</td><td>D</td><td>1</td><td>1</td><td>0</td><td>73,894</td></tr><tr><td>Temperature Rain</td><td>Nature</td><td>D</td><td>32,072</td><td>1</td><td>0</td><td>22,290,040</td></tr><tr><td>Vehicle Trips</td><td>Transport</td><td>D</td><td>329</td><td>1</td><td>0</td><td>32,512</td></tr><tr><td>Weather</td><td>Climate</td><td>D</td><td>3,010</td><td>1</td><td>0</td><td>42,941,700</td></tr><tr><td>Car Parts</td><td>Sales</td><td>M</td><td>2,674</td><td>1</td><td>0</td><td>104,286</td></tr><tr><td>FRED MD</td><td>Econ/Fin</td><td>M</td><td>107</td><td>1</td><td>0</td><td>76,612</td></tr><tr><td>Pedestrian Counts</td><td>Transport</td><td>H</td><td>66</td><td>1</td><td>0</td><td>3,130,762</td></tr><tr><td>Hospital</td><td>Healthcare</td><td>M</td><td>767</td><td>1</td><td>0</td><td>55,224</td></tr><tr><td>COVID Deaths</td><td>Healthcare</td><td>D</td><td>266</td><td>1</td><td>0</td><td>48,412</td></tr><tr><td>KDD Cup 2018</td><td>Energy</td><td>H</td><td>270</td><td>1</td><td>0</td><td>2,897,004</td></tr><tr><td>Bitcoin</td><td>Econ/Fin</td><td>D</td><td>18</td><td>1</td><td>0</td><td>74,824</td></tr><tr><td>US Births</td><td>Healthcare</td><td>D</td><td>1</td><td>1</td><td>0</td><td>7,275</td></tr></table>

• London Smart Meters, Wind Farms, Wind Power, Solar Power, Oikolab Weather, Covid Mobility: Originally not included in the Monash benchmark. 

• Extended Web Traffic, Kaggle Web Traffic Weekly: We include the extended version of Web Traffic which contains overlap with the original Web Traffic dataset. 

• M1 Yearly, M1 Quarterly, M3 Yearly, M3 Quarterly, M4 Yearly, M4 Quarterly, Tourism Yearly: Some time series in these datasets are too short after train/test splits, thus we do not split them (setting a minimum of 16 time steps to achieve at least 2 patches). 

We omit Electricity (Trindade, 2015) and Solar (Lai et al., 2018) datasets for out-of-distribution evaluation. Note that the “Weather” from Monash is a different dataset from that used in the out-of-distribution evaluations. 


Table 15. Datasets and key properties from the ProEnFo library.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>Covid19 Energy</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>6</td><td>31,912</td></tr><tr><td>GEF12</td><td>Energy</td><td>H</td><td>20</td><td>1</td><td>1</td><td>788,280</td></tr><tr><td>GEF14</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>1</td><td>17,520</td></tr><tr><td>GEF17</td><td>Energy</td><td>H</td><td>8</td><td>1</td><td>1</td><td>140,352</td></tr><tr><td>PDB</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>1</td><td>17,520</td></tr><tr><td>Spanish</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>1</td><td>35,064</td></tr><tr><td>BDG-2 Hog</td><td>Energy</td><td>H</td><td>24</td><td>1</td><td>5</td><td>421,056</td></tr><tr><td>BDG-2 Bull</td><td>Energy</td><td>H</td><td>41</td><td>1</td><td>3</td><td>719,304</td></tr><tr><td>BDG-2 Cockatoo</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>5</td><td>17,544</td></tr><tr><td>ELF</td><td>Energy</td><td>H</td><td>1</td><td>1</td><td>0</td><td>21,792</td></tr></table>

ProEnFo ProEnFo (Wang et al., 2023b) (Table 15) provides a range of datasets for load forecasting. Unlike Buildings-Bench, ProEnFo provides various covariates such as temperature, humidity, and wind speed. We again omit Electricity (Trindade, 2015) which is originally included in ProEnFo for out-of-distribution evaluations. 


Table 16. Datasets and key properties from the SubseasonalClimateUSA library.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>Subseasonal</td><td>Climate</td><td>D</td><td>862</td><td>4</td><td>0</td><td>14,097,148</td></tr><tr><td>Subseasonal Precipitation</td><td>Climate</td><td>D</td><td>862</td><td>1</td><td>0</td><td>9,760,426</td></tr></table>

SubseasonalClimateUSA The SubseasonalClimateUSA library (Mouatadid et al., 2023) (Table 16) provides climate time series data at a lower granularity (daily) for subseasonal level forecasting. We extract two datasets Subseasonal Precipitation which is the precipitation data from 1948 - 1978, and Subseasonal, which is precipitation and temperature data from 1979 - 2023. 

Others Finally, detailed in Table 17, we further collect datasets from miscellaneous sources not provided by a library or collection. These datasets require more extensive pre-processing since they are not provided by a library, and are raw data instead. We take a standard approach of filtering out time series which are either too short, or have too many missing values. Fo each time series, we consider all variates to be targets, unless otherwise specified by the creators of the dataset (e.g. KDD Cup 2022 is a competition dataset, for which only the “Patv” variate is defined to be the target, with 9 other covariates). 


Table 17. Datasets and key properties from other miscellaneous sources.


<table><tr><td>Dataset</td><td>Source</td><td>Domain</td><td>Frequency</td><td># Time Series</td><td># Targets</td><td># Past Covariates</td><td># Obs.</td></tr><tr><td>KDD Cup 2022</td><td>Zhou et al. (2022a)</td><td>Energy</td><td>10T</td><td>134</td><td>1</td><td>9</td><td>4,727,519</td></tr><tr><td>GoDaddy</td><td>Kaggle</td><td>Econ/Fin</td><td>M</td><td>3,135</td><td>2</td><td>0</td><td>128,535</td></tr><tr><td>Favorita Sales</td><td>Kaggle</td><td>Sales</td><td>D</td><td>111,840</td><td>1</td><td>0</td><td>139,179,538</td></tr><tr><td>Favorita Transactions</td><td>Kaggle</td><td>Sales</td><td>D</td><td>54</td><td>1</td><td>0</td><td>84,408</td></tr><tr><td>Restaurant</td><td>Kaggle</td><td>Sales</td><td>D</td><td>216</td><td>1</td><td>0</td><td>76,573</td></tr><tr><td>Hierarchical Sales</td><td>Mancuso et al. (2021)</td><td>Sales</td><td>D</td><td>118</td><td>1</td><td>0</td><td>212,164</td></tr><tr><td>China Air Quality</td><td>Zheng et al. (2015)</td><td>Nature</td><td>H</td><td>437</td><td>6</td><td>0</td><td>5,739,234</td></tr><tr><td>Beijing Air Quality</td><td>Chen (2019)</td><td>Nature</td><td>H</td><td>12</td><td>11</td><td>0</td><td>420,768</td></tr><tr><td>Residential Load Power</td><td>Bergmeir et al. (2023)</td><td>Energy</td><td>T</td><td>271</td><td>3</td><td>0</td><td>145,994,559</td></tr><tr><td>Residential PV Power</td><td>Bergmeir et al. (2023)</td><td>Energy</td><td>T</td><td>233</td><td>3</td><td>0</td><td>125,338,950</td></tr><tr><td>CDC Fluview ILINet</td><td>CDC (2017)</td><td>Healthcare</td><td>W</td><td>75</td><td>5</td><td>0</td><td>63,903</td></tr><tr><td>CDC Fluview WHO NREVSS</td><td>CDC (2017)</td><td>Healthcare</td><td>W</td><td>74</td><td>4</td><td>0</td><td>41,760</td></tr><tr><td>Project Tycho</td><td>van Panhuis et al. (2018)</td><td>Healthcare</td><td>W</td><td>1,258</td><td>1</td><td>0</td><td>1,377,707</td></tr></table>

## B. MOIRAI Architecture Details

## B.1. Multi Patch Size Projection Layers

Each multi patch size projection is a simple Linear layer, for input projections, mapping patch size to hidden state, and for output projections, mapping hidden state to distribution parameters. In practice, we pre-define the frequency to patch size mapping heuristically, selecting smaller patch sizes for low frequency data and larger patch sizes for high frequency data as follows: 

• Yearly, Quarterly: 8 

• Monthly: 8, 16, 32 

• Weekly, Daily: 16, 32 

• Hourly: 32, 64 

• Minute-level: 32, 64, 128 

• Second-level: 64, 128 

Note that we only learn one Linear layer per patch size, and share them across frequencies if there is overlap. This means that we learn five input projection layers and five output projection layers. 

## B.2. Mixture Distribution

As described in Salinas et al. (2020), our model predicts the parameters of a probability distribution, in this case, a mixture distribution. We apply a softmax layer to the parameters associated to the mixture weights, constraining them to the probability simplex. The mixture components are as described. 

Student’s t-distribution A random variable x following the Student’s t-distribution has p.d.f.: 

$$
p (x; \nu , \mu , \tau) = \frac {\Gamma (\frac {\nu + 1}{2})}{\Gamma (\frac {\nu}{2}) \sqrt {\pi \nu} \tau} \left(1 + \frac {1}{\nu} \left(\frac {x - \mu}{\tau}\right) ^ {2}\right) ^ {- (\nu + 1) / 2}
$$

with parameters $\nu > 0 , \mu \in \mathbb { R } , \tau > 0$ , the degrees-of-freedom (df), location, and scale parameters respectively, and Γ is the gamma function. We predict the df, location, and scale parameters, and apply a softplus function for the positivity constraint. We further lower bound the df parameter to 2, since variance is undefined otherwise. 

Log-normal distribution A random variable x which follows a log-normal distribution has p.d.f.: 

$$
p (x; \mu , \sigma) = \frac {1}{x \sigma \sqrt {2 \pi}} \exp \left(- \frac {(\ln x - \mu) ^ {2}}{2 \sigma^ {2}}\right)
$$

with parameters $\mu \in \mathbb { R } , \sigma > 0$ . We predict both parameters, applying softplus function for positivity. 

Negative binomial distribution Following Awasthi et al. (2022), we implement a continuous extension of the negative binomial distribution. A random variable x following such a distribution has p.d.f.: 

$$
p (x; r, p) \propto \frac {\Gamma (x + r)}{\Gamma (x + 1) \Gamma (r)} (1 - p) ^ {r} p ^ {x}
$$

given parameters $r > 0$ and $p \in [ 0 , 1 ]$ , and Γ is the gamma function. We predict both parameters, applying the softplus function for positivity, and sigmoid function to constrain to a probability. 

Low variance normal distribution A random variable x following a normal distribution has p.d.f.: 

$$
p (x; \mu , \sigma) = \frac {1}{\sigma \sqrt {2 \pi}} \exp \left(- \frac {(x - \mu) ^ {2}}{2 \sigma^ {2}}\right)
$$

where $\mu \in \mathbb { R } , \sigma > 0$ . For a low variance normal distribution, we only predict $\mu ,$ and fix σ to a small number, in this case we fix $\sigma = \mathrm { 1 e { - } 3 }$ 

## B.3. Discussion on “Flexible Distribution”

Table 1 categorizes various pre-trained forecasting models with the notion of a “flexible distribution” – this is largel a qualitative categorization rather than a quantitative one. As of writing, only 3 other models considered probabilistic forecasting – Lag-llama, TimeGPT, and LLMTime. The other models only considered point forecasts, and thus the concept of ”flexible distribution” does not apply to them. The following are specific reasons on why we made the categorization for the 3 models which can handle probabilistic forecasting: 

• Lag-llama uses a Student-T distribution which is only able to model symmetric distributions. This is an inflexible distribution which is unable to model asymmetric distributions, as demonstrated in Figure 4 of our paper. They also raise this issue in their paper (Section 4.3), citing the use of more expressive distribution heads such as normalizing flows and copulas in future work. 

• TimeGPT uses conformal prediction to construct prediction intervals. We refer to a tweet<sup>2</sup> from the creators, which claim: ”Some prediction intervals don’t account for domain constraints. A few users highlighted intervals suggesting negative values for time series that only take positive values.” Thus, we consider it to be inflexible. 

• LLMTime uses a categorical distribution. In their paper (paragraph titled ”Language models as flexible distributions” in Section 3), they demonstrated that this approach is a flexible distribution which can approximate many different kinds of continuous distributions. 

## C. Probabilistic Forecasting

## C.1. Evaluation Metrics

Continuous Ranked Probability Score The CRPS (Gneiting & Raftery, 2007) is a probabilistic forecasting evaluation metric, given a predicted distribution with c.d.f. F and ground truth $y ,$ it is defined as: 

$$
\begin{array}{c} \text {CRPS} = \int_ {0} ^ {1} 2 \Lambda_ {\alpha} (F ^ {- 1} (\alpha), y) d \alpha \\ \Lambda_ {\alpha} (q, y) = (\alpha - \mathbf {1} _ {\mathrm{y} <   \mathrm{q}}) (y - q), \end{array}
$$

where $\Lambda _ { \alpha }$ is the α-quantile loss, also known as the pinball loss at quantile level $\alpha .$ 

In practice, the CRPS is intractable or computationally expensive to compute, and we also want to compute a normalized metric, thus we compute a normalized discrete approximation, the mean weighted sum quantile loss (Park et al., 2022), defined as the average of K quantiles: 

$$
\begin{array}{c} \text {CRPS} \approx \frac {1}{K} \sum_ {k = 1} ^ {K} \mathrm{wQL} [ \alpha_ {k} ] \\ \mathrm{wQL} [ \alpha ] = 2 \frac {\sum_ {t} \Lambda_ {\alpha} (\hat {q} _ {t} (\alpha) , y _ {t})}{\sum_ {t} | y _ {t} |}, \end{array}
$$

where $\hat { q } _ { t } ( \alpha )$ is the predicted α-quantile at time step t. We take $K = 9 , \alpha _ { 1 } = 0 . 1 , \alpha _ { 2 } = 0 . 2 , . . . , \alpha _ { 9 } = 0 . 9$ in practice. 

Mean Scaled Interval Score The MSIS is a metric to evaluate uncertainty around point forecasts, introduced in the M4 Competition (Makridakis et al., 2020). Given an upper bound prediction, $U _ { t }$ , and lower bound prediction $L _ { t }$ , the MSIS is defined as: 

$$
\mathrm{MSIS} = \frac {1}{h} \frac {\sum_ {t = 1} ^ {h} (U _ {t} - L _ {t}) + \frac {2}{a} (L _ {t} - Y _ {t}) \mathbb {1} _ {\{Y _ {t} <   L _ {t} \}} + \frac {2}{a} (Y _ {t} - U _ {t}) \mathbb {1} _ {\{Y _ {t} > U _ {t} \}}}{\frac {1}{n - m} \sum_ {t = m + 1} ^ {n} | Y _ {t} - Y _ {t - m} |}
$$

where $a = 0 . 0 5$ is the significance level for a 95% prediction interval, over a forecast horizon of length h, and m is the seasonal factor. 

## C.2. Evaluation Setup


Table 18. Summary of datasets used in the out-of-distribution probabilistic forecasting evaluation setting.


<table><tr><td>Dataset</td><td>Domain</td><td>Frequency</td><td>Prediction Length</td><td>Rolling Evaluations</td></tr><tr><td>Electricity (Trindade, 2015)</td><td>Energy</td><td>H</td><td>24</td><td>7</td></tr><tr><td>Solar (Lai et al., 2018)</td><td>Energy</td><td>H</td><td>24</td><td>7</td></tr><tr><td>Walmart (Walmart Competition Admin, 2014)</td><td>Sales</td><td>W</td><td>8</td><td>4</td></tr><tr><td>Weather</td><td>Climate</td><td>10T</td><td>144</td><td>7</td></tr><tr><td><eq>Istanbul Traffic ^{3}</eq></td><td>Transport</td><td>H</td><td>24</td><td>7</td></tr><tr><td><eq>Turkey Power ^{4}</eq></td><td>Energy</td><td>H</td><td>24</td><td>7</td></tr></table>

We perform evaluation in a non-overlapping rolling window fashion, i.e. stride is equal to prediction length. The test set is defined as the last $h * r$ time steps where h is the prediction length of the forecast horizon, and r is the number of rolling evaluation windows. We take the validation set to be the last forecast horizon before the test set, and the train set to be everything before that. From Table 18, our evaluation spans four domains, from minute-level to weekly frequencies. Prediction length and rolling evaluations are defined for each dataset based on frequency, making day ahead predictions for sub-hourly frequencies for seven days, and eight week ahead predictions for 32 weeks for weekly frequency. 

## C.3. Baselines


Table 19. Hyperparameter search values for probabilistic forecasting evaluation baselines.


<table><tr><td></td><td>Hyperparameter</td><td>Values</td></tr><tr><td rowspan="2">PatchTST</td><td>d_model</td><td>{64, 128, 256}</td></tr><tr><td>num_encoder_layers</td><td>[2, 6]</td></tr><tr><td rowspan="2">DeepAR</td><td>hidden_size</td><td>{64, 128, 256}</td></tr><tr><td>num_layers</td><td>[2, 6]</td></tr><tr><td>TFT</td><td>hidden_dim</td><td>{64, 128, 256}</td></tr><tr><td rowspan="2">TiDE</td><td>hidden_dim</td><td>{64, 128, 256}</td></tr><tr><td>num_encoder_layers = num_decoder_layers</td><td>[2, 6]</td></tr></table>

For the four deep learning baselines, DeepAR (Salinas et al., 2020), PatchTST (Nie et al., 2023), TiDE (Das et al., 2023a), and TFT (Lim et al., 2021), we perform hyperparameter tuning based on the values presented in Table 19, and also tune learning rate [1e-6, 1e-3] in log scale, and the context length as $l = m * h ,$ , where m is tuned in the range [2, 20], and h is the prediction length. We perform random search through these values over 15 training runs, and report results on 5 independen training runs based on the best validation CRPS. 

## D. Full Experimental Results

D.1. In-distribution Forecasting: Monash Time Series Forecasting Benchmark 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/e7c020de463dc159d59c5775e7e462a301368396ac930439a5fa077a897b24d1.jpg)



(a) Results aggregated over full all datasets in Table 20.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/dbe85a71c4d5958df9bc51501b75e0892e61ee3d8b97bb69f46d819e3ce48dde.jpg)



(b) Results aggregated over LLaMA2 subset in Table 20.


Figure 7. Extended aggregate results of the Monash Time Series Forecasting Benchmark as per Figure 3. GPT3.5 is our reproduction of LLMTime based on the GPT3.5 API, whereas LLaMA2 is based on the results reported in Gruver et al. (2023). 


Table 20. Full results of Monash Time Series Forecasting Benchmark. MAE is reported.


<table><tr><td></td><td>MOIRAISmall</td><td>MOIRAIBase</td><td>MOIRAILarge</td><td>Naive</td><td>SES</td><td>Theta</td><td>TBATS</td><td>ETS</td><td>(DHR)-ARIMA</td><td>PR</td><td>CatBoost</td><td>FFNN</td><td>DeepAR</td><td>N-BEATS</td><td>WaveNet</td><td>Transformer</td><td>GPT3.5</td><td>LLaMA2</td></tr><tr><td>M1 Monthly</td><td>2,082.26</td><td>2,068.63</td><td>1,983.18</td><td>2,707.75</td><td>2,259.04</td><td>2,166.18</td><td>2,237.50</td><td>1,905.28</td><td>2,080.13</td><td>2,088.25</td><td>2,052.32</td><td>2,162.58</td><td>1,860.81</td><td>1,820.37</td><td>2,184.42</td><td>2,723.88</td><td>2562.84</td><td>-</td></tr><tr><td>M3 Monthly</td><td>713.41</td><td>658.17</td><td>664.03</td><td>837.14</td><td>743.41</td><td>623.71</td><td>630.59</td><td>626.46</td><td>654.8</td><td>692.97</td><td>732</td><td>692.48</td><td>728.81</td><td>648.6</td><td>699.3</td><td>798.38</td><td>877.97</td><td>-</td></tr><tr><td>M3 Other</td><td>263.54</td><td>198.62</td><td>202.41</td><td>278.43</td><td>277.83</td><td>215.35</td><td>189.42</td><td>194.98</td><td>193.02</td><td>234.43</td><td>318.13</td><td>240.17</td><td>247.56</td><td>221.85</td><td>245.29</td><td>239.24</td><td>300.30</td><td>-</td></tr><tr><td>M4 Monthly</td><td>597.6</td><td>592.09</td><td>584.36</td><td>671.27</td><td>625.24</td><td>563.58</td><td>589.52</td><td>582.6</td><td>575.36</td><td>596.19</td><td>611.69</td><td>612.52</td><td>615.22</td><td>578.48</td><td>655.51</td><td>780.47</td><td>728.27</td><td>-</td></tr><tr><td>M4 Weekly</td><td>339.76</td><td>328.08</td><td>301.52</td><td>347.99</td><td>336.82</td><td>333.32</td><td>296.15</td><td>335.66</td><td>321.61</td><td>293.21</td><td>364.65</td><td>338.37</td><td>351.78</td><td>277.73</td><td>359.46</td><td>378.89</td><td>518.44</td><td>-</td></tr><tr><td>M4 Daily</td><td>189.1</td><td>192.66</td><td>189.78</td><td>180.83</td><td>178.27</td><td>178.86</td><td>176.6</td><td>193.26</td><td>179.67</td><td>181.92</td><td>231.36</td><td>177.91</td><td>299.79</td><td>190.44</td><td>189.47</td><td>201.08</td><td>266.52</td><td>-</td></tr><tr><td>M4 Hourly</td><td>268.04</td><td>209.87</td><td>197.79</td><td>1,218.06</td><td>1,218.06</td><td>1,220.97</td><td>386.27</td><td>3,358.10</td><td>1,310.85</td><td>257.39</td><td>285.35</td><td>385.49</td><td>886.02</td><td>425.75</td><td>393.63</td><td>320.54</td><td>576.06</td><td>-</td></tr><tr><td>Tourism Quarterly</td><td>18,352.44</td><td>17,196.86</td><td>15,820.02</td><td>15,845.10</td><td>15,014.19</td><td>7,656.49</td><td>9,972.42</td><td>8,925.52</td><td>10,475.47</td><td>9,092.58</td><td>10,267.97</td><td>8,981.04</td><td>9,511.37</td><td>8,640.56</td><td>9,137.12</td><td>9,521.67</td><td>16918.86</td><td>9311.98</td></tr><tr><td>Tourism Monthly</td><td>3,569.85</td><td>2,862.06</td><td>2,688.55</td><td>5,636.83</td><td>5,302.10</td><td>2,069.96</td><td>2,940.08</td><td>2,004.51</td><td>2,536.77</td><td>2,187.28</td><td>2,537.04</td><td>2,022.21</td><td>1,871.69</td><td>2,003.02</td><td>2,095.13</td><td>2,146.98</td><td>5608.61</td><td>3145.48</td></tr><tr><td>CIF 2016</td><td>655,888.58</td><td>539,222.03</td><td>695,156.92</td><td>578,596.53</td><td>581,875.97</td><td>714,818.58</td><td>855,578.40</td><td>642,421.42</td><td>469,059.49</td><td>563,205.57</td><td>603,531.30</td><td>1,495,923.44</td><td>3,200,418.00</td><td>679,034.80</td><td>5,998,224.62</td><td>4,057,973.00</td><td>599,313.84</td><td>684057.87</td></tr><tr><td>Aus. Elec. Demand</td><td>266.57</td><td>201.39</td><td>177.68</td><td>659.6</td><td>659.6</td><td>665.04</td><td>370.74</td><td>1,282.99</td><td>1,045.92</td><td>247.18</td><td>241.77</td><td>258.76</td><td>302.41</td><td>213.83</td><td>227.5</td><td>231.45</td><td>760.81</td><td>560.48</td></tr><tr><td>Bitcoin</td><td>1.76E+18</td><td>1.62E+18</td><td>1.87E+18</td><td>7.78E+17</td><td>4.33E+18</td><td>5.33E+18</td><td>9.90E+17</td><td>1.10E+18</td><td>3.62E+18</td><td>6.66E+17</td><td>1.93E+18</td><td>1.45E+18</td><td>1.95E+18</td><td>1.06E+18</td><td>2.46E+18</td><td>2.61E+18</td><td>1.74E+18</td><td>8.57E+17</td></tr><tr><td>Pedestrian Counts</td><td>54.88</td><td>54.08</td><td>41.66</td><td>170.88</td><td>170.87</td><td>170.94</td><td>222.38</td><td>216.5</td><td>635.16</td><td>44.18</td><td>43.41</td><td>46.41</td><td>44.78</td><td>66.84</td><td>46.46</td><td>47.29</td><td>97.77</td><td>65.92</td></tr><tr><td>Vehicle Trips</td><td>24.46</td><td>23.17</td><td>21.85</td><td>31.42</td><td>29.98</td><td>30.76</td><td>21.21</td><td>30.95</td><td>30.07</td><td>27.24</td><td>22.61</td><td>22.93</td><td>22</td><td>28.16</td><td>24.15</td><td>28.01</td><td>31.48</td><td>-</td></tr><tr><td>KDD cup</td><td>39.81</td><td>38.66</td><td>39.09</td><td>42.13</td><td>42.04</td><td>42.06</td><td>39.2</td><td>44.88</td><td>52.2</td><td>36.85</td><td>34.82</td><td>37.16</td><td>48.98</td><td>49.1</td><td>37.08</td><td>44.46</td><td>42.72</td><td>-</td></tr><tr><td>Weather</td><td>1.96</td><td>1.8</td><td>1.75</td><td>2.36</td><td>2.24</td><td>2.51</td><td>2.3</td><td>2.35</td><td>2.45</td><td>8.17</td><td>2.51</td><td>2.09</td><td>2.02</td><td>2.34</td><td>2.29</td><td>2.03</td><td>2.17</td><td>2.09</td></tr><tr><td>NN5 Daily</td><td>5.37</td><td>4.26</td><td>3.77</td><td>8.26</td><td>6.63</td><td>3.8</td><td>3.7</td><td>3.72</td><td>4.41</td><td>5.47</td><td>4.22</td><td>4.06</td><td>3.94</td><td>4.92</td><td>3.97</td><td>4.16</td><td>7.10</td><td>6.67</td></tr><tr><td>NN5 Weekly</td><td>15.07</td><td>16.42</td><td>15.3</td><td>16.71</td><td>15.66</td><td>15.3</td><td>14.98</td><td>15.7</td><td>15.38</td><td>14.94</td><td>15.29</td><td>15.02</td><td>14.69</td><td>14.19</td><td>19.34</td><td>20.34</td><td>15.76</td><td>15.60</td></tr><tr><td>Carparts</td><td>0.53</td><td>0.47</td><td>0.49</td><td>0.65</td><td>0.55</td><td>0.53</td><td>0.58</td><td>0.56</td><td>0.56</td><td>0.41</td><td>0.53</td><td>0.39</td><td>0.39</td><td>0.98</td><td>0.4</td><td>0.39</td><td>0.44</td><td>-</td></tr><tr><td>FRED-MD</td><td>2,568.48</td><td>2,679.29</td><td>2,792.55</td><td>2,825.67</td><td>2,798.22</td><td>3,492.84</td><td>1,989.97</td><td>2,041.42</td><td>2,957.11</td><td>8,921.94</td><td>2,475.68</td><td>2,339.57</td><td>4,264.36</td><td>2,557.80</td><td>2,508.40</td><td>4,666.04</td><td>2804.64</td><td>1781.41</td></tr><tr><td>Traffic Hourly</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.03</td><td>0.04</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.03</td><td>0.02</td></tr><tr><td>Traffic Weekly</td><td>1.17</td><td>1.14</td><td>1.13</td><td>1.19</td><td>1.12</td><td>1.13</td><td>1.17</td><td>1.14</td><td>1.22</td><td>1.13</td><td>1.17</td><td>1.15</td><td>1.18</td><td>1.11</td><td>1.2</td><td>1.42</td><td>1.15</td><td>1.15</td></tr><tr><td>Rideshare</td><td>1.35</td><td>1.39</td><td>1.29</td><td>6.29</td><td>6.29</td><td>7.62</td><td>6.45</td><td>6.29</td><td>3.37</td><td>6.3</td><td>6.07</td><td>6.59</td><td>6.28</td><td>5.55</td><td>2.75</td><td>6.29</td><td>6.28</td><td>-</td></tr><tr><td>Hospital</td><td>23</td><td>19.4</td><td>19.44</td><td>24.07</td><td>21.76</td><td>18.54</td><td>17.43</td><td>17.97</td><td>19.6</td><td>19.24</td><td>19.17</td><td>22.86</td><td>18.25</td><td>20.18</td><td>19.35</td><td>36.19</td><td>25.68</td><td>22.75</td></tr><tr><td>COVID Deaths</td><td>124.32</td><td>126.11</td><td>117.11</td><td>353.71</td><td>353.71</td><td>321.32</td><td>96.29</td><td>85.59</td><td>85.77</td><td>347.98</td><td>475.15</td><td>144.14</td><td>201.98</td><td>158.81</td><td>1,049.48</td><td>408.66</td><td>653.31</td><td>66.14</td></tr><tr><td>Temperature Rain</td><td>5.3</td><td>5.08</td><td>5.27</td><td>9.39</td><td>8.18</td><td>8.22</td><td>7.14</td><td>8.21</td><td>7.19</td><td>6.13</td><td>6.76</td><td>5.56</td><td>5.37</td><td>7.28</td><td>5.81</td><td>5.24</td><td>6.37</td><td>-</td></tr><tr><td>Sunspot</td><td>0.11</td><td>0.08</td><td>0.13</td><td>3.93</td><td>4.93</td><td>4.93</td><td>2.57</td><td>4.93</td><td>2.57</td><td>3.83</td><td>2.27</td><td>7.97</td><td>0.77</td><td>14.47</td><td>0.17</td><td>0.13</td><td>5.07</td><td>0.28</td></tr><tr><td>Saugeen River Flow</td><td>24.07</td><td>24.4</td><td>24.76</td><td>21.5</td><td>21.5</td><td>21.49</td><td>22.26</td><td>30.69</td><td>22.38</td><td>25.24</td><td>21.28</td><td>22.98</td><td>23.51</td><td>27.92</td><td>22.17</td><td>28.06</td><td>34.84</td><td>23.01</td></tr><tr><td>US Births</td><td>872.51</td><td>624.3</td><td>476.5</td><td>1,152.67</td><td>1,192.20</td><td>586.93</td><td>399</td><td>419.73</td><td>526.33</td><td>574.93</td><td>441.7</td><td>557.87</td><td>424.93</td><td>422</td><td>504.4</td><td>452.87</td><td>1374.99</td><td>638.82</td></tr></table>

We include the full breakdown of results for the Monash benchmark in Table 20, including two versions of LLMTime: GPT3.5 (our reproduction), and LLaMA2 (results from Gruver et al. (2023)). GPT3.5 is our reproduction by running their code<sup>5</sup> on the full dataset, using GPT3.5-Turbo-Instruct since text-davinci-003 has been deprecated. LLaMA2 only has results for a subset of datasets in Table 20, thus in Figure 7, we present two aggregated results, one aggregated on the full Table 20, and one on the subset with results available for LLaMA2. As observed, MOIRAI retains the top rankings for with the base and large models in all settings. 

## D.2. Out-of-distribution Forecasting: Probabilistic Forecasting

Table 21 provides the full results of the probabilistic forecasting experiments with additional point forecasting metrics, including the symmetric mean absolute percentage error (sMAPE) (Hyndman, 2014), mean absolute scaled error (MASE) (Hyndman & Koehler, 2006), normalized deviation (ND), and normalized root mean squared error (NRMSE) (Yu et al., 2016). 

## D.3. Out-of-distribution Forecasting: Long Sequence Forecasting

Table 22 provides the full breakdown of results for the long sequence forecasting experiments, listing results for each prediction length. 


Table 21. Full results for probabilistic forecasting experiments. Best results are highlighted in bold, and second best results are underlined.


<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Zero-shot</td><td colspan="4">Full-shot</td><td colspan="2">Baseline</td></tr><tr><td><eq>MOIRAI_{Small}</eq></td><td><eq>MOIRAI_{Base}</eq></td><td><eq>MOIRAI_{Large}</eq></td><td>PatchTST</td><td>TiDE</td><td>TFT</td><td>DeepAR</td><td>AutoARIMA</td><td>Seasonal Naive</td></tr><tr><td rowspan="6">Electricity</td><td>CRPS</td><td>0.072</td><td>0.055</td><td>0.050</td><td>0.052±0.00</td><td>0.048±0.00</td><td>0.050±0.00</td><td>0.065±0.01</td><td>0.327</td><td>0.070</td></tr><tr><td>MSIS</td><td>7.999</td><td>6.172</td><td>5.875</td><td>5.744±0.12</td><td>5.672±0.08</td><td>6.278±0.24</td><td>6.893±0.82</td><td>29.412</td><td>35.251</td></tr><tr><td>sMAPE</td><td>0.134</td><td>0.111</td><td>0.106</td><td>0.107±0.00</td><td>0.102±0.00</td><td>0.106±0.01</td><td>0.118±0.02</td><td>0.318</td><td>0.108</td></tr><tr><td>MASE</td><td>0.981</td><td>0.792</td><td>0.751</td><td>0.753±0.01</td><td>0.706±0.02</td><td>0.747±0.03</td><td>0.844±0.16</td><td>3.229</td><td>0.881</td></tr><tr><td>ND</td><td>0.092</td><td>0.069</td><td>0.063</td><td>0.065±0.00</td><td>0.061±0.00</td><td>0.063±0.00</td><td>0.080±0.02</td><td>0.357</td><td>0.070</td></tr><tr><td>NRMSE</td><td>0.840</td><td>0.551</td><td>0.465</td><td>0.506±0.02</td><td>0.514±0.02</td><td>0.511±0.02</td><td>0.704±0.11</td><td>3.296</td><td>0.478</td></tr><tr><td rowspan="6">Solar</td><td>CRPS</td><td>0.471</td><td>0.419</td><td>0.406</td><td>0.518±0.09</td><td>0.420±0.00</td><td>0.446±0.03</td><td>0.431±0.01</td><td>1.055</td><td>0.512</td></tr><tr><td>MSIS</td><td>8.425</td><td>7.011</td><td>6.250</td><td>8.447±1.59</td><td>13.754±0.32</td><td>8.057±3.51</td><td>11.181±0.67</td><td>25.849</td><td>48.130</td></tr><tr><td>sMAPE</td><td>1.445</td><td>1.410</td><td>1.400</td><td>1.501±0.10</td><td>1.400±0.00</td><td>1.391±0.01</td><td>1.385±0.00</td><td>1.685</td><td>0.691</td></tr><tr><td>MASE</td><td>1.465</td><td>1.292</td><td>1.237</td><td>1.607±0.25</td><td>1.265±0.02</td><td>1.399±0.11</td><td>1.222±0.01</td><td>2.583</td><td>1.203</td></tr><tr><td>ND</td><td>0.624</td><td>0.551</td><td>0.528</td><td>0.685±0.11</td><td>0.538±0.01</td><td>0.594±0.05</td><td>0.520±0.00</td><td>1.098</td><td>0.512</td></tr><tr><td>NRMSE</td><td>1.135</td><td>1.034</td><td>1.014</td><td>1.408±0.26</td><td>1.093±0.00</td><td>1.236±0.06</td><td>1.033±0.01</td><td>1.784</td><td>1.168</td></tr><tr><td rowspan="6">Walmart</td><td>CRPS</td><td>0.103</td><td>0.093</td><td>0.098</td><td>0.082±0.01</td><td>0.077±0.00</td><td>0.087±0.00</td><td>0.121±0.00</td><td>0.124</td><td>0.151</td></tr><tr><td>MSIS</td><td>9.371</td><td>8.421</td><td>8.520</td><td>6.005±0.21</td><td>6.258±0.12</td><td>8.718±0.10</td><td>12.502±0.03</td><td>9.888</td><td>49.458</td></tr><tr><td>sMAPE</td><td>0.179</td><td>0.168</td><td>0.174</td><td>0.150±0.01</td><td>0.145±0.00</td><td>0.172±0.00</td><td>0.216±0.00</td><td>0.219</td><td>0.205</td></tr><tr><td>MASE</td><td>1.048</td><td>0.964</td><td>1.007</td><td>0.867±0.09</td><td>0.814±0.01</td><td>0.948±0.02</td><td>1.193±0.02</td><td>1.131</td><td>1.236</td></tr><tr><td>ND</td><td>0.129</td><td>0.117</td><td>0.124</td><td>0.105±0.01</td><td>0.097±0.00</td><td>0.108±0.00</td><td>0.147±0.00</td><td>0.141</td><td>0.151</td></tr><tr><td>NRMSE</td><td>0.324</td><td>0.291</td><td>0.332</td><td>0.218±0.02</td><td>0.204±0.00</td><td>0.235±0.01</td><td>0.298±0.00</td><td>0.305</td><td>0.328</td></tr><tr><td rowspan="6">Weather</td><td>CRPS</td><td>0.049</td><td>0.041</td><td>0.051</td><td>0.059±0.01</td><td>0.054±0.00</td><td>0.043±0.00</td><td>0.132±0.11</td><td>0.252</td><td>0.068</td></tr><tr><td>MSIS</td><td>5.236</td><td>5.136</td><td>4.962</td><td>7.759±0.49</td><td>8.095±1.74</td><td>7.791±0.44</td><td>21.651±17.34</td><td>19.805</td><td>31.293</td></tr><tr><td>sMAPE</td><td>0.686</td><td>0.623</td><td>0.688</td><td>0.668±0.01</td><td>0.636±0.01</td><td>0.672±0.01</td><td>0.776±0.05</td><td>0.770</td><td>0.401</td></tr><tr><td>MASE</td><td>0.521</td><td>0.487</td><td>0.515</td><td>0.844±0.19</td><td>0.832±0.13</td><td>0.692±0.02</td><td>3.170±3.47</td><td>0.938</td><td>0.782</td></tr><tr><td>ND</td><td>0.063</td><td>0.048</td><td>0.063</td><td>0.072±0.01</td><td>0.066±0.01</td><td>0.051±0.00</td><td>0.163±0.15</td><td>0.139</td><td>0.068</td></tr><tr><td>NRMSE</td><td>0.229</td><td>0.417</td><td>0.331</td><td>0.260±0.01</td><td>0.214±0.00</td><td>0.211±0.00</td><td>0.486±0.43</td><td>0.465</td><td>0.290</td></tr><tr><td rowspan="6">Istanbul Traffic</td><td>CRPS</td><td>0.173</td><td>0.116</td><td>0.112</td><td>0.112±0.00</td><td>0.110±0.01</td><td>0.110±0.01</td><td>0.108±0.00</td><td>0.589</td><td>0.257</td></tr><tr><td>MSIS</td><td>5.937</td><td>4.461</td><td>4.277</td><td>3.813±0.09</td><td>4.752±0.17</td><td>4.057±0.44</td><td>4.094±0.31</td><td>16.317</td><td>45.473</td></tr><tr><td>sMAPE</td><td>0.359</td><td>0.284</td><td>0.288</td><td>0.287±0.01</td><td>0.280±0.01</td><td>0.287±0.01</td><td>0.249±0.01</td><td>1.141</td><td>0.391</td></tr><tr><td>MASE</td><td>0.990</td><td>0.644</td><td>0.631</td><td>0.653±0.02</td><td>0.618±0.03</td><td>0.620±0.03</td><td>0.613±0.03</td><td>3.358</td><td>1.137</td></tr><tr><td>ND</td><td>0.224</td><td>0.146</td><td>0.143</td><td>0.148±0.01</td><td>0.140±0.01</td><td>0.141±0.01</td><td>0.139±0.01</td><td>0.758</td><td>0.257</td></tr><tr><td>NRMSE</td><td>0.294</td><td>0.194</td><td>0.186</td><td>0.190±0.01</td><td>0.185±0.01</td><td>0.185±0.01</td><td>0.181±0.01</td><td>0.959</td><td>0.384</td></tr><tr><td rowspan="6">Turkey Power</td><td>CRPS</td><td>0.048</td><td>0.040</td><td>0.036</td><td>0.054±0.01</td><td>0.046±0.01</td><td>0.039±0.00</td><td>0.066±0.02</td><td>0.116</td><td>0.085</td></tr><tr><td>MSIS</td><td>7.127</td><td>6.766</td><td>6.341</td><td>8.978±0.51</td><td>8.579±0.52</td><td>7.943±0.31</td><td>13.520±1.17</td><td>14.863</td><td>36.256</td></tr><tr><td>sMAPE</td><td>0.389</td><td>0.378</td><td>0.375</td><td>0.416±0.01</td><td>0.389±0.00</td><td>0.383±0.00</td><td>0.404±0.01</td><td>0.244</td><td>0.125</td></tr><tr><td>MASE</td><td>0.948</td><td>0.888</td><td>0.870</td><td>1.234±0.12</td><td>0.904±0.02</td><td>0.890±0.05</td><td>1.395±0.30</td><td>1.700</td><td>0.906</td></tr><tr><td>ND</td><td>0.061</td><td>0.051</td><td>0.046</td><td>0.071±0.01</td><td>0.059±0.01</td><td>0.049±0.00</td><td>0.083±0.02</td><td>0.150</td><td>0.085</td></tr><tr><td>NRMSE</td><td>0.149</td><td>0.118</td><td>0.102</td><td>0.158±0.01</td><td>0.139±0.03</td><td>0.104±0.01</td><td>0.181±0.05</td><td>0.383</td><td>0.231</td></tr></table>


Table 22. Full results of long sequence forecasting experiments. Best results are highlighted in bold, and second best results are underlined. Full-shot results are obtained from Liu et al. (2023b).


<table><tr><td rowspan="3" colspan="2"></td><td colspan="6">Zero-shot</td><td colspan="16">Full-shot</td></tr><tr><td colspan="2"><eq>MOIRAI_{Small}</eq></td><td colspan="2"><eq>MOIRAI_{Base}</eq></td><td colspan="2"><eq>MOIRAI_{Large}</eq></td><td colspan="2">iTransformer</td><td colspan="2">TimesNet</td><td colspan="2">PatchTST</td><td colspan="2">Crossformer</td><td colspan="2">TiDE</td><td colspan="2">DLinear</td><td colspan="2">SCINet</td><td colspan="2">FEDformer</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.375</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.380</td><td>0.398</td><td>0.386</td><td>0.405</td><td>0.384</td><td>0.402</td><td>0.414</td><td>0.419</td><td>0.423</td><td>0.448</td><td>0.479</td><td>0.464</td><td>0.386</td><td>0.400</td><td>0.654</td><td>0.599</td><td>0.376</td><td>0.419</td></tr><tr><td>192</td><td>0.399</td><td>0.419</td><td>0.425</td><td>0.429</td><td>0.440</td><td>0.434</td><td>0.441</td><td>0.436</td><td>0.436</td><td>0.429</td><td>0.460</td><td>0.445</td><td>0.471</td><td>0.474</td><td>0.525</td><td>0.492</td><td>0.437</td><td>0.432</td><td>0.719</td><td>0.631</td><td>0.420</td><td>0.448</td></tr><tr><td>336</td><td>0.412</td><td>0.429</td><td>0.456</td><td>0.450</td><td>0.514</td><td>0.474</td><td>0.487</td><td>0.458</td><td>0.491</td><td>0.469</td><td>0.501</td><td>0.466</td><td>0.570</td><td>0.546</td><td>0.565</td><td>0.515</td><td>0.481</td><td>0.459</td><td>0.778</td><td>0.659</td><td>0.459</td><td>0.465</td></tr><tr><td>720</td><td>0.413</td><td>0.444</td><td>0.470</td><td>0.473</td><td>0.705</td><td>0.568</td><td>0.503</td><td>0.491</td><td>0.521</td><td>0.500</td><td>0.500</td><td>0.488</td><td>0.653</td><td>0.621</td><td>0.594</td><td>0.558</td><td>0.519</td><td>0.516</td><td>0.836</td><td>0.699</td><td>0.506</td><td>0.507</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.281</td><td>0.334</td><td>0.277</td><td>0.327</td><td>0.287</td><td>0.325</td><td>0.297</td><td>0.349</td><td>0.340</td><td>0.374</td><td>0.302</td><td>0.348</td><td>0.745</td><td>0.584</td><td>0.400</td><td>0.440</td><td>0.333</td><td>0.387</td><td>0.707</td><td>0.621</td><td>0.358</td><td>0.397</td></tr><tr><td>192</td><td>0.340</td><td>0.373</td><td>0.340</td><td>0.374</td><td>0.347</td><td>0.367</td><td>0.380</td><td>0.400</td><td>0.402</td><td>0.414</td><td>0.388</td><td>0.400</td><td>0.877</td><td>0.656</td><td>0.528</td><td>0.509</td><td>0.477</td><td>0.476</td><td>0.860</td><td>0.689</td><td>0.429</td><td>0.439</td></tr><tr><td>336</td><td>0.362</td><td>0.393</td><td>0.371</td><td>0.401</td><td>0.377</td><td>0.393</td><td>0.428</td><td>0.432</td><td>0.452</td><td>0.541</td><td>0.426</td><td>0.433</td><td>1.043</td><td>0.731</td><td>0.643</td><td>0.571</td><td>0.594</td><td>0.541</td><td>1.000</td><td>0.744</td><td>0.496</td><td>0.487</td></tr><tr><td>720</td><td>0.380</td><td>0.416</td><td>0.394</td><td>0.426</td><td>0.404</td><td>0.421</td><td>0.427</td><td>0.445</td><td>0.462</td><td>0.657</td><td>0.431</td><td>0.446</td><td>1.104</td><td>0.763</td><td>0.874</td><td>0.679</td><td>0.831</td><td>0.657</td><td>1.249</td><td>0.838</td><td>0.463</td><td>0.474</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.404</td><td>0.383</td><td>0.335</td><td>0.360</td><td>0.353</td><td>0.363</td><td>0.334</td><td>0.368</td><td>0.338</td><td>0.375</td><td>0.329</td><td>0.367</td><td>0.404</td><td>0.426</td><td>0.364</td><td>0.387</td><td>0.345</td><td>0.372</td><td>0.418</td><td>0.438</td><td>0.379</td><td>0.419</td></tr><tr><td>192</td><td>0.435</td><td>0.402</td><td>0.366</td><td>0.379</td><td>0.376</td><td>0.380</td><td>0.377</td><td>0.391</td><td>0.374</td><td>0.387</td><td>0.367</td><td>0.385</td><td>0.450</td><td>0.451</td><td>0.398</td><td>0.404</td><td>0.380</td><td>0.389</td><td>0.439</td><td>0.450</td><td>0.426</td><td>0.441</td></tr><tr><td>336</td><td>0.462</td><td>0.416</td><td>0.391</td><td>0.394</td><td>0.399</td><td>0.395</td><td>0.426</td><td>0.420</td><td>0.410</td><td>0.411</td><td>0.399</td><td>0.410</td><td>0.532</td><td>0.515</td><td>0.428</td><td>0.425</td><td>0.413</td><td>0.413</td><td>0.490</td><td>0.485</td><td>0.445</td><td>0.459</td></tr><tr><td>720</td><td>0.490</td><td>0.437</td><td>0.434</td><td>0.419</td><td>0.432</td><td>0.417</td><td>0.491</td><td>0.459</td><td>0.478</td><td>0.450</td><td>0.454</td><td>0.439</td><td>0.666</td><td>0.589</td><td>0.487</td><td>0.461</td><td>0.474</td><td>0.453</td><td>0.595</td><td>0.550</td><td>0.543</td><td>0.490</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.205</td><td>0.282</td><td>0.195</td><td>0.269</td><td>0.189</td><td>0.260</td><td>0.180</td><td>0.264</td><td>0.187</td><td>0.267</td><td>0.175</td><td>0.259</td><td>0.287</td><td>0.366</td><td>0.207</td><td>0.305</td><td>0.193</td><td>0.292</td><td>0.286</td><td>0.377</td><td>0.203</td><td>0.287</td></tr><tr><td>192</td><td>0.261</td><td>0.318</td><td>0.247</td><td>0.303</td><td>0.247</td><td>0.300</td><td>0.250</td><td>0.309</td><td>0.249</td><td>0.309</td><td>0.241</td><td>0.302</td><td>0.414</td><td>0.492</td><td>0.290</td><td>0.364</td><td>0.284</td><td>0.362</td><td>0.399</td><td>0.445</td><td>0.269</td><td>0.328</td></tr><tr><td>336</td><td>0.319</td><td>0.355</td><td>0.291</td><td>0.333</td><td>0.295</td><td>0.334</td><td>0.311</td><td>0.348</td><td>0.321</td><td>0.351</td><td>0.305</td><td>0.343</td><td>0.597</td><td>0.542</td><td>0.377</td><td>0.422</td><td>0.369</td><td>0.427</td><td>0.637</td><td>0.591</td><td>0.325</td><td>0.366</td></tr><tr><td>720</td><td>0.415</td><td>0.410</td><td>0.355</td><td>0.377</td><td>0.372</td><td>0.386</td><td>0.412</td><td>0.407</td><td>0.408</td><td>0.403</td><td>0.402</td><td>0.400</td><td>1.730</td><td>1.042</td><td>0.558</td><td>0.524</td><td>0.554</td><td>0.522</td><td>0.960</td><td>0.735</td><td>0.421</td><td>0.415</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.205</td><td>0.299</td><td>0.158</td><td>0.248</td><td>0.152</td><td>0.242</td><td>0.148</td><td>0.240</td><td>0.168</td><td>0.272</td><td>0.195</td><td>0.285</td><td>0.219</td><td>0.314</td><td>0.237</td><td>0.329</td><td>0.197</td><td>0.282</td><td>0.247</td><td>0.345</td><td>0.193</td><td>0.308</td></tr><tr><td>192</td><td>0.220</td><td>0.310</td><td>0.174</td><td>0.263</td><td>0.171</td><td>0.259</td><td>0.162</td><td>0.253</td><td>0.184</td><td>0.289</td><td>0.199</td><td>0.289</td><td>0.231</td><td>0.322</td><td>0.236</td><td>0.330</td><td>0.196</td><td>0.285</td><td>0.257</td><td>0.355</td><td>0.201</td><td>0.315</td></tr><tr><td>336</td><td>0.236</td><td>0.323</td><td>0.191</td><td>0.278</td><td>0.192</td><td>0.278</td><td>0.178</td><td>0.269</td><td>0.198</td><td>0.300</td><td>0.215</td><td>0.305</td><td>0.246</td><td>0.337</td><td>0.249</td><td>0.344</td><td>0.209</td><td>0.301</td><td>0.269</td><td>0.369</td><td>0.214</td><td>0.329</td></tr><tr><td>720</td><td>0.270</td><td>0.347</td><td>0.229</td><td>0.307</td><td>0.236</td><td>0.313</td><td>0.225</td><td>0.317</td><td>0.220</td><td>0.320</td><td>0.256</td><td>0.337</td><td>0.280</td><td>0.363</td><td>0.284</td><td>0.373</td><td>0.245</td><td>0.333</td><td>0.299</td><td>0.390</td><td>0.246</td><td>0.355</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.173</td><td>0.212</td><td>0.167</td><td>0.203</td><td>0.177</td><td>0.208</td><td>0.174</td><td>0.214</td><td>0.172</td><td>0.220</td><td>0.177</td><td>0.218</td><td>0.158</td><td>0.230</td><td>0.202</td><td>0.261</td><td>0.196</td><td>0.255</td><td>0.221</td><td>0.306</td><td>0.217</td><td>0.296</td></tr><tr><td>192</td><td>0.216</td><td>0.250</td><td>0.209</td><td>0.241</td><td>0.219</td><td>0.249</td><td>0.221</td><td>0.254</td><td>0.219</td><td>0.261</td><td>0.225</td><td>0.259</td><td>0.206</td><td>0.277</td><td>0.242</td><td>0.298</td><td>0.237</td><td>0.296</td><td>0.261</td><td>0.340</td><td>0.276</td><td>0.336</td></tr><tr><td>336</td><td>0.260</td><td>0.282</td><td>0.256</td><td>0.276</td><td>0.277</td><td>0.292</td><td>0.278</td><td>0.296</td><td>0.280</td><td>0.306</td><td>0.278</td><td>0.297</td><td>0.272</td><td>0.335</td><td>0.287</td><td>0.335</td><td>0.283</td><td>0.335</td><td>0.309</td><td>0.378</td><td>0.339</td><td>0.380</td></tr><tr><td>720</td><td>0.320</td><td>0.322</td><td>0.321</td><td>0.323</td><td>0.365</td><td>0.350</td><td>0.358</td><td>0.349</td><td>0.365</td><td>0.359</td><td>0.354</td><td>0.348</td><td>0.398</td><td>0.418</td><td>0.351</td><td>0.386</td><td>0.345</td><td>0.381</td><td>0.377</td><td>0.427</td><td>0.403</td><td>0.428</td></tr></table>

## D.4. Computation Costs


Table 23. Computational cost in terms of seconds of various models in terms of seconds for inference for a batch size of 32. “(32)” for MOIRAI refers to patch size.


<table><tr><td rowspan="2"></td><td colspan="5">Context Length</td><td colspan="5">Prediction Length</td></tr><tr><td>1000</td><td>2000</td><td>3000</td><td>4000</td><td>5000</td><td>1000</td><td>2000</td><td>3000</td><td>4000</td><td>5000</td></tr><tr><td><eq>MOIRAI_{Small}</eq> (32)</td><td>0.03</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.07</td><td>0.03</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.07</td></tr><tr><td><eq>MOIRAI_{Base}</eq> (32)</td><td>0.05</td><td>0.06</td><td>0.08</td><td>0.11</td><td>0.13</td><td>0.05</td><td>0.06</td><td>0.08</td><td>0.11</td><td>0.13</td></tr><tr><td><eq>MOIRAI_{Large}</eq> (32)</td><td>0.09</td><td>0.14</td><td>0.19</td><td>0.25</td><td>0.3</td><td>0.09</td><td>0.14</td><td>0.19</td><td>0.25</td><td>0.3</td></tr><tr><td>PatchTST</td><td>0.01</td><td>0.02</td><td>0.02</td><td>0.03</td><td>0.04</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.02</td></tr><tr><td>TiDE</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>TFT</td><td>0.02</td><td>0.04</td><td>0.06</td><td>0.08</td><td>0.09</td><td>0.03</td><td>0.07</td><td>0.12</td><td>0.17</td><td>OOM</td></tr><tr><td>DeepAR</td><td>0.26</td><td>0.32</td><td>0.37</td><td>0.43</td><td>0.49</td><td>2.02</td><td>4.06</td><td>6.1</td><td>8.17</td><td>10.24</td></tr></table>

We perform an analysis on the computation cost of MOIRAI compared to other deep learning based models, while varying the context and prediction lengths. Overall, given the same model size and setting, the cost of inference compared to other deep learning models would be similar. From an architecture perspective, MOIRAI has the following benefits: 

• Patch based inputs: This decreases the computation cost significantly by reducing the number of input tokens. 

• Masked encoder architecture: Unlike decoder-only Transformers, the masked encoder architecture can make multi step predictions in a single forward pass. For decoder-only Transformers and RNNs, they need to autoregressively make predictions, making multiple forward passes for a multi step forecast. For long horizons, this can be quite costly. 

Furthermore, compared to standard baselines, MOIRAI performs zero-shot forecasting. The standard baseline approach has to be trained (multiple times with hyperparameter tuning) for each dataset, leading to increased costs. As MOIRAI continues to be utilized on new datasets, the pre-training costs are amortized and only becomes cheaper, while standard approaches need to be trained over and over again on new datasets. We note that while MOIRAI indeed incurs increased costs due to model size, inference is still highly competitive, taking under 1 second to construct forecasts even with extremely long context/prediction lengths. 

## E. Forecast Visualizations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/a4abf6607f577d119282fdeec27eda6b579b08cf17f4d325842fc7f7f2901662.jpg)



(a) ETTh1-1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/53bb556f950c2bad14d08c0f1da7a5bc1bb7bcb4ee90546eaa173fdcbd2fb989.jpg)



(b) ETTh1-2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/44e0251295a0836bc6a1b1df4da688f0e09c026de30b8dc7c74012c6419786aa.jpg)



(c) ETTm1-1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/abf8e6f207832e3833e901be54248d6fca907bf32d06d769ab5e9c27c224adf4.jpg)



(d) ETTm1-2



Figure 8. Visualizations of zero-shot forecasts from MOIRAI<sub>Base</sub> on ETTh1 and ETTm1 datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/b489239e3b6dbf77f39df23d0b98499e8ccdcc80b2fdbe716b2465692dc4ef48.jpg)



(a) Istanbul Traffic-1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/6c1fb1b165eff63b2b033a16b051ab54f54e3678d54a0e2ed4eb6bdb921aa608.jpg)



(b) Istanbul Traffic-2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/7c49b5f53b5701f2a94c250bec733d1b2e7613e69850510db9a7ef6a8c6b1dee.jpg)



(c) Turkey Power-1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-06/f5d7815c-d86c-4a73-a678-499359e8478a/8c7f0811588fadd87372495c31ca47c4c86bbde3e31128a0f56595e340078108.jpg)



(d) Turkey Power-2



Figure 9. Visualizations of zero-shot forecasts from MOIRAI<sub>Base</sub> on Istanbul Traffic and Turkey Power datasets.
