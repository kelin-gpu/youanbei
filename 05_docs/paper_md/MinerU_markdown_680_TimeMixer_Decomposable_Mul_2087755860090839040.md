# TIMEMIXER: DECOMPOSABLE MULTISCALE MIXING FOR TIME SERIES FORECASTING

Shiyu Wang<sup>1∗</sup>, Haixu Wu<sup>2∗</sup>, Xiaoming Shi<sup>1</sup>, Tengge Hu<sup>2</sup>, Huakun Luo<sup>2</sup>, Lintao Ma<sup>1B</sup>, James Y. Zhang<sup>1</sup>, Jun Zhou<sup>1B</sup> 

<sup>1</sup>Ant Group, Hangzhou, China <sup>2</sup>Tsinghua University, Beijing, China 

{weiming.wsy,lintao.mlt,peter.sxm,james.z,jun.zhoujun}@antgroup.com, {wuhx23,htg21,luohk19}@mails.tsinghua.edu.cn 

## ABSTRACT

Time series forecasting is widely used in extensive applications, such as traffic planning and weather forecasting. However, real-world time series usually present intricate temporal variations, making forecasting extremely challenging. Going beyond the mainstream paradigms of plain decomposition and multiperiodicity analysis, we analyze temporal variations in a novel view of multiscale-mixing, which is based on an intuitive but important observation that time series present distinct patterns in different sampling scales. The microscopic and the macroscopic information are reflected in fine and coarse scales respectively, and thereby complex variations can be inherently disentangled. Based on this observation, we propose TimeMixer as a fully MLP-based architecture with Past-Decomposable-Mixing (PDM) and Future-Multipredictor-Mixing (FMM) blocks to take full advantage of disentangled multiscale series in both past extraction and future prediction phases. Concretely, PDM applies the decomposition to multiscale series and further mixes the decomposed seasonal and trend components in fine-to-coarse and coarse-to-fine directions separately, which successively aggregates the microscopic seasonal and macroscopic trend information. FMM further ensembles multiple predictors to utilize complementary forecasting capabilities in multiscale observations. Consequently, TimeMixer is able to achieve consistent state-of-the-art performances in both long-term and short-term forecasting tasks with favorable run-time efficiency. 

## 1 INTRODUCTION

Time series forecasting has been studied with immense interest in extensive applications, such as economics (Granger & Newbold, 2014), energy (Martín et al., 2010; Qian et al., 2019), traffic planning (Chen et al., 2001; Yin et al., 2021) and weather prediction (Wu et al., 2023b), which is to predict future temporal variations based on past observations of time series (Wu et al., 2023a). However, due to the complex and non-stationary nature of the real world or systems, the observed series usually present intricate temporal patterns, where the multitudinous variations, such as increasing, decreasing, and fluctuating, are deeply mixed, bringing severe challenges to the forecasting task. 

Recently, deep models have achieved promising progress in time series forecasting. The representative models capture temporal variations with well-designed architectures, which span a wide range of foundation backbones, including CNN (Wang et al., 2023; Wu et al., 2023a; Hewage et al., 2020), RNN (Lai et al., 2018; Qin et al., 2017; Salinas et al., 2020), Transformer (Vaswani et al., 2017; Zhou et al., 2021; Wu et al., 2021; Zhou et al., 2022b; Nie et al., 2023) and MLP (Zeng et al., 2023; Zhang et al., 2022; Oreshkin et al., 2019; Challu et al., 2023). In the development of elaborative model architectures, to tackle intricate temporal patterns, some special designs are also involved in these deep models. The widely-acknowledged paradigms primarily include series decomposition and multiperiodicity analysis. As a classical time series analysis technology, decomposition is introduced to deep models as a basic module by (Wu et al., 2021), which decomposes the complex temporal patterns into more predictable components, such as seasonal and trend, and thereby benefiting the forecasting process (Zeng et al., 2023; Zhou et al., 2022b; Wang et al., 2023). Furthermore, multiperiodicity analysis is also involved in time series forecasting (Wu et al., 2023a; Zhou et al., 2022a) to disentangle mixed temporal variations into multiple components with different period lengths. Empowered with these designs, deep models are able to highlight inherent properties of time series from tanglesome variations and further boost the forecasting performance. 

Going beyond the above mentioned designs, we further observe that time series present distinct temporal variations in different sampling scales, e.g., the hourly recorded traffic flow presents traffic changes at different times of the day, while for the daily sampled series, these fine-grained variations disappear but fluctuations associated with holidays emerge. On the other hand, the trend of macroeconomics dominates the yearly averaged patterns. These observations naturally call for a multiscale analysis paradigm to disentangle complex temporal variations, where fine and coarse scales can reflect the micro- and the macro-scopic information respectively. Especially for the time series forecasting task, it is also notable that the future variation is jointly determined by the variations in multiple scales. Therefore, in this paper, we attempt to design the forecasting model from a novel view of multiscale-mixing, which is able to take advantage of both disentangled variations and complementary forecasting capabilities from multiscale series simultaneously. 

Technically, we propose TimeMixer with a multiscale mixing architecture that is able to extract essential information from past variations by Past-Decomposable-Mixing (PDM) blocks and then predicts the future series by the Future-Multipredictor-Mixing (FMM) block. Concretely, TimeMixer first generates multiscale observations through average downsampling. Next, PDM adopts a decom posable design to better cope with distinct properties of seasonal and trend variations, by mixing decomposed multiscale seasonal and trend components in fine-to-coarse and coarse-to-fine directions separately. With our novel design, PDM is able to successfully aggregate the detailed seasonal information starting from the finest series and dive into macroscopic trend components along with the knowledge from coarser scales. In the forecasting phase, FMM ensembles multiple predictors to utilize complementary forecasting capabilities from multiscale observations. With our meticulous architecture, TimeMixer achieves the consistent state-of-the-art performances in both long-term and short-term forecasting tasks with superior efficiency across all of our experiments, covering extensive well-established benchmarks. Our contributions are summarized as follows: 

• Going beyond previous methods, we tackle intricate temporal variations in series forecasting from a novel view of multiscale mixing, taking advantage of disentangled variations and complementary forecasting capabilities from multiscale series simultaneously. 

• We propose TimeMixer as a simple but effective forecasting model, which enables the combination of the multiscale information in both history extraction and future prediction phases, empowered by our tailored decomposable and multiple-predictor mixing technique. 

• TimeMixer achieves consistent state-of-the-art in performances in both long-term and short term forecasting tasks with superior efficiency on a wide range of benchmarks. 

## 2 RELATED WORK

## 2.1 TEMPORAL MODELING IN DEEP TIME SERIES FORECASTING

As the key problem in time series analysis (Wu et al., 2023a), temporal modeling has been widely explored. According to foundation backbones, deep models can be roughly categorized into the following four paradigms: RNN-, CNN-, Transformer- and MLP-based methods. Typically, CNNbased models employ the convolution kernels along the time dimension to capture temporal patterns (Wang et al., 2023; Hewage et al., 2020). And RNN-based methods adopt the recurrent structure to model the temporal state transition (Lai et al., 2018; Zhao et al., 2017). However, both RNNand CNN-based methods suffer from the limited receptive field, limiting the long-term forecasting capability. Recently, benefiting from the global modeling capacity, Transformer-based models have been widely-acknowledged in long-term series forecasting (Zhou et al., 2021; Wu et al., 2021; Liu et al., 2022b; Kitaev et al., 2020; Nie et al., 2023), which can capture the long-term temporal dependencies adaptively with attention mechanism. Furthermore, multiple layer projection (MLP) is also introduced to time series forecasting (Oreshkin et al., 2019; Challu et al., 2023; Zeng et al., 2023), which achieves favourable performance in both forecasting performance and efficiency. 

Additionally, several specific designs are proposed to better capture intricate temporal patterns, including series decomposition and multi-periodicity analysis. Firstly, for the series decomposition, 

Autoformer (Wu et al., 2021) presents the series decomposition block based on moving average to decompose complex temporal variations into seasonal and trend components. Afterwards, FEDformer (Zhou et al., 2022b) enhances the series decomposition block with multiple kernels moving average. DLinear (Zeng et al., 2023) utilizes the series decomposition as the pre-processing before linear regression. MICN (Wang et al., 2023) also decomposes input series into seasonal and trend terms, and then integrates the global and local context for forecasting. As for the multi-periodicity analysis, N-BEATS (Oreshkin et al., 2019) fits the time series with multiple trigonometric basis functions. FiLM (Zhou et al., 2022a) projects time series into Legendre Polynomials space, where different basis functions correspond to different period components in the original series. Recently, TimesNet (Wu et al., 2023a) adopts Fourier Transform to map time series into multiple components with different period lengths and presents a modular architecture to process decomposed components. 

Unlike the designs mentioned above, this paper explores the multiscale mixing architecture in time series forecasting. Although there exist some models with temporal multiscale designs, such as Pyraformer (Liu et al., 2021) with pyramidal attention and SCINet (Liu et al., 2022a) with a bifurcate downsampling tree, their future predictions do not make use of the information at different scales extracted from the past observations simultaneously. In TimeMixer, we present a new multiscale mixing architecture with Past-Decomposable-Mixing to utilize the disentangled series for multiscale representation learning and Future-Multipredictor-Mixing to ensemble the complementary forecasting skills of multiscale series for better prediction. 

## 2.2 MIXING NETWORKS

Mixing is an effective way of information integration and has been applied to computer vision and natural language processing. For instance, MLP-Mixer (Tolstikhin et al., 2021) designs a two-stage mixing structure for image recognition, which mixes the channel information and patch information successively with linear layers. FNet (Lee-Thorp et al., 2022) replaces attention layers in Transformer with simple Fourier Transform, achieving the efficient token mixing of a sentence. In this paper, we further explore the mixing structure in time series forecasting. Unlike previous designs, TimeMixer presents a decomposable multi-scale mixing architecture and distinguishes the mixing methods in both past information extraction and future prediction phases. 

## 3 TIMEMIXER

Given a series x with one or multiple observed variates, the main objective of time series forecasting is to utilize past observations (length-P) to obtain the most probable future prediction (length-F). As mentioned above, the key challenge of accurate forecasting is to tackle intricate temporal variations. In this paper, we propose TimeMixer of multiscale-mixing, benefiting from disentangled variations and complementary forecasting capabilities from multiscale series. Technically, TimeMixer consists of a multiscale mixing architecture with Past-Decomposable-Mixing and Future-Multipredictor-Mixing for past information extraction and future prediction respectively. 

## 3.1 MULTISCALE MIXING ARCHITECTURE

Time series of different scales naturally exhibit distinct properties, where fine scales mainly depict detailed patterns and coarse scales highlight macroscopic variations (Mozer, 1991). This multiscale view can inherently disentangle intricate variations in multiple components, thereby benefiting temporal variation modeling. It is also notable that, especially for the forecasting task, multiscale time series present different forecasting capabilities, due to their distinct dominating temporal patterns (Ferreira et al., 2006). Therefore, we present TimeMixer in a multiscale mixing architecture to utilize multiscale series with distinguishing designs for past extraction and future prediction phases. 

As shown in Figure 1, to disentangle complex variations, we first downsample the past observations $\mathbf { x } \in \mathbb { R } ^ { P \times C }$ into M scales by average pooling and finally obtain a set of multiscale time series $\mathcal { X } = \{ \mathbf { x } _ { 0 } , \cdots , \mathbf { x } _ { M } \}$ , where $\mathbf { x } _ { m } \in \mathbb { R } ^ { \lfloor \frac { P } { 2 ^ { m } } \rfloor \times C } , m \in \{ 0 , \cdots , M \}$ , C denotes the variate number. The lowest level series $\mathbf { x } _ { 0 } = \mathbf { x }$ is the input series, which contains the finest temporal variations, while the highest-level series $\mathbf { x } _ { M }$ is for the macroscopic variations. Then we project these multiscale series into deep features $\mathcal { X } ^ { 0 }$ by the embedding layer, which can be formalized as $\mathcal { X } ^ { 0 } = \mathrm { E m b e d } ( \mathcal { X } )$ . With the above designs, we obtain the multiscale representations of input series. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/b6dac832d0ffc7fbda798179a22eaef46dd1aa5c75120620632ab25743a260c8.jpg)



Figure 1: Overall architecture of TimeMixer, which consists of Past-Decomposable Mixing and Future-Multipredictor-Mixing for past observations and future predictions respectively.


Next, we utilize stacked Past-Decomposable-Mixing (PDM) blocks to mix past information across different scales. For the l-th layer, the input is $\mathcal { X } ^ { l - 1 }$ and the process of PDM can be formalized as: 

$$
\mathcal {X} ^ {l} = \mathrm{PDM} (\mathcal {X} ^ {l - 1}), l \in \{0, \dots , L \},\tag{1}
$$

where L is the total layer and $\mathcal { X } ^ { l } = \{ \mathbf { x } _ { 0 } ^ { l } , \cdot \cdot \cdot , \mathbf { x } _ { M } ^ { l } \} , \mathbf { x } _ { m } ^ { l } \in \mathbb { R } ^ { \lfloor \frac { P } { 2 ^ { m } } \rfloor }$ <sup>⌋×dmodel</sup> denotes the mixed past representations with $d _ { \mathrm { m o d e l } }$ channels. More details of PDM are described in the next section. 

As for the future prediction phase, we adopt the Future-Multipredictor-Mixing (FMM) block to ensemble extracted multiscale past information $\mathcal { X } ^ { L }$ and generate future predictions, which is: 

$$
\widehat {\mathbf {x}} = \operatorname{FMM} (\mathcal {X} ^ {L}),\tag{2}
$$

where $\widehat { \mathbf { x } } \in \mathbb { R } ^ { F \times C }$ represents the final prediction. With the above designs, TimeMixer can successfully capture essential past information from disentangled multiscale observations and predict the future with benefits from multiscale past information. 

## 3.2 PAST DECOMPOSABLE MIXING

We observe that for past observations, due to the complex nature of real-world series, even the coarsest scale series present mixed variations. As shown in Figure 1, the series in the top layer still present clear seasonality and trend simultaneously. It is notable that the seasonal and trend components hold distinct properties in time series analysis (Cleveland et al., 1990), which corresponds to short-term and long-term variations or stationary and non-stationary dynamics respectively. Therefore, instead of directly mixing multiscale series as a whole, we propose the Past-Decomposable-Mixing (PDM) block to mix the decomposed seasonal and trend components in multiple scales separately. 

Concretely, for the l-th PDM block, we first decompose the multiscale time series $\mathcal { X } _ { l }$ into seasonal parts $\mathcal { S } ^ { l } \dot { = } \{ \mathbf { s } _ { 0 } ^ { l } , \cdots , \mathbf { s } _ { M } ^ { l } \}$ and trend parts $\mathcal { T } ^ { l } = \{ \mathbf { t } _ { 0 } ^ { l ^ { \star } } , \cdot \cdot \cdot , \mathbf { t } _ { M } ^ { l } \}$ by series decomposition block from Autoformer (Wu et al., 2021). As the above analyzed, taking the distinct properties of seasonal-trend parts into account, we apply the mixing operation to seasonal and trend terms separately to interact information from multiple scales. Overall, the l-th PDM block can be formalized as: 

$$
\begin{array}{c} \mathbf {s} _ {m} ^ {l}, \mathbf {t} _ {m} ^ {l} = \text {SeriesDecomp} (\mathbf {x} _ {m} ^ {l}), m \in \{0, \dots , M \}, \\ \mathcal {X} ^ {l} = \mathcal {X} ^ {l - 1} + \text {FeedForward} \left(\text {S - Mix} \left(\{\mathbf {s} _ {m} ^ {l} \} _ {m = 0} ^ {M}\right) + \text {T - Mix} \left(\{\mathbf {t} _ {m} ^ {l} \} _ {m = 0} ^ {M}\right)\right), \end{array}\tag{3}
$$

where FeedForward(·) contains two linear layers with intermediate GELU activation function for information interaction among channels, S-Mix(·), T-Mix(·) denote seasonal and trend mixing. 

Seasonal Mixing In seasonality analysis (Box & Jenkins, 1970), larger periods can be seen as the aggregation of smaller periods, such as the weekly period of traffic flow formed by seven daily changes, addressing the importance of detailed information in predicting future seasonal variations. 

Therefore, in seasonal mixing, we adopt the bottom-up approach to incorporate information from the lower-level fine-scale time series upwards, which can supplement detailed information to the seasonality modeling of coarser scales. Technically, for the set of multiscale seasonal parts $\mathcal { S } ^ { l } =$ $\{ \mathbf { s } _ { 0 } ^ { l } , \cdots , \mathbf { s } _ { M } ^ { l } \}$ , we use the Bottom-Up-Mixing layer for the m-th scale in a residual way to achieve bottom-up seasonal information interaction, which can be formalized as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/603e6f883b1f0f0bea627e793b65d5808bc1fa69fc3a21e087e05e41d357919e.jpg)



Figure 2: The temporal linear layer in seasonal mixing (a), trend mixing (b) and future prediction (c).


$$
\mathrm{for} m \colon 1 \to M \mathrm{do:} \quad \mathbf {s} _ {m} ^ {l} = \mathbf {s} _ {m} ^ {l} + \mathrm{Bottom-Up-Mixing} (\mathbf {s} _ {m - 1} ^ {l}).\tag{4}
$$

where $\mathrm { B o t t o m \mathrm { - } U p \mathrm { - } M i x i n g ( \cdot ) }$ is instantiated as two linear layers with an intermediate GELU activation function along the temporal dimension, whose input dimension is $\left\lfloor { \frac { P } { 2 ^ { m - 1 } } } \right\rfloor$ and output dimension is $\lfloor { \frac { P } { 2 ^ { m } } } \rfloor$ . See Figure 2 for an intuitive understanding. 

Trend Mixing Contrary to seasonal parts, for trend items, the detailed variations can introduce noise in capturing macroscopic trend. Note that the upper coarse scale time series can easily provide clear macro information than the lower level. Therefore, we adopt a top-down mixing method to utilize the macro knowledge from coarser scales to guide the trend modeling of finer scales. 

Technically, for multiscale trend components $\mathcal { T } ^ { l } = \{ \mathbf { t } _ { 0 } ^ { l } , \cdots , \mathbf { t } _ { M } ^ { l } \}$ , we adopt the Top-Down-Mixing layer for the m-th scale in a residual way to achieve top-down trend information interaction: 

$$
\mathrm{for} m \colon (M - 1) \to 0 \mathrm{do:} \quad \mathbf {t} _ {m} ^ {l} = \mathbf {t} _ {m} ^ {l} + \mathrm{Top-Down-Mixing} (\mathbf {t} _ {m + 1} ^ {l}),\tag{5}
$$

where Top-Down-Mixing(·) is two linear layers with an intermediate GELU activation function, whose input dimension is $\left\lfloor { \frac { P } { 2 ^ { m + 1 } } } \right\rfloor$ and output dimension is $\lfloor { \frac { P } { 2 ^ { m } } } \rfloor$ as shown in Figure 2. 

Empowered by seasonal and trend mixing, PDM progressively aggregates the detailed seasonal information from fine to coarse and dive into the macroscopic trend information with prior knowledge from coarser scales, eventually achieving the multiscale mixing in past information extraction. 

## 3.3 FUTURE MULTIPREDICTOR MIXING

After L PDM blocks, we obtain the multiscale past information as $\mathcal { X } ^ { L } = \bigl \{ \mathbf { x } _ { 0 } ^ { L } , \cdot \cdot \cdot , \mathbf { x } _ { M } ^ { L } \bigr \} , \mathbf { x } _ { m } ^ { L } \in$ $\mathbb { R } ^ { \lfloor \frac { P } { 2 ^ { m } } \rfloor \times d _ { \mathrm { m o d e l } } }$ . Since the series in different scales presents different dominating variations, their predictions also present different capabilities. To fully utilize the multiscale information, we propose to aggregate predictions from multiscale series and present Future-Multipredictor-Mixing block as: 

$$
\widehat {\mathbf {x}} _ {m} = \mathrm{Predictor} _ {m} (\mathbf {x} _ {m} ^ {L}), m \in \{0, \dots , M \}, \widehat {\mathbf {x}} = \sum_ {m = 0} ^ {M} \widehat {\mathbf {x}} _ {m},\tag{6}
$$

where $\widehat { \mathbf { x } } _ { m } \in \mathbb { R } ^ { F \times C }$ represents the future prediction from the m-th scale series and the final output is $\widehat { \mathbf { x } } \in \mathbb { R } ^ { F \times C }$ . Predictor (·) denotes the predictor of the m-th scale series, which firstly adopts one single linear layer to directly regress length-F future from length- $\cdot \left\lfloor { \frac { P } { 2 ^ { m } } } \right\rfloor$ extracted past information (Figure 2) and then projects deep representations into C variates. Note that FMM is an ensemble of multiple predictors, where different predictors are based on past information from different scales, enabling FMM to integrate complementary forecasting capabilities of mixed multiscale series. 

## 4 EXPERIMENTS

We conduct extensive experiments to evaluate the performance and efficiency of TimeMixer, covering long-term and short-term forecasting, including 18 real-world benchmarks and 15 baselines. The detailed model and experiment configurations are summarized in Appendix A. 


Table 1: Summary of benchmarks. Forecastability is one minus the entropy of Fourier domain.


<table><tr><td>Tasks</td><td>Dataset</td><td>Variate</td><td>Predict Length</td><td>Frequency</td><td>Forecastability</td><td>Information</td></tr><tr><td rowspan="5">Long-term forecasting</td><td>ETT (4 subsets)</td><td>7</td><td>96~720</td><td>15 mins</td><td>0.46</td><td>Temperature</td></tr><tr><td>Weather</td><td>21</td><td>96~720</td><td>10 mins</td><td>0.75</td><td>Weather</td></tr><tr><td>Solar-Energy</td><td>137</td><td>96~720</td><td>10min</td><td>0.33</td><td>Electricity</td></tr><tr><td>Electricity</td><td>321</td><td>96~720</td><td>Hourly</td><td>0.77</td><td>Electricity</td></tr><tr><td>Traffic</td><td>862</td><td>96~720</td><td>Hourly</td><td>0.68</td><td>Transportation</td></tr><tr><td rowspan="2">Short-term forecasting</td><td>PEMS (4 subsets)</td><td>170~883</td><td>12</td><td>5min</td><td>0.55</td><td>Traffic network</td></tr><tr><td>M4 (6 subsets)</td><td>1</td><td>6~48</td><td>Hourly~Yearly</td><td>0.47</td><td>Database</td></tr></table>

Benchmarks For long-term forecasting, we experiment on 8 well-established benchmarks: ETT datasets (including 4 subsets: ETTh1, ETTh2, ETTm1, ETTm2), Weather, Solar-Energy, Electricity, and Traffic following (Zhou et al., 2021; Wu et al., 2021; Liu et al., 2022a). For short-term forecasting, we adopt the PeMS (Chen et al., 2001) which contains four public traffic network datasets (PEMS03, PEMS04, PEMS07, PEMS08), and M4 dataset which involves 100,000 different time series collected in different frequencies. Furthermore, we measure the forecastability (Goerg, 2013) of all datasets. It is observed that ETT, M4, and Solar-Energy exhibit relatively low forecastability, indicating the challenges in these benchmarks. More information is summarized in Table 1. 

Baselines We compare TimeMixer with 15 baselines, which comprise the state-of-the-art long-term forecasting model PatchTST (2023) and advanced short-term forecasting models TimesNet (2023a) and SCINet (2022a), as well as other competitive models including Crossformer (2023), MICN (2023), FiLM (2022a), DLinear (2023), LightTS (2022) ,FEDformer (2022b), Stationary (2022b), Pyraformer (2021), Autoformer (2021), Informer (2021), N-HiTS (2023) and N-BEATS (2019). 

Unified experiment settings Note that experimental results reported by the above mentioned baselines cannot be compared directly due to different choices of input length and hyper-parameter searching strategy. For fairness, we make a great effort to provide two types of experiments. In the main text, we align the input length of all baselines and report results averaged from three repeats (see Appendix C for error bars). In Appendix, to compare the upper bound of models, we also conduct a comprehensive hyperparameter searching and report the best results in Table 14 of Appendix. 

Implementation details All the experiments are implemented in PyTorch (Paszke et al., 2019) and conducted on a single NVIDIA A100 80GB GPU. We utilize the L2 loss for model training. The number of scales M is set according to the time series length to trade off performance and efficiency. 

## 4.1 MAIN RESULTS

Long-term forecasting As shown in Table 2, TimeMixer achieves consistent state-of-the-art performance in all benchmarks, covering a large variety of series with different frequencies, variate numbers and real-world scenarios. Especially, TimeMixer outperforms PatchTST by a considerable margin, with a 9.4% MSE reduction in Weather and a 24.7% MSE reduction in Solar-Energy. It is worth noting that TimeMixer exhibits good performance even for datasets with low forecastability, such as Solar-Energy and ETT, further proving the generality and effectiveness of TimeMixer. 


Table 2: Long-term forecasting results. All the results are averaged from 4 different prediction lengths, that is {96, 192, 336, 720}. A lower MSE or MAE indicates a better prediction. We fix the input length as 96 for all experiments. See Table 13 in Appendix for the full results.


<table><tr><td>Models</td><td colspan="2">TimeMixer (Ours)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">TimesNet (2023a)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">MICN (2023)</td><td colspan="2">FiLM (2022a)</td><td colspan="2">DLinear (2023)</td><td colspan="2">FEDformer (2022b)</td><td colspan="2">Stationary (2022b)</td><td colspan="2">Autoformer (2021)</td><td colspan="2">Informer (2021)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Weather</td><td>0.240</td><td>0.271</td><td>0.265</td><td>0.285</td><td>0.251</td><td>0.294</td><td>0.264</td><td>0.320</td><td>0.268</td><td>0.321</td><td>0.271</td><td>0.291</td><td>0.265</td><td>0.315</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td><td>0.634</td><td>0.548</td></tr><tr><td>Solar-Energy</td><td>0.216</td><td>0.280</td><td>0.287</td><td>0.333</td><td>0.403</td><td>0.374</td><td>0.406</td><td>0.442</td><td>0.283</td><td>0.358</td><td>0.380</td><td>0.371</td><td>0.330</td><td>0.401</td><td>0.328</td><td>0.383</td><td>0.350</td><td>0.390</td><td>0.586</td><td>0.557</td><td>0.331</td><td>0.381</td></tr><tr><td>Electricity</td><td>0.182</td><td>0.272</td><td>0.216</td><td>0.318</td><td>0.193</td><td>0.304</td><td>0.244</td><td>0.334</td><td>0.196</td><td>0.309</td><td>0.223</td><td>0.302</td><td>0.225</td><td>0.319</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td><td>0.311</td><td>0.397</td></tr><tr><td>Traffic</td><td>0.484</td><td>0.297</td><td>0.529</td><td>0.341</td><td>0.620</td><td>0.336</td><td>0.667</td><td>0.426</td><td>0.593</td><td>0.356</td><td>0.637</td><td>0.384</td><td>0.625</td><td>0.383</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td><td>0.764</td><td>0.416</td></tr><tr><td>ETTh1</td><td>0.447</td><td>0.440</td><td>0.516</td><td>0.484</td><td>0.495</td><td>0.450</td><td>0.529</td><td>0.522</td><td>0.475</td><td>0.480</td><td>0.516</td><td>0.483</td><td>0.461</td><td>0.457</td><td>0.498</td><td>0.484</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td><td>1.040</td><td>0.795</td></tr><tr><td>ETTh2</td><td>0.364</td><td>0.395</td><td>0.391</td><td>0.411</td><td>0.414</td><td>0.427</td><td>0.942</td><td>0.684</td><td>0.574</td><td>0.531</td><td>0.402</td><td>0.420</td><td>0.563</td><td>0.519</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td>0.459</td><td>4.431</td><td>1.729</td></tr><tr><td>ETTm1</td><td>0.381</td><td>0.395</td><td>0.406</td><td>0.407</td><td>0.400</td><td>0.406</td><td>0.513</td><td>0.495</td><td>0.423</td><td>0.422</td><td>0.411</td><td>0.402</td><td>0.404</td><td>0.408</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td><td>0.961</td><td>0.734</td></tr><tr><td>ETTm2</td><td>0.275</td><td>0.323</td><td>0.290</td><td>0.334</td><td>0.291</td><td>0.333</td><td>0.757</td><td>0.610</td><td>0.353</td><td>0.402</td><td>0.287</td><td>0.329</td><td>0.354</td><td>0.402</td><td>0.305</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.327</td><td>0.371</td><td>1.410</td><td>0.810</td></tr></table>


Table 3: Short-term forecasting results in the PEMS datasets with multiple variates. All input lengths are 96 and prediction lengths are 12. A lower MAE, MAPE or RMSE indicates a better prediction.


<table><tr><td colspan="2">Models</td><td>TimeMixer (Ours)</td><td>SCINet (2022a)</td><td>Crossformer (2023)</td><td>PatchTST (2023)</td><td>TimesNet (2023a)</td><td>MICN (2023)</td><td>FiLM (2022a)</td><td>DLinear (2023)</td><td>FEDformer (2022b)</td><td>Stationary (2022b)</td><td>Autoformer (2021)</td><td>Informer (2021)</td></tr><tr><td rowspan="3">PEMS03</td><td>MAE</td><td>14.63</td><td>15.97</td><td><eq>\underline{15.64}</eq></td><td>18.95</td><td>16.41</td><td>15.71</td><td>21.36</td><td>19.70</td><td>19.00</td><td>17.64</td><td>18.08</td><td>19.19</td></tr><tr><td>MAPE</td><td>14.54</td><td>15.89</td><td>15.74</td><td>17.29</td><td><eq>\underline{15.17}</eq></td><td>15.67</td><td>18.35</td><td>18.35</td><td>18.57</td><td>17.56</td><td>18.75</td><td>19.58</td></tr><tr><td>RMSE</td><td>23.28</td><td><eq>\underline{25.20}</eq></td><td>25.56</td><td>30.15</td><td>26.72</td><td>24.55</td><td>35.07</td><td>32.35</td><td>30.05</td><td>28.37</td><td>27.82</td><td>32.70</td></tr><tr><td rowspan="3">PEMS04</td><td>MAE</td><td>19.21</td><td><eq>\underline{20.35}</eq></td><td>20.38</td><td>24.86</td><td>21.63</td><td>21.62</td><td>26.74</td><td>24.62</td><td>26.51</td><td>22.34</td><td>25.00</td><td>22.05</td></tr><tr><td>MAPE</td><td>12.53</td><td><eq>\underline{12.84}</eq></td><td><eq>\underline{12.84}</eq></td><td>16.65</td><td>13.15</td><td>13.53</td><td>16.46</td><td>16.12</td><td>16.76</td><td>14.85</td><td>16.70</td><td>14.88</td></tr><tr><td>RMSE</td><td>30.92</td><td><eq>\underline{32.31}</eq></td><td>32.41</td><td>40.46</td><td>34.90</td><td>34.39</td><td>42.86</td><td>39.51</td><td>41.81</td><td>35.47</td><td>38.02</td><td>36.20</td></tr><tr><td rowspan="3">PEMS07</td><td>MAE</td><td>20.57</td><td>22.79</td><td><eq>\underline{22.54}</eq></td><td>27.87</td><td>25.12</td><td>22.28</td><td>28.76</td><td>28.65</td><td>27.92</td><td>26.02</td><td>26.92</td><td>27.26</td></tr><tr><td>MAPE</td><td>8.62</td><td>9.41</td><td><eq>\underline{9.38}</eq></td><td>12.69</td><td>10.60</td><td>9.57</td><td>11.21</td><td>12.15</td><td>12.29</td><td>11.75</td><td>11.83</td><td>11.63</td></tr><tr><td>RMSE</td><td>33.59</td><td>35.61</td><td>35.49</td><td>42.56</td><td>40.71</td><td><eq>\underline{35.40}</eq></td><td>45.85</td><td>45.02</td><td>42.29</td><td>42.34</td><td>40.60</td><td>45.81</td></tr><tr><td rowspan="3">PEMS08</td><td>MAE</td><td>15.22</td><td><eq>\underline{17.38}</eq></td><td>17.56</td><td>20.35</td><td>19.01</td><td>17.76</td><td>22.11</td><td>20.26</td><td>20.56</td><td>19.29</td><td>20.47</td><td>20.96</td></tr><tr><td>MAPE</td><td>9.67</td><td>10.80</td><td>10.92</td><td>13.15</td><td>11.83</td><td><eq>\underline{10.76}</eq></td><td>12.81</td><td>12.09</td><td>12.41</td><td>12.21</td><td>12.27</td><td>13.20</td></tr><tr><td>RMSE</td><td>24.26</td><td>27.34</td><td><eq>\underline{27.21}</eq></td><td>31.04</td><td>30.65</td><td>27.26</td><td>35.13</td><td>32.38</td><td>32.97</td><td>38.62</td><td>31.52</td><td>30.61</td></tr></table>


Table 4: Short-term forecasting results in the M4 dataset with a single variate. All prediction lengths are in [6, 48]. A lower SMAPE, MASE or OWA indicates a better prediction. ∗. in the Transformers indicates the name of ∗former. Stationary means the Non-stationary Transformer.


<table><tr><td colspan="2">Models</td><td>TimeMixer (Ours)</td><td>TimesNet (2023a)</td><td>N-HiTS (2023)</td><td>N-BEATS* (2019)</td><td>SCINet (2022a)</td><td>PatchTST (2023)</td><td>MICN (2023)</td><td>FiLM (2022a)</td><td>LightTS (2022)</td><td>DLinear (2023)</td><td>FED. (2022b)</td><td>Stationary (2022b)</td><td>Auto. (2021)</td><td>Pyra. (2021)</td><td>In. (2021)</td></tr><tr><td rowspan="3">Yearly</td><td>SMAPE</td><td>13.206</td><td>13.387</td><td>13.418</td><td>13.436</td><td>18.605</td><td>16.463</td><td>25.022</td><td>17.431</td><td>14.247</td><td>16.965</td><td>13.728</td><td>13.717</td><td>13.974</td><td>15.530</td><td>14.727</td></tr><tr><td>MASE</td><td>2.916</td><td>2.996</td><td>3.045</td><td>3.043</td><td>4.471</td><td>3.967</td><td>7.162</td><td>4.043</td><td>3.109</td><td>4.283</td><td>3.048</td><td>3.078</td><td>3.134</td><td>3.711</td><td>3.418</td></tr><tr><td>OWA</td><td>0.776</td><td>0.786</td><td>0.793</td><td>0.794</td><td>1.132</td><td>1.003</td><td>1.667</td><td>1.042</td><td>0.827</td><td>1.058</td><td>0.803</td><td>0.807</td><td>0.822</td><td>0.942</td><td>0.881</td></tr><tr><td rowspan="3">Quarterly</td><td>SMAPE</td><td>9.996</td><td>10.100</td><td>10.202</td><td>10.124</td><td>14.871</td><td>10.644</td><td>15.214</td><td>12.925</td><td>11.364</td><td>12.145</td><td>10.792</td><td>10.958</td><td>11.338</td><td>15.449</td><td>11.360</td></tr><tr><td>MASE</td><td>1.166</td><td>1.182</td><td>1.194</td><td>1.169</td><td>2.054</td><td>1.278</td><td>1.963</td><td>1.664</td><td>1.328</td><td>1.520</td><td>1.283</td><td>1.325</td><td>1.365</td><td>2.350</td><td>1.401</td></tr><tr><td>OWA</td><td>0.825</td><td>0.890</td><td>0.899</td><td>0.886</td><td>1.424</td><td>0.949</td><td>1.407</td><td>1.193</td><td>1.000</td><td>1.106</td><td>0.958</td><td>0.981</td><td>1.012</td><td>1.558</td><td>1.027</td></tr><tr><td rowspan="3">Monthly</td><td>SMAPE</td><td>12.605</td><td>12.670</td><td>12.791</td><td>12.677</td><td>14.925</td><td>13.399</td><td>16.943</td><td>15.407</td><td>14.014</td><td>13.514</td><td>14.260</td><td>13.917</td><td>13.958</td><td>17.642</td><td>14.062</td></tr><tr><td>MASE</td><td>0.919</td><td>0.933</td><td>0.969</td><td>0.937</td><td>1.131</td><td>1.031</td><td>1.442</td><td>1.298</td><td>1.053</td><td>1.037</td><td>1.102</td><td>1.097</td><td>1.103</td><td>1.913</td><td>1.141</td></tr><tr><td>OWA</td><td>0.869</td><td>0.878</td><td>0.899</td><td>0.880</td><td>1.027</td><td>0.949</td><td>1.265</td><td>1.144</td><td>0.981</td><td>0.956</td><td>1.012</td><td>0.998</td><td>1.002</td><td>1.511</td><td>1.024</td></tr><tr><td rowspan="3">Others</td><td>SMAPE</td><td>4.564</td><td>4.891</td><td>5.061</td><td>4.925</td><td>16.655</td><td>6.558</td><td>41.985</td><td>7.134</td><td>15.880</td><td>6.709</td><td>4.954</td><td>6.302</td><td>5.485</td><td>24.786</td><td>24.460</td></tr><tr><td>MASE</td><td>3.115</td><td>3.302</td><td>3.216</td><td>3.391</td><td>15.034</td><td>4.511</td><td>62.734</td><td>5.09</td><td>11.434</td><td>4.953</td><td>3.264</td><td>4.064</td><td>3.865</td><td>18.581</td><td>20.960</td></tr><tr><td>OWA</td><td>0.982</td><td>1.035</td><td>1.040</td><td>1.053</td><td>4.123</td><td>1.401</td><td>14.313</td><td>1.553</td><td>3.474</td><td>1.487</td><td>1.036</td><td>1.304</td><td>1.187</td><td>5.538</td><td>5.879</td></tr><tr><td rowspan="3">Weighted Average</td><td>SMAPE</td><td>11.723</td><td>11.829</td><td>11.927</td><td>11.851</td><td>15.542</td><td>13.152</td><td>19.638</td><td>14.863</td><td>13.525</td><td>13.639</td><td>12.840</td><td>12.780</td><td>12.909</td><td>16.987</td><td>14.086</td></tr><tr><td>MASE</td><td>1.559</td><td>1.585</td><td>1.613</td><td>1.559</td><td>2.816</td><td>1.945</td><td>5.947</td><td>2.207</td><td>2.111</td><td>2.095</td><td>1.701</td><td>1.756</td><td>1.771</td><td>3.265</td><td>2.718</td></tr><tr><td>OWA</td><td>0.840</td><td>0.851</td><td>0.861</td><td>0.855</td><td>1.309</td><td>0.998</td><td>2.279</td><td>1.125</td><td>1.051</td><td>1.051</td><td>0.918</td><td>0.930</td><td>0.939</td><td>1.480</td><td>1.230</td></tr></table>


∗ The original paper of N-BEATS (2019) adopts a special ensemble method to promote the performance. For fair comparisons, we remove the ensemble and only compare the pure forecasting models.


Short-term forecasting TimeMixer also shows great performance in short-term forecasting under both multivariate and univariate settings (Table 3-4). For PeMS benchmarks that record multiple time series of citywide traffic networks, due to the complex spatiotemporal correlations among multiple variates, many advanced models degenerate a lot in this task, such as PatchTST (2023) and DLinear (2023), which adopt the channel independence design. In contrast, TimeMixer still performs favourablely in this challenging problem, verifying its effectiveness in handling complex multivariate time series forecasting. As for the M4 dataset for univariate forecasting, it contains various temporal variations under different sampling frequencies, including hourly, daily, weekly, monthly, quarterly, and yearly, which exhibits low predictability and distinctive characteristics across different frequencies. Remarkably, Timemixer consistently performs best across all frequencies, affirming the multiscale mixing architecture’s capacity in modeling complex temporal variations. 

## 4.2 MODEL ANALYSIS

Ablations To verify the effectiveness of each component of TimeMixer, we provide detailed ablation study on every possible design in both Past-Decomposable-Mixing and Future-Multipredictor-Mixing blocks on all 18 experiment benchmarks. From Table 5, we have the following observations. 

The exclusion of Future-Multipredictor-Mixing in ablation ② results in a significant decrease in the model’s forecasting accuracy for both short and long-term predictions. This demonstrates that mixing future predictions from multiscale series can effectively boost the model performance. 

For the past mixing, we verify the effectiveness by removing or replacing components gradually. In ablations ③ and ④ that remove seasonal mixing and trend mixing respectively, also cause a decline of performance. This illustrates that solely relying on seasonal or trend information interaction is insufficient for accurate predictions. Furthermore, in both ablations ⑤ and ⑥, we employed the same mixing approach for both seasons and trends. However, it cannot bring better predictive performance. Similar situation occurs in $\textcircled{7}$ that adopts opposite mixing strategies to our design. These results demonstrate the effectiveness of our design in both bottom-up seasonal mixing and top-down trend mixing. Concurrently, in ablations $\textcircled{8}$ and ⑨, we opted to eliminate the decomposition architecture and mix the multiscale series directly. However, without decomposition, neither bottom-up nor top-down mixing method can achieve a good performance, indicating the necessity of season-trend separate mixing. Furthermore, in ablations ⑩, eliminating the entire Past-Decomposable-Mixing block causes a serious drop in the model’s predictive performance. The above findings highlight the substantial influence of an appropriate past mixing method on the final performance of the model. Starting from the insights in time series, TimeMixer presents the best mixing method in past information extraction. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/c8bbf89b7ac087ba9432b692ba7a32439e029b37f9f58138856f513308066633.jpg)



Table 5: Ablations on both PDM (Decompose, Season Mixing, Trend Mixing) and FMM blocks in M4, PEMS04 and predict-336 setting of ETTm1. $\nearrow$ indicates the bottom-up mixing while $\swarrow$ indicates top-down. A check mark $\checkmark$ and a wrong mark × indicate with and without certain components respectively. ① is the official design in TimeMixer (See Appendix F for complete ablation results).


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">M4</td><td colspan="3">PEMS04</td><td colspan="2">ETTm1</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MSE</td><td>MAE</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>√</td><td>√</td><td>11.723</td><td>1.559</td><td>0.840</td><td>19.21</td><td>12.53</td><td>30.92</td><td>0.390</td><td>0.404</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>√</td><td>×</td><td>12.503</td><td>1.634</td><td>0.925</td><td>21.67</td><td>13.45</td><td>34.89</td><td>0.402</td><td>0.415</td></tr><tr><td>3</td><td>√</td><td>×</td><td>√</td><td>√</td><td>13.051</td><td>1.676</td><td>0.962</td><td>24.49</td><td>16.28</td><td>38.79</td><td>0.411</td><td>0.427</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>12.911</td><td>1.655</td><td>0.941</td><td>22.91</td><td>15.02</td><td>37.04</td><td>0.405</td><td>0.414</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>12.008</td><td>1.628</td><td>0.871</td><td>20.78</td><td>13.02</td><td>32.47</td><td>0.392</td><td>0.413</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>11.978</td><td>1.626</td><td>0.859</td><td>21.09</td><td>13.78</td><td>33.11</td><td>0.396</td><td>0.415</td></tr><tr><td>7</td><td>√</td><td>√</td><td>↗</td><td>√</td><td>13.012</td><td>1.657</td><td>0.954</td><td>22.27</td><td>15.14</td><td>34.67</td><td>0.412</td><td>0.429</td></tr><tr><td>8</td><td>×</td><td colspan="2">↗</td><td>√</td><td>11.975</td><td>1.617</td><td>0.851</td><td>21.51</td><td>13.47</td><td>34.81</td><td>0.395</td><td>0.408</td></tr><tr><td>9</td><td>×</td><td colspan="2">√</td><td>√</td><td>11.973</td><td>1.622</td><td>0.850</td><td>21.79</td><td>14.03</td><td>35.23</td><td>0.393</td><td>0.406</td></tr><tr><td>10</td><td>×</td><td colspan="2">×</td><td>√</td><td>12.468</td><td>1.671</td><td>0.916</td><td>24.87</td><td>16.66</td><td>39.48</td><td>0.405</td><td>0.412</td></tr></table>

Seasonal and trend mixing visualization To provide an intuitive understanding of PDM, we visualize temporal linear weights for seasonal mixing and trend mixing in Figure $3 ( \mathbf { a } ) { \sim } ( \mathbf { b } )$ . We find that the seasonal and trend items present distinct mixing properties, where the seasonal mixing layer presents periodic changes (repeated blue lines in (a)) and the trend mixing layer is dominated by local aggregations (the dominating diagonal yellow line in (b)). This also verifies the necessity of adopting separate mixing techniques for seasonal and trend terms. Furthermore, Figure 3(c) shows the predictions of season and trend terms in fine (scale 0) and coarse (scale 3) scales. We can observe that the seasonal terms of fine-scale and trend parts of coarse-scale are crucial for accurate predictions. This observation provides insights for our design in utilizing bottom-up mixing for seasonal terms and top-down mixing for trend components. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/bc09b9fb8c321d39b99793284044cc4e2d9cef4e1e5d91942633ded2528ebb4c.jpg)



(a) Seasonal Mixing Weights (bottom-up: from 96 to 48)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/79d459158c3de622c62b4e421b3dd7160a6d04c6601f60ca5a68bbc3e4953378.jpg)



(b) Trend Mixing Weights (top-down: from 48 to 96)



(c) Multiscale Season-trend Predictions (input-96-predict-96)



Figure 3: Visualization of temporal linear weights in seasonal mixing (Eq. 4), trend mixing (Eq. 5), and predictions from multiscale season-trend items. All the experiments are on the ETTh1 dataset under the input-96-predict-96 setting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/05ee45ca3c1358057e09fdf30c050ea3aa045f04d8d2420640a572a42d007169.jpg)



(a) Multiscale mixing


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/48766e900a0c47dca4b2b0dc687be88155495082cda964c36936c1dd6f61c332.jpg)



(b) Scale 0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/0fded759de078392e7203dba23b86ececc8380adeceb3219302509860911183c.jpg)



(c) Scale 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/989d98e67997f8e6e1385e5bb80b438ba88d83af707fd61b353b0ae987c22fcb.jpg)



(d) Scale 2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/bc75ebf32614edf7064d9f93cbd9dacc5ceed853014ddc05bcfb21faf346f1b0.jpg)



(e) Scale 3



Figure 4: Visualization of predictions from different scales $( \widehat { \mathbf { x } } _ { m } ^ { L }$ in Eq. 6) on the input-96-predict-96 settings of the ETTh1 dataset. The implementation details are included in Appendix A.



GPU Memory by Series Length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/2b931217ebcdceb1b0cc3cae9de335534f4660d7188d267c5b9fa536150a51a3.jpg)



Running Time by Series Length



(a)	Memory	Efficiency	Analysis


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/befc063fb81254798c164457b70bdb8f0f651e9c64c0be1bc632282afc1bcd45.jpg)



(b)	Running	Time	Efficiency	Analysis



Figure 5: Efficiency analysis in both GPU memory and running time. The results are recorded on the ETTh1 dataset with batch size as 16. The running time is averaged from $1 0 ^ { 2 }$ iterations.


Multipredictor visualization To provide an intuitive understanding of the forecasting skills of multiscale series, we plot the forecasting results from different scales for qualitative comparison. Figure 4(a) presents the overall prediction of our model with Future-Multipredictor-Mixing, which indicates accurate prediction according to the future variations using mixed scales. To study the component of each individual scale, we demonstrate the prediction results for each scale in Figure 4(b)∼(e). Specifically, prediction results from fine-scale time series concentrate more on the detailed variations of time series and capture seasonal patterns with greater precision. In contrast, as shown in Figure 4(c)∼(e), with multiple downsampling, the predictions from coarse-scale series focus more on macro trends. The above results also highlight the benefits of Future-Multipredictor-Mixing in utilizing complementary forecasting skills from multiscale series. 

Efficiency analysis We compare the running memory and time against the latest state-of-the-art models in Figure 5 under the training phase, where TimeMixer consistently demonstrates favorable efficiency, in terms of both GPU memory and running time, for various series lengths (ranging from 192 to 3072), in addition to the consistent state-of-the-art performances for both long-term and short-term forecasting tasks. 

Analysis on number of scales We explore the impact from the number of scales (M) in Figure 6 under different series lengths. Specifically, when M increases, the performance gain declines for shorter prediction lengths. In contrast, for longer prediction lengths, the performance improves more as M increases. Therefore, we set M as 3 for long-term forecast and 1 for short-term forecast to trade off performance and efficiency. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/9c4d6ca7e390d339ba6775f8282cc43d5eb4f8145e1345c0b7947673c6cc8c98.jpg)



Figure 6: Analysis on number of scales on ETTm1 dataset.


## 5 CONCLUSION

We presented TimeMixer with a multiscale mixing architecture to tackle the intricate temporal variations in time series forecasting. Empowered by Past-Decomposable-Mixing and Future-Multipredictor-Mixing blocks, TimeMixer took advantage of both disentangled variations and complementary forecasting capabilities. In all of our experiments, TimeMixer achieved consistent state-of-the-art performances in both long-term and short-term forecasting tasks. Moreover, benefiting from the fully MLP-based architecture, TimeMixer demonstrated favorable run-time efficiency. Detailed visualizations and ablations are included to provide insights for our design. 

## 6 ETHICS STATEMENT

Our work only focuses on the scientific problem, so there is no potential ethical risk. 

## 7 REPRODUCIBILITY STATEMENT

We involve the implementation details in Appendix A, including dataset descriptions, metric calculation and experiment configuration. The source code is provided in supplementary materials and public in GitHub (https://github.com/kwuking/TimeMixer) for reproducibility. 

## ACKNOWLEDGMENTS

This work was supported by Ant Group through CCF-Ant Research Fund. 

## REFERENCES



Lili Meng Amin Shabani, Amir Abdi and Tristan Sylvain. Scaleformer: iterative multi-scale refining trans- formers for time series forecasting. ICLR, 2023. 





G. E. P. Box and Gwilym M. Jenkins. Time series analysis, forecasting and control. 1970. 





Cristian Challu, Kin G Olivares, Boris N Oreshkin, Federico Garza, Max Mergenthaler, and Artur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. AAAI, 2023. 





Chao Chen, Karl F. Petty, Alexander Skabardonis, Pravin Pratap Varaiya, and Zhanfeng Jia. Freeway performance measurement system: Mining loop detector data. Transportation Research Record, 2001. 





Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O. Arik, and Tomas Pfister. Tsmixer: An all-mlp architecture for time series forecasting. arXiv preprint arXiv:2303.06053, 2023. 





Robert B Cleveland, William S Cleveland, Jean E McRae, and Irma Terpenning. STL: A seasonaltrend decomposition. Journal ofOfficial Statistics, 1990. 





Marco A. R. Ferreira, Michael A. West, Herbert K. H. Lee, and David M. Higdon. Multi-scale and hidden resolution time series models. Bayesian Analysis, 2006. 





Georg Goerg. Forecastable component analysis. ICML, 2013. 





Clive William John Granger and Paul Newbold. Forecasting economic time series. Academic press, 2014. 





Pradeep Hewage, Ardhendu Behera, Marcello Trovati, Ella Pereira, Morteza Ghahremani, Francesco Palmieri, and Yonghuai Liu. Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station. Soft Computing, 2020. 





Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015. 





Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. ICLR, 2020. 





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. SIGIR, 2018. 





James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. Fnet: Mixing tokens with fourier transforms. NAACL, 2022. 





Zhe Li, Zhongwen Rao, Lujia Pan, and Zenglin Xu. Mts-mixers: Multivariate time series forecasting via factorized temporal and channel mixing. arXiv preprint arXiv:2302.04501, 2023. 





Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. SCINet: time series modeling and forecasting with sample convolution and interaction. NeurIPS, 2022a. 





Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. ICLR, 2021. 





Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Rethinking the stationarity in time series forecasting. NeurIPS, 2022b. 





Luis Martín, Luis F Zarzalejo, Jesus Polo, Ana Navarro, Ruth Marchante, and Marco Cony. Prediction of global solar irradiance based on time series analysis: Application to solar thermal power plants energy production planning. Solar Energy, 2010. 





Michael C. Mozer. Induction of multiscale temporal structure. NeurIPS, 1991. 





Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. ICLR, 2023. 





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR, 2019. 





Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019. 





Zheng Qian, Yan Pei, Hamidreza Zareipour, and Niya Chen. A review and discussion of decomposition-based hybrid models for wind energy forecasting applications. Applied energy, 2019. 





Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison Cottrell. A dual-stage attention-based recurrent neural network for time series prediction. IJCAI, 2017. 





David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 2020. 





Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. MLP-Mixer: An all-MLP architecture for vision. NeurIPS, 2021. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 





Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: Multiscale local and global context modeling for long-term series forecasting. ICLR, 2023. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. NeurIPS, 2021. 





Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2d-variation modeling for general time series analysis. ICLR, 2023a. 





Haixu Wu, Hang Zhou, Mingsheng Long, and Jianmin Wang. Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 2023b. 





Xueyan Yin, Genze Wu, Jinze Wei, Yanming Shen, Heng Qi, and Baocai Yin. Deep learning on traffic prediction: Methods, analysis, and future directions. IEEE Transactions on Intelligent Transportation Systems, 2021. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? AAAI, 2023. 





Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022. 





Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. ICLR, 2023. 



Zheng Zhao, Weihai Chen, Xingming Wu, Peter CY Chen, and Jingmeng Liu. Lstm network: a deep learning approach for short-term traffic forecast. IET Intelligent Transport Systems, 2017. 

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. AAAI, 2021. 

Tian Zhou, Ziqing Ma, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin, et al. Film: Frequency improved legendre memory model for long-term time series forecasting. NeurIPS, 2022a. 

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. ICML, 2022b. 

## A IMPLEMENTATION DETAILS

We summarized details of datasets, evaluation metrics, experiments and visualizations in this section. 

Datasets details We evaluate the performance of different models for long-term forecasting on 8 well-established datasets, including Weather, Traffic, Electricity, Solar-Energy, and ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2). Furthermore, we adopt PeMS and M4 datasets for short-term forecasting. We detail the descriptions of the dataset in Table 6. 


Table 6: Dataset detailed descriptions. The dataset size is organized in (Train, Validation, Test).


<table><tr><td>Tasks</td><td>Dataset</td><td>Dim</td><td>Series Length</td><td>Dataset Size</td><td>Frequency</td><td>Forecastability*</td><td>Information</td></tr><tr><td rowspan="8">Long-term Forecasting</td><td>ETTm1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.46</td><td>Temperature</td></tr><tr><td>ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.55</td><td>Temperature</td></tr><tr><td>ETTh1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 min</td><td>0.38</td><td>Temperature</td></tr><tr><td>ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>15 min</td><td>0.45</td><td>Temperature</td></tr><tr><td>Electricity</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Hourly</td><td>0.77</td><td>Electricity</td></tr><tr><td>Traffic</td><td>862</td><td>{96, 192, 336, 720}</td><td>(12185, 1757, 3509)</td><td>Hourly</td><td>0.68</td><td>Transportation</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10 min</td><td>0.75</td><td>Weather</td></tr><tr><td>Solar-Energy</td><td>137</td><td>{96, 192, 336, 720}</td><td>(36601, 5161, 10417)</td><td>10min</td><td>0.33</td><td>Electricity</td></tr><tr><td rowspan="10">Short-term Forecasting</td><td>PEMS03</td><td>358</td><td>12</td><td>(15701,5216,434)</td><td>5min</td><td>0.65</td><td>Transportation</td></tr><tr><td>PEMS04</td><td>307</td><td>12</td><td>(10172,3375,281)</td><td>5min</td><td>0.45</td><td>Transportation</td></tr><tr><td>PEMS07</td><td>883</td><td>12</td><td>(16911,5622,468)</td><td>5min</td><td>0.58</td><td>Transportation</td></tr><tr><td>PEMS08</td><td>170</td><td>12</td><td>(10690,3548,265)</td><td>5min</td><td>0.52</td><td>Transportation</td></tr><tr><td>M4-Yearly</td><td>1</td><td>6</td><td>(23000, 0, 23000)</td><td>Yearly</td><td>0.43</td><td>Demographic</td></tr><tr><td>M4-Quarterly</td><td>1</td><td>8</td><td>(24000, 0, 24000)</td><td>Quarterly</td><td>0.47</td><td>Finance</td></tr><tr><td>M4-Monthly</td><td>1</td><td>18</td><td>(48000, 0, 48000)</td><td>Monthly</td><td>0.44</td><td>Industry</td></tr><tr><td>M4-Weakly</td><td>1</td><td>13</td><td>(359, 0, 359)</td><td>Weakly</td><td>0.43</td><td>Macro</td></tr><tr><td>M4-Daily</td><td>1</td><td>14</td><td>(4227, 0, 4227)</td><td>Daily</td><td>0.44</td><td>Micro</td></tr><tr><td>M4-Hourly</td><td>1</td><td>48</td><td>(414, 0, 414)</td><td>Hourly</td><td>0.46</td><td>Other</td></tr></table>


∗ The forecastability is calculated by one minus the entropy of Fourier decomposition of time series (Goerg, 2013). A larger value indicates better predictability. 


Metric details Regarding metrics, we utilize the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting. In the case of short-term forecasting, we follow the metrics of SCINet (Liu et al., 2022a) on the PeMS datasets, including mean absolute error (MAE), mean absolute percentage error (MAPE), root mean squared error (RMSE). As for the M4 datasets, we follow the methodology of N-BEATS (Oreshkin et al., 2019) and implement the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA) as metrics. It is worth noting that OWA is a specific metric utilized in the M4 competition. The calculations of these metrics are: 

$$
\begin{array}{l l} \text {RMSE} = (\sum_ {i = 1} ^ {F} (\mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i}) ^ {2}) ^ {\frac {1}{2}}, & \text {MAE} = \sum_ {i = 1} ^ {F} | \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |, \\ \text {SMAPE} = \frac {2 0 0}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} | + | \widehat {\mathbf {X}} _ {i} |}, & \text {MAPE} = \frac {1 0 0}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} |}, \\ \text {MASE} = \frac {1}{F} \sum_ {i = 1} ^ {F} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{\frac {1}{F - s} \sum_ {j = s + 1} ^ {F} | \mathbf {X} _ {j} - \mathbf {X} _ {j - s} |}, & \text {OWA} = \frac {1}{2} \left[ \frac {\text {SMAPE}}{\text {SMAPE} _ {\text {Naïve2}}} + \frac {\text {MASE}}{\text {MASE} _ {\text {Naïve2}}} \right], \end{array}
$$

where s is the periodicity of the data. $\mathbf { X } , \widehat { \mathbf { X } } \in \mathbb { R } ^ { F \times C }$ are the ground truth and prediction results of the future with F time pints and C dimensions. $\mathbf { X } _ { i }$ means the i-th future time point. 

Experiment details All experiments were run three times, implemented in Pytorch (Paszke et al., 2019), and conducted on a single NVIDIA A100 80GB GPU. We set the initial learning rate as $1 0 ^ { - 2 }$ or $1 0 ^ { - 3 }$ and used the ADAM optimizer (Kingma & Ba, 2015) with L2 loss for model optimization. And the batch size was set to be 8 between 128. By default, TimeMixer contains 2 Past Decomposable Mixing blocks. We choose the number of scales M according to the length of the time series to achieve a balance between performance and efficiency. To handle longer series in long-term forecasting, we set M to 3. As for short-term forecasting with limited series length, we set M to 1. Detailed model configuration information is presented in Table 7. 

Visualization details To verify complementary forecasting capabilities of multiscale series, we fix the PDM and train a new predictor for the feature at each scale with the ground truth future as supervision in Figure 4; Figure 3(c) also utilizes the same operations. Especially for Figure 4, we also provide the visualization of directly plotting the output of each predictor, i.e. $\widehat { \mathbf { x } } _ { m } , m \in \{ 0 , \cdots , M \}$ in Eq. 6. Note that in FMM, we adopt the sum ensemble $\begin{array} { r } { \widehat { \mathbf { x } } = \sum _ { m = 0 } ^ { M } \widehat { \mathbf { x } } _ { m } } \end{array}$ as the final output, the scale of each plotted cure is around $\frac { 1 } { M + 1 }$ of ground truth, while we can still observe the distinct forecasting capability of series in different scales. For clearness, we also plot the $( M + 1 ) \times \widehat { \mathbf { x } } _ { m }$ in the second row of Figure 7, where the visualizations are similar to Figure 4. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f957a769c8e89f1727ae32fce8b6f7130ed43b440edf04479dcc9016864a90fd.jpg)



(b) Scale 0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/fb8925f5cc1927775807eb0a02aabb3a8b2e93e9e9f2ee52fadcefdbe0457b3e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d82c0fa587af8ded75a2a3dd8370610760754a5dff1cf5e7cadc373a0ae4deb5.jpg)



(c) Scale 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/6f980f1be3d4b1290832f558771f7123d048772add29f955f61ccb8c3119035a.jpg)



(d) Scale 2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/351e967af8560bdbad369036c6f306fbd03d802fec5670f5798b13e99ea514ac.jpg)



(e) Scale 3



(a) Multiscale mixing


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/bcbf43363455bd7d24f9f1ef7f6c9984e6f4e7d40cb883ebf19a481f9aba432e.jpg)



(f) Scale 0 × (� + 1)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/c260c1a1d5853b54bf20a63af1399989771f76a6ad570270ed9fffd02bea50b4.jpg)



(g) Scale 1 × (� + 1)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/b63bf1edbbe9228472c676bec4a3fb2b919a6d16e884b2c73d3d953590598933.jpg)



(h) Scale 2 × (� + 1)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f9e13c74a5606df641af5572d083040594ee72dcdd47fa793465fb7552b56275.jpg)



(i) Scale 3 × (� + 1)



Figure 7: Direct visualization of predictions from different scales $( \widehat { \mathbf { x } } _ { m } ^ { L }$ in Eq. 6) on the input-96- predict-96 settings of the ETTh1 dataset. We multiply the $( M + 1 )$ by the predictions of each scale in the second row.


## B EFFICIENCY ANALYSIS

In the main text, we have ploted the curve of efficiency in Figure 5. Here we present the quantitive results in Table 8. It should be noted that TimeMixer’s outstanding efficiency advantage over Transformer-based models, such as PatchTST, FEDformer, and Autoformer, is attributed to its fully MLP-based network architecture. 


Table 7: Experiment configuration of TimeMixer. All the experiments use the ADAM (2015) optimizer with the default hyperparameter configuration for $( \beta _ { 1 } , \beta _ { 2 } )$ as (0.9, 0.999).


<table><tr><td rowspan="2">Dataset / Configurations</td><td colspan="3">Model Hyper-parameter</td><td colspan="4">Training Process</td></tr><tr><td>M (Equ. 1)</td><td>Layers</td><td><eq>d_{model}</eq></td><td>LR*</td><td>Loss</td><td>Batch Size</td><td>Epochs</td></tr><tr><td>ETTh1</td><td>3</td><td>2</td><td>16</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>128</td><td>10</td></tr><tr><td>ETTh2</td><td>3</td><td>2</td><td>16</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>128</td><td>10</td></tr><tr><td>ETTm1</td><td>3</td><td>2</td><td>16</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>128</td><td>10</td></tr><tr><td>ETTm2</td><td>3</td><td>2</td><td>32</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>128</td><td>10</td></tr><tr><td>Weather</td><td>3</td><td>2</td><td>16</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>128</td><td>20</td></tr><tr><td>Electricity</td><td>3</td><td>2</td><td>16</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>32</td><td>20</td></tr><tr><td>Solar-Energy</td><td>3</td><td>2</td><td>128</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>32</td><td>20</td></tr><tr><td>Traffic</td><td>3</td><td>2</td><td>32</td><td><eq>10^{-2}</eq></td><td>MSE</td><td>8</td><td>20</td></tr><tr><td>PEMS</td><td>1</td><td>5</td><td>128</td><td><eq>10^{-3}</eq></td><td>MSE</td><td>32</td><td>10</td></tr><tr><td>M4</td><td>1</td><td>4</td><td>32</td><td><eq>10^{-2}</eq></td><td>SMAPE</td><td>128</td><td>50</td></tr></table>


∗ LR means the initial learning rate. 



Table 8: The GPU memory (MiB) and speed (running time, s/iter) of each model.


<table><tr><td rowspan="2">Series LengthModels</td><td colspan="2">192</td><td colspan="2">384</td><td colspan="2">768</td><td colspan="2">1536</td><td colspan="2">3072</td></tr><tr><td>Mem</td><td>Speed</td><td>Mem</td><td>Speed</td><td>Mem</td><td>Speed</td><td>Mem</td><td>Speed</td><td>Mem</td><td>Speed</td></tr><tr><td>TimeMixer (Ours)</td><td>1003</td><td>0.007</td><td>1043</td><td>0.007</td><td>1075</td><td>0.008</td><td>1151</td><td>0.009</td><td>1411</td><td>0.016</td></tr><tr><td>PatchTST (2023)</td><td>1919</td><td>0.018</td><td>2097</td><td>0.019</td><td>2749</td><td>0.021</td><td>5465</td><td>0.032</td><td>16119</td><td>0.094</td></tr><tr><td>TimesNet (2023a)</td><td>1148</td><td>0.028</td><td>1245</td><td>0.024</td><td>1585</td><td>0.042</td><td>2491</td><td>0.045</td><td>2353</td><td>0.073</td></tr><tr><td>Crossformer (2023)</td><td>1737</td><td>0.027</td><td>1799</td><td>0.027</td><td>1895</td><td>0.028</td><td>2303</td><td>0.033</td><td>3759</td><td>0.035</td></tr><tr><td>MICN (2023)</td><td>1771</td><td>0.014</td><td>1801</td><td>0.016</td><td>1873</td><td>0.017</td><td>1991</td><td>0.018</td><td>2239</td><td>0.020</td></tr><tr><td>DLinear (2023)</td><td>1001</td><td>0.002</td><td>1021</td><td>0.003</td><td>1041</td><td>0.003</td><td>1081</td><td>0.0.004</td><td>1239</td><td>0.015</td></tr><tr><td>FEDFormer (2022b)</td><td>2567</td><td>0.132</td><td>5977</td><td>0.141</td><td>7111</td><td>0.143</td><td>9173</td><td>0.178</td><td>12485</td><td>0.288</td></tr><tr><td>Autoformer (2021)</td><td>1761</td><td>0.028</td><td>2101</td><td>0.070</td><td>3209</td><td>0.071</td><td>5395</td><td>0.129</td><td>10043</td><td>0.255</td></tr></table>

## C ERROR BARS

In this paper, we repeat all the experiments three times. Here we report the standard deviation of our model and the second best model, as well as the statistical significance test in Table 9, 10, 11. 


Table 9: Standard deviation and statistical tests for our TimeMixer method and second-best method (PatchTST) on ETT, Weather, Solar-Energy, Electricity and Traffic datasets.


<table><tr><td>Model</td><td colspan="2">TimeMixer</td><td colspan="2">PatchTST (2023)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>Interval</td></tr><tr><td>Weather</td><td><eq>0.240 \pm 0.010</eq></td><td><eq>0.271 \pm 0.009</eq></td><td><eq>0.265 \pm 0.012</eq></td><td><eq>0.285 \pm 0.011</eq></td><td>99%</td></tr><tr><td>Solar-Energy</td><td><eq>0.216 \pm 0.002</eq></td><td><eq>0.280 \pm 0.022</eq></td><td><eq>0.287 \pm 0.020</eq></td><td><eq>0.333 \pm 0.018</eq></td><td>99%</td></tr><tr><td>Electricity</td><td><eq>0.182 \pm 0.017</eq></td><td><eq>0.272 \pm 0.006</eq></td><td><eq>0.216 \pm 0.012</eq></td><td><eq>0.318 \pm 0.015</eq></td><td>99%</td></tr><tr><td>Traffic</td><td><eq>0.484 \pm 0.015</eq></td><td><eq>0.297 \pm 0.013</eq></td><td><eq>0.529 \pm 0.008</eq></td><td><eq>0.341 \pm 0.002</eq></td><td>99%</td></tr><tr><td>ETTh1</td><td><eq>0.447 \pm 0.002</eq></td><td><eq>0.440 \pm 0.005</eq></td><td><eq>0.516 \pm 0.003</eq></td><td><eq>0.484 \pm 0.005</eq></td><td>99%</td></tr><tr><td>ETTh2</td><td><eq>0.364 \pm 0.008</eq></td><td><eq>0.395 \pm 0.010</eq></td><td><eq>0.391 \pm 0.005</eq></td><td><eq>0.411 \pm 0.003</eq></td><td>99%</td></tr><tr><td>ETTm1</td><td><eq>0.381 \pm 0.003</eq></td><td><eq>0.395 \pm 0.006</eq></td><td><eq>0.400 \pm 0.002</eq></td><td><eq>0.407 \pm 0.005</eq></td><td>99%</td></tr><tr><td>ETTm2</td><td><eq>0.275 \pm 0.001</eq></td><td><eq>0.323 \pm 0.003</eq></td><td><eq>0.290 \pm 0.002</eq></td><td><eq>0.334 \pm 0.002</eq></td><td>99%</td></tr></table>


Table 10: Standard deviation and statistical tests for our TimeMixer method and second-best method (SCINet) on PEMS dataset.


<table><tr><td>Model</td><td colspan="3">TimeMixer</td><td colspan="3">SCINet (2022a)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>Interval</td></tr><tr><td>PEMS03</td><td><eq>14.63 \pm 0.112</eq></td><td><eq>14.54 \pm 0.105</eq></td><td><eq>23.28 \pm 0.128</eq></td><td><eq>15.97 \pm 0.153</eq></td><td><eq>15.89 \pm 0.122</eq></td><td><eq>25.20 \pm 0.137</eq></td><td>99%</td></tr><tr><td>PEMS04</td><td><eq>19.21 \pm 0.217</eq></td><td><eq>12.53 \pm 0.154</eq></td><td><eq>30.92 \pm 0.143</eq></td><td><eq>20.35 \pm 0.201</eq></td><td><eq>12.84 \pm 0.213</eq></td><td><eq>32.31 \pm 0.178</eq></td><td>95%</td></tr><tr><td>PEMS07</td><td><eq>20.57 \pm 0.158</eq></td><td><eq>8.62 \pm 0.112</eq></td><td><eq>33.59 \pm 0.273</eq></td><td><eq>22.79 \pm 0.179</eq></td><td><eq>9.41 \pm 0.105</eq></td><td><eq>35.61 \pm 0.112</eq></td><td>99%</td></tr><tr><td>PEMS08</td><td><eq>15.22 \pm 0.311</eq></td><td><eq>9.67 \pm 0.101</eq></td><td><eq>24.26 \pm 0.212</eq></td><td><eq>17.38 \pm 0.332</eq></td><td><eq>10.80 \pm 0.219</eq></td><td><eq>27.34 \pm 0.178</eq></td><td>99%</td></tr></table>


Table 11: Standard deviation and statistical tests for our TimeMixer method and second-best method (TimesNet) on M4 dataset.


<table><tr><td>Model</td><td colspan="3">TimeMixer</td><td colspan="3">TimesNet (2023a)</td><td>Confidence</td></tr><tr><td>Dataset</td><td>SMAPE</td><td>MAPE</td><td>OWA</td><td>SMAPE</td><td>MAPE</td><td>OWA</td><td>Interval</td></tr><tr><td>Yearly</td><td><eq>13.206 \pm 0.121</eq></td><td><eq>2.916 \pm 0.022</eq></td><td><eq>0.776 \pm 0.002</eq></td><td><eq>13.387 \pm 0.112</eq></td><td><eq>2.996 \pm 0.017</eq></td><td><eq>0.786 \pm 0.010</eq></td><td>95%</td></tr><tr><td>Quarterly</td><td><eq>9.996 \pm 0.101</eq></td><td><eq>1.166 \pm 0.015</eq></td><td><eq>0.825 \pm 0.008</eq></td><td><eq>10.100 \pm 0.105</eq></td><td><eq>1.182 \pm 0.012</eq></td><td><eq>0.890 \pm 0.006</eq></td><td>95%</td></tr><tr><td>Monthly</td><td><eq>12.605 \pm 0.115</eq></td><td><eq>0.919 \pm 0.011</eq></td><td><eq>0.869 \pm 0.003</eq></td><td><eq>12.670 \pm 0.106</eq></td><td><eq>0.933 \pm 0.008</eq></td><td><eq>0.878 \pm 0.001</eq></td><td>95%</td></tr><tr><td>Others</td><td><eq>4.564 \pm 0.114</eq></td><td><eq>3.115 \pm 0.027</eq></td><td><eq>0.982 \pm 0.011</eq></td><td><eq>4.891 \pm 0.120</eq></td><td><eq>3.302 \pm 0.023</eq></td><td><eq>1.035 \pm 0.017</eq></td><td>99%</td></tr><tr><td>Averaged</td><td><eq>11.723 \pm 0.011</eq></td><td><eq>1.559 \pm 0.022</eq></td><td><eq>0.840 \pm 0.001</eq></td><td><eq>11.829 \pm 0.120</eq></td><td><eq>1.585 \pm 0.017</eq></td><td><eq>0.851 \pm 0.003</eq></td><td>95%</td></tr></table>

## D HYPERPARAMTER SENSITIVITY

In the main text, we have explored the effect of number of scales M. Here, we further evaluate the number of layers L. As shown in Table 12, we can find that in general, increasing the number of layers (L) will bring improvements across different prediction lengths. Therefore, we set to 2 to trade off efficiency and performance. 


Table 12: The MSE results of different number of scales (M) and layers (L) on the ETTm1 dataset.


<table><tr><td>Predict Length Num. of Scales</td><td>96</td><td>192</td><td>336</td><td>720</td><td>Predict Length Num. of Layers</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>1</td><td>0.326</td><td>0.371</td><td>0.405</td><td>0.469</td><td>1</td><td>0.328</td><td>0.369</td><td>0.405</td><td>0.467</td></tr><tr><td>2</td><td>0.323</td><td>0.365</td><td>0.401</td><td>0.460</td><td>2</td><td>0.320</td><td>0.361</td><td>0.390</td><td>0.454</td></tr><tr><td>3</td><td>0.320</td><td>0.361</td><td>0.390</td><td>0.454</td><td>3</td><td>0.321</td><td>0.360</td><td>0.389</td><td>0.451</td></tr><tr><td>4</td><td>0.321</td><td>0.360</td><td>0.388</td><td>0.454</td><td>4</td><td>0.318</td><td>0.361</td><td>0.385</td><td>0.452</td></tr><tr><td>5</td><td>0.321</td><td>0.362</td><td>0.389</td><td>0.461</td><td>5</td><td>0.322</td><td>0.359</td><td>0.390</td><td>0.457</td></tr></table>

## E FULL RESULTS

To ensure a fair comparison between models, we conducted experiments using unified parameters and reported results in the main text, including aligning all the input lengths, batch sizes, and training epochs in all experiments. Here, we provide the full results for each forecasting setting in Table 13. 

In addition, considering that the reported results in different papers are mostly obtained through hyperparameter search, we provide the experiment results with the full version of the parameter search. We searched for input length among 96, 192, 336, and 512, learning rate from $1 0 ^ { - 5 }$ to 0.05, encoder layers from 1 to 5, the $d _ { \mathrm { m o d e l } }$ from 16 to 512, training epochs from 10 to 100. The results are included in Table 14, which can be used to compare the upper bound of each forecasting model. 

We can find that the relative promotion of TimesMixer over PatchTST is smaller under comprehensive hyperparameter search than the unified hyperparameter setting. It is worth noticing that TimeMixer runs much faster than PatchTST according to the efficiency comparison in Table 8. Therefore, considering perfromance, hyperparameter-search cost and efficiency, we believe TimeMixer is a practical model in real-world applications and is valuable to deep time series forecasting community. 


Table 13: Unified hyperparameter results for the long-term forecasting task. We compare extensive competitive models under different prediction lengths. Avg is averaged from all four prediction lengths, that is 96, 192, 336, 720.


<table><tr><td colspan="2">Models</td><td colspan="2">TimeMixer (Ours)</td><td colspan="2">PatchTST 2023</td><td colspan="2">TimesNet 2023a</td><td colspan="2">Crossformer 2023</td><td colspan="2">MICN 2023</td><td colspan="2">FiLM 2022a</td><td colspan="2">DLinear 2023</td><td colspan="2">FEDformer 2022b</td><td colspan="2">Stationary 2022b</td><td colspan="2">Autoformer 2021</td><td colspan="2">Informer 2021</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.163</td><td>0.209</td><td>0.186</td><td>0.227</td><td>0.172</td><td>0.220</td><td>0.195</td><td>0.271</td><td>0.198</td><td>0.261</td><td>0.195</td><td>0.236</td><td>0.195</td><td>0.252</td><td>0.217</td><td>0.296</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td><td>0.300</td><td>0.384</td></tr><tr><td>192</td><td>0.208</td><td>0.250</td><td>0.234</td><td>0.265</td><td>0.219</td><td>0.261</td><td>0.209</td><td>0.277</td><td>0.239</td><td>0.299</td><td>0.239</td><td>0.271</td><td>0.237</td><td>0.295</td><td>0.276</td><td>0.336</td><td>0.245</td><td>0.285</td><td>0.307</td><td>0.367</td><td>0.598</td><td>0.544</td></tr><tr><td>336</td><td>0.251</td><td>0.287</td><td>0.284</td><td>0.301</td><td>0.246</td><td>0.337</td><td>0.273</td><td>0.332</td><td>0.285</td><td>0.336</td><td>0.289</td><td>0.306</td><td>0.282</td><td>0.331</td><td>0.339</td><td>0.380</td><td>0.321</td><td>0.338</td><td>0.359</td><td>0.395</td><td>0.578</td><td>0.523</td></tr><tr><td>720</td><td>0.339</td><td>0.341</td><td>0.356</td><td>0.349</td><td>0.365</td><td>0.359</td><td>0.379</td><td>0.401</td><td>0.351</td><td>0.388</td><td>0.361</td><td>0.351</td><td>0.345</td><td>0.382</td><td>0.403</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.419</td><td>0.428</td><td>1.059</td><td>0.741</td></tr><tr><td></td><td>Avg</td><td>0.240</td><td>0.271</td><td>0.265</td><td>0.285</td><td>0.251</td><td>0.294</td><td>0.264</td><td>0.320</td><td>0.268</td><td>0.321</td><td>0.271</td><td>0.291</td><td>0.265</td><td>0.315</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td><td>0.634</td><td>0.548</td></tr><tr><td rowspan="4">Solar-Energy</td><td>96</td><td>0.189</td><td>0.259</td><td>0.265</td><td>0.323</td><td>0.373</td><td>0.358</td><td>0.232</td><td>0.302</td><td>0.257</td><td>0.325</td><td>0.333</td><td>0.350</td><td>0.290</td><td>0.378</td><td>0.286</td><td>0.341</td><td>0.321</td><td>0.380</td><td>0.456</td><td>0.446</td><td>0.287</td><td>0.323</td></tr><tr><td>192</td><td>0.222</td><td>0.283</td><td>0.288</td><td>0.332</td><td>0.397</td><td>0.376</td><td>0.371</td><td>0.410</td><td>0.278</td><td>0.354</td><td>0.371</td><td>0.372</td><td>0.320</td><td>0.398</td><td>0.291</td><td>0.337</td><td>0.346</td><td>0.369</td><td>0.588</td><td>0.561</td><td>0.297</td><td>0.341</td></tr><tr><td>336</td><td>0.231</td><td>0.292</td><td>0.301</td><td>0.339</td><td>0.420</td><td>0.380</td><td>0.495</td><td>0.515</td><td>0.298</td><td>0.375</td><td>0.408</td><td>0.385</td><td>0.353</td><td>0.415</td><td>0.354</td><td>0.416</td><td>0.357</td><td>0.387</td><td>0.595</td><td>0.588</td><td>0.367</td><td>0.429</td></tr><tr><td>720</td><td>0.223</td><td>0.285</td><td>0.295</td><td>0.336</td><td>0.420</td><td>0.381</td><td>0.526</td><td>0.542</td><td>0.299</td><td>0.379</td><td>0.406</td><td>0.377</td><td>0.357</td><td>0.413</td><td>0.380</td><td>0.437</td><td>0.375</td><td>0.424</td><td>0.733</td><td>0.633</td><td>0.374</td><td>0.431</td></tr><tr><td></td><td>Avg</td><td>0.216</td><td>0.280</td><td>0.287</td><td>0.333</td><td>0.403</td><td>0.374</td><td>0.406</td><td>0.442</td><td>0.283</td><td>0.358</td><td>0.380</td><td>0.371</td><td>0.330</td><td>0.401</td><td>0.328</td><td>0.383</td><td>0.350</td><td>0.390</td><td>0.586</td><td>0.557</td><td>0.331</td><td>0.381</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.153</td><td>0.247</td><td>0.190</td><td>0.296</td><td>0.168</td><td>0.272</td><td>0.219</td><td>0.314</td><td>0.180</td><td>0.293</td><td>0.198</td><td>0.274</td><td>0.210</td><td>0.302</td><td>0.193</td><td>0.308</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td><td>0.274</td><td>0.368</td></tr><tr><td>192</td><td>0.166</td><td>0.256</td><td>0.199</td><td>0.304</td><td>0.184</td><td>0.322</td><td>0.231</td><td>0.322</td><td>0.189</td><td>0.302</td><td>0.198</td><td>0.278</td><td>0.210</td><td>0.305</td><td>0.201</td><td>0.315</td><td>0.182</td><td>0.286</td><td>0.222</td><td>0.334</td><td>0.296</td><td>0.386</td></tr><tr><td>336</td><td>0.185</td><td>0.277</td><td>0.217</td><td>0.319</td><td>0.198</td><td>0.300</td><td>0.246</td><td>0.337</td><td>0.198</td><td>0.312</td><td>0.217</td><td>0.300</td><td>0.223</td><td>0.319</td><td>0.214</td><td>0.329</td><td>0.200</td><td>0.304</td><td>0.231</td><td>0.443</td><td>0.300</td><td>0.394</td></tr><tr><td>720</td><td>0.225</td><td>0.310</td><td>0.258</td><td>0.352</td><td>0.220</td><td>0.320</td><td>0.280</td><td>0.363</td><td>0.217</td><td>0.330</td><td>0.278</td><td>0.356</td><td>0.258</td><td>0.350</td><td>0.246</td><td>0.355</td><td>0.222</td><td>0.321</td><td>0.254</td><td>0.361</td><td>0.373</td><td>0.439</td></tr><tr><td></td><td>Avg</td><td>0.182</td><td>0.272</td><td>0.216</td><td>0.318</td><td>0.193</td><td>0.304</td><td>0.244</td><td>0.334</td><td>0.196</td><td>0.309</td><td>0.223</td><td>0.302</td><td>0.225</td><td>0.319</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td><td>0.311</td><td>0.397</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.462</td><td>0.285</td><td>0.526</td><td>0.347</td><td>0.593</td><td>0.321</td><td>0.644</td><td>0.429</td><td>0.577</td><td>0.350</td><td>0.647</td><td>0.384</td><td>0.650</td><td>0.396</td><td>0.587</td><td>0.366</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td><td>0.719</td><td>0.391</td></tr><tr><td>192</td><td>0.473</td><td>0.296</td><td>0.522</td><td>0.332</td><td>0.617</td><td>0.336</td><td>0.665</td><td>0.431</td><td>0.589</td><td>0.356</td><td>0.600</td><td>0.361</td><td>0.598</td><td>0.370</td><td>0.604</td><td>0.373</td><td>0.613</td><td>0.340</td><td>0.616</td><td>0.382</td><td>0.696</td><td>0.379</td></tr><tr><td>336</td><td>0.498</td><td>0.296</td><td>0.517</td><td>0.334</td><td>0.629</td><td>0.336</td><td>0.674</td><td>0.420</td><td>0.594</td><td>0.358</td><td>0.610</td><td>0.367</td><td>0.605</td><td>0.373</td><td>0.621</td><td>0.383</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.337</td><td>0.777</td><td>0.420</td></tr><tr><td>720</td><td>0.506</td><td>0.313</td><td>0.552</td><td>0.352</td><td>0.640</td><td>0.350</td><td>0.683</td><td>0.424</td><td>0.613</td><td>0.361</td><td>0.691</td><td>0.425</td><td>0.645</td><td>0.394</td><td>0.626</td><td>0.382</td><td>0.653</td><td>0.355</td><td>0.660</td><td>0.408</td><td>0.864</td><td>0.472</td></tr><tr><td></td><td>Avg</td><td>0.484</td><td>0.297</td><td>0.529</td><td>0.341</td><td>0.620</td><td>0.336</td><td>0.667</td><td>0.426</td><td>0.593</td><td>0.356</td><td>0.637</td><td>0.384</td><td>0.625</td><td>0.383</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td><td>0.764</td><td>0.416</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.375</td><td>0.400</td><td>0.460</td><td>0.447</td><td>0.384</td><td>0.402</td><td>0.423</td><td>0.448</td><td>0.426</td><td>0.446</td><td>0.438</td><td>0.433</td><td>0.397</td><td>0.412</td><td>0.395</td><td>0.424</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td><td>0.865</td><td>0.713</td></tr><tr><td>192</td><td>0.429</td><td>0.421</td><td>0.512</td><td>0.477</td><td>0.436</td><td>0.429</td><td>0.471</td><td>0.474</td><td>0.454</td><td>0.464</td><td>0.493</td><td>0.466</td><td>0.446</td><td>0.441</td><td>0.469</td><td>0.470</td><td>0.534</td><td>0.504</td><td>0.500</td><td>0.482</td><td>1.008</td><td>0.792</td></tr><tr><td>336</td><td>0.484</td><td>0.458</td><td>0.546</td><td>0.496</td><td>0.638</td><td>0.469</td><td>0.570</td><td>0.546</td><td>0.493</td><td>0.487</td><td>0.547</td><td>0.495</td><td>0.489</td><td>0.467</td><td>0.530</td><td>0.499</td><td>0.588</td><td>0.535</td><td>0.521</td><td>0.496</td><td>1.107</td><td>0.809</td></tr><tr><td>720</td><td>0.498</td><td>0.482</td><td>0.544</td><td>0.517</td><td>0.521</td><td>0.500</td><td>0.653</td><td>0.621</td><td>0.526</td><td>0.526</td><td>0.586</td><td>0.538</td><td>0.513</td><td>0.510</td><td>0.598</td><td>0.544</td><td>0.643</td><td>0.616</td><td>0.514</td><td>0.512</td><td>1.181</td><td>0.865</td></tr><tr><td></td><td>Avg</td><td>0.447</td><td>0.440</td><td>0.516</td><td>0.484</td><td>0.495</td><td>0.450</td><td>0.529</td><td>0.522</td><td>0.475</td><td>0.480</td><td>0.516</td><td>0.483</td><td>0.461</td><td>0.457</td><td>0.498</td><td>0.484</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td><td>1.040</td><td>0.795</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.289</td><td>0.341</td><td>0.308</td><td>0.355</td><td>0.340</td><td>0.374</td><td>0.745</td><td>0.584</td><td>0.372</td><td>0.424</td><td>0.322</td><td>0.364</td><td>0.340</td><td>0.394</td><td>0.358</td><td>0.397</td><td>0.476</td><td>0.458</td><td>0.346</td><td>0.388</td><td>3.755</td><td>1.525</td></tr><tr><td>192</td><td>0.372</td><td>0.392</td><td>0.393</td><td>0.405</td><td>0.402</td><td>0.414</td><td>0.877</td><td>0.656</td><td>0.492</td><td>0.492</td><td>0.404</td><td>0.414</td><td>0.482</td><td>0.479</td><td>0.429</td><td>0.439</td><td>0.512</td><td>0.493</td><td>0.456</td><td>0.452</td><td>5.602</td><td>1.931</td></tr><tr><td>336</td><td>0.386</td><td>0.414</td><td>0.427</td><td>0.436</td><td>0.452</td><td>0.452</td><td>1.043</td><td>0.731</td><td>0.607</td><td>0.555</td><td>0.435</td><td>0.445</td><td>0.591</td><td>0.541</td><td>0.496</td><td>0.487</td><td>0.552</td><td>0.551</td><td>0.482</td><td>0.486</td><td>4.721</td><td>1.835</td></tr><tr><td>720</td><td>0.412</td><td>0.434</td><td>0.436</td><td>0.450</td><td>0.462</td><td>0.468</td><td>1.104</td><td>0.763</td><td>0.824</td><td>0.655</td><td>0.447</td><td>0.458</td><td>0.839</td><td>0.661</td><td>0.463</td><td>0.474</td><td>0.562</td><td>0.560</td><td>0.515</td><td>0.511</td><td>3.647</td><td>1.625</td></tr><tr><td></td><td>Avg</td><td>0.364</td><td>0.395</td><td>0.391</td><td>0.411</td><td>0.414</td><td>0.427</td><td>0.942</td><td>0.684</td><td>0.574</td><td>0.531</td><td>0.402</td><td>0.420</td><td>0.563</td><td>0.519</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td>0.459</td><td>4.431</td><td>1.729</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.320</td><td>0.357</td><td>0.352</td><td>0.374</td><td>0.338</td><td>0.375</td><td>0.404</td><td>0.426</td><td>0.365</td><td>0.387</td><td>0.353</td><td>0.370</td><td>0.346</td><td>0.374</td><td>0.379</td><td>0.419</td><td>0.386</td><td>0.398</td><td>0.505</td><td>0.475</td><td>0.672</td><td>0.571</td></tr><tr><td>192</td><td>0.361</td><td>0.381</td><td>0.390</td><td>0.393</td><td>0.374</td><td>0.387</td><td>0.450</td><td>0.451</td><td>0.403</td><td>0.408</td><td>0.389</td><td>0.387</td><td>0.382</td><td>0.391</td><td>0.426</td><td>0.441</td><td>0.459</td><td>0.444</td><td>0.553</td><td>0.496</td><td>0.795</td><td>0.669</td></tr><tr><td>336</td><td>0.390</td><td>0.404</td><td>0.421</td><td>0.414</td><td>0.410</td><td>0.411</td><td>0.532</td><td>0.515</td><td>0.436</td><td>0.431</td><td>0.421</td><td>0.408</td><td>0.415</td><td>0.415</td><td>0.445</td><td>0.459</td><td>0.495</td><td>0.464</td><td>0.621</td><td>0.537</td><td>1.212</td><td>0.871</td></tr><tr><td>720</td><td>0.454</td><td>0.441</td><td>0.462</td><td>0.449</td><td>0.478</td><td>0.450</td><td>0.666</td><td>0.589</td><td>0.489</td><td>0.462</td><td>0.481</td><td>0.441</td><td>0.473</td><td>0.451</td><td>0.543</td><td>0.490</td><td>0.585</td><td>0.516</td><td>0.671</td><td>0.561</td><td>1.166</td><td>0.823</td></tr><tr><td></td><td>Avg</td><td>0.381</td><td>0.395</td><td>0.406</td><td>0.407</td><td>0.400</td><td>0.406</td><td>0.513</td><td>0.495</td><td>0.423</td><td>0.422</td><td>0.411</td><td>0.402</td><td>0.404</td><td>0.408</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td><td>0.961</td><td>0.734</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.175</td><td>0.258</td><td>0.183</td><td>0.270</td><td>0.187</td><td>0.267</td><td>0.287</td><td>0.366</td><td>0.197</td><td>0.296</td><td>0.183</td><td>0.266</td><td>0.193</td><td>0.293</td><td>0.203</td><td>0.287</td><td>0.192</td><td>0.274</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td></tr><tr><td>192</td><td>0.237</td><td>0.299</td><td>0.255</td><td>0.314</td><td>0.249</td><td>0.309</td><td>0.414</td><td>0.492</td><td>0.284</td><td>0.361</td><td>0.248</td><td>0.305</td><td>0.284</td><td>0.361</td><td>0.269</td><td>0.328</td><td>0.280</td><td>0.339</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td></tr><tr><td>336</td><td>0.298</td><td>0.340</td><td>0.309</td><td>0.347</td><td>0.321</td><td>0.351</td><td>0.597</td><td>0.542</td><td>0.381</td><td>0.429</td><td>0.309</td><td>0.343</td><td>0.382</td><td>0.429</td><td>0.325</td><td>0.366</td><td>0.334</td><td>0.361</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td></tr><tr><td>720</td><td>0.391</td><td>0.396</td><td>0.412</td><td>0.404</td><td>0.408</td><td>0.403</td><td>1.730</td><td>1.042</td><td>0.549</td><td>0.522</td><td>0.410</td><td>0.400</td><td>0.558</td><td>0.525</td><td>0.421</td><td>0.415</td><td>0.417</td><td>0.413</td><td>0.433</td><td>0.432</td><td>3.379</td><td>1.338</td></tr><tr><td></td><td>Avg</td><td>0.275</td><td>0.323</td><td>0.290</td><td>0.334</td><td>0.291</td><td>0.333</td><td>0.757</td><td>0.610</td><td>0.353</td><td>0.402</td><td>0.287</td><td>0.329</td><td>0.354</td><td>0.402</td><td>0.305</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.327</td><td>0.371</td><td>1.410</td><td>0.810</td></tr></table>


Table 14: Experiment results under hyperparameter searching for the long-term forecasting task. Avg is averaged from all four prediction lengths.


<table><tr><td colspan="2">Models</td><td colspan="2">TimeMixer (Ours)</td><td colspan="2">PatchTST 2023</td><td colspan="2">TimesNet 2023a</td><td colspan="2">Crossformer 2023</td><td colspan="2">MICN 2023</td><td colspan="2">FiLM 2022a</td><td colspan="2">DLinear 2023</td><td colspan="2">FEDformer 2022b</td><td colspan="2">Stationary 2022b</td><td colspan="2">Autoformer 2021</td><td colspan="2">Informer 2021</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.147</td><td>0.197</td><td>0.149</td><td>0.198</td><td>0.172</td><td>0.220</td><td>0.232</td><td>0.302</td><td>0.161</td><td>0.229</td><td>0.199</td><td>0.262</td><td>0.176</td><td>0.237</td><td>0.217</td><td>0.296</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td><td>0.300</td><td>0.384</td></tr><tr><td>192</td><td>0.189</td><td>0.239</td><td>0.194</td><td>0.241</td><td>0.219</td><td>0.261</td><td>0.371</td><td>0.410</td><td>0.220</td><td>0.281</td><td>0.228</td><td>0.288</td><td>0.220</td><td>0.282</td><td>0.276</td><td>0.336</td><td>0.245</td><td>0.285</td><td>0.307</td><td>0.367</td><td>0.598</td><td>0.544</td></tr><tr><td>336</td><td>0.241</td><td>0.280</td><td>0.306</td><td>0.282</td><td>0.246</td><td>0.337</td><td>0.495</td><td>0.515</td><td>0.278</td><td>0.331</td><td>0.267</td><td>0.323</td><td>0.265</td><td>0.319</td><td>0.339</td><td>0.380</td><td>0.321</td><td>0.338</td><td>0.359</td><td>0.395</td><td>0.578</td><td>0.523</td></tr><tr><td>720</td><td>0.310</td><td>0.330</td><td>0.314</td><td>0.334</td><td>0.365</td><td>0.359</td><td>0.526</td><td>0.542</td><td>0.311</td><td>0.356</td><td>0.319</td><td>0.361</td><td>0.323</td><td>0.362</td><td>0.403</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.419</td><td>0.428</td><td>1.059</td><td>0.741</td></tr><tr><td></td><td>Avg</td><td>0.222</td><td>0.262</td><td>0.241</td><td>0.264</td><td>0.251</td><td>0.294</td><td>0.406</td><td>0.442</td><td>0.242</td><td>0.299</td><td>0.253</td><td>0.309</td><td>0.246</td><td>0.300</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td><td>0.634</td><td>0.548</td></tr><tr><td rowspan="4">Solar-Energy</td><td>96</td><td>0.167</td><td>0.220</td><td>0.224</td><td>0.278</td><td>0.219</td><td>0.314</td><td>0.181</td><td>0.240</td><td>0.188</td><td>0.252</td><td>0.320</td><td>0.339</td><td>0.289</td><td>0.377</td><td>0.201</td><td>0.304</td><td>0.321</td><td>0.380</td><td>0.456</td><td>0.446</td><td>0.200</td><td>0.247</td></tr><tr><td>192</td><td>0.187</td><td>0.249</td><td>0.253</td><td>0.298</td><td>0.231</td><td>0.322</td><td>0.196</td><td>0.252</td><td>0.215</td><td>0.280</td><td>0.360</td><td>0.362</td><td>0.319</td><td>0.397</td><td>0.237</td><td>0.337</td><td>0.346</td><td>0.369</td><td>0.588</td><td>0.561</td><td>0.220</td><td>0.251</td></tr><tr><td>336</td><td>0.200</td><td>0.258</td><td>0.273</td><td>0.306</td><td>0.246</td><td>0.337</td><td>0.216</td><td>0.243</td><td>0.222</td><td>0.267</td><td>0.398</td><td>0.375</td><td>0.352</td><td>0.415</td><td>0.254</td><td>0.362</td><td>0.357</td><td>0.387</td><td>0.595</td><td>0.588</td><td>0.260</td><td>0.287</td></tr><tr><td>720</td><td>0.215</td><td>0.250</td><td>0.272</td><td>0.308</td><td>0.280</td><td>0.363</td><td>0.220</td><td>0.256</td><td>0.226</td><td>0.264</td><td>0.399</td><td>0.368</td><td>0.356</td><td>0.412</td><td>0.280</td><td>0.397</td><td>0.335</td><td>0.384</td><td>0.733</td><td>0.633</td><td>0.244</td><td>0.301</td></tr><tr><td></td><td>Avg</td><td>0.192</td><td>0.244</td><td>0.256</td><td>0.298</td><td>0.244</td><td>0.334</td><td>0.204</td><td>0.248</td><td>0.213</td><td>0.266</td><td>0.369</td><td>0.361</td><td>0.329</td><td>0.400</td><td>0.243</td><td>0.350</td><td>0.340</td><td>0.380</td><td>0.593</td><td>0.557</td><td>0.231</td><td>0.272</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.129</td><td>0.224</td><td>0.129</td><td>0.222</td><td>0.168</td><td>0.272</td><td>0.150</td><td>0.251</td><td>0.164</td><td>0.269</td><td>0.154</td><td>0.267</td><td>0.140</td><td>0.237</td><td>0.193</td><td>0.308</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td><td>0.274</td><td>0.368</td></tr><tr><td>192</td><td>0.140</td><td>0.220</td><td>0.147</td><td>0.240</td><td>0.184</td><td>0.322</td><td>0.161</td><td>0.260</td><td>0.177</td><td>0.285</td><td>0.164</td><td>0.258</td><td>0.153</td><td>0.249</td><td>0.201</td><td>0.315</td><td>0.182</td><td>0.286</td><td>0.222</td><td>0.334</td><td>0.296</td><td>0.386</td></tr><tr><td>336</td><td>0.161</td><td>0.255</td><td>0.163</td><td>0.259</td><td>0.198</td><td>0.300</td><td>0.182</td><td>0.281</td><td>0.193</td><td>0.304</td><td>0.188</td><td>0.283</td><td>0.169</td><td>0.267</td><td>0.214</td><td>0.329</td><td>0.200</td><td>0.304</td><td>0.231</td><td>0.338</td><td>0.300</td><td>0.394</td></tr><tr><td>720</td><td>0.194</td><td>0.287</td><td>0.197</td><td>0.290</td><td>0.220</td><td>0.320</td><td>0.251</td><td>0.339</td><td>0.212</td><td>0.321</td><td>0.236</td><td>0.332</td><td>0.203</td><td>0.301</td><td>0.246</td><td>0.355</td><td>0.222</td><td>0.321</td><td>0.254</td><td>0.361</td><td>0.373</td><td>0.439</td></tr><tr><td></td><td>Avg</td><td>0.156</td><td>0.246</td><td>0.159</td><td>0.253</td><td>0.192</td><td>0.295</td><td>0.186</td><td>0.283</td><td>0.186</td><td>0.295</td><td>0.186</td><td>0.285</td><td>0.166</td><td>0.264</td><td>0.214</td><td>0.321</td><td>0.213</td><td>0.296</td><td>0.227</td><td>0.338</td><td>0.311</td><td>0.397</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.360</td><td>0.249</td><td>0.360</td><td>0.249</td><td>0.593</td><td>0.321</td><td>0.514</td><td>0.267</td><td>0.519</td><td>0.309</td><td>0.416</td><td>0.294</td><td>0.410</td><td>0.282</td><td>0.587</td><td>0.366</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td><td>0.719</td><td>0.391</td></tr><tr><td>192</td><td>0.375</td><td>0.250</td><td>0.379</td><td>0.256</td><td>0.617</td><td>0.336</td><td>0.549</td><td>0.252</td><td>0.537</td><td>0.315</td><td>0.408</td><td>0.288</td><td>0.423</td><td>0.287</td><td>0.604</td><td>0.373</td><td>0.613</td><td>0.340</td><td>0.616</td><td>0.382</td><td>0.696</td><td>0.379</td></tr><tr><td>336</td><td>0.385</td><td>0.270</td><td>0.392</td><td>0.264</td><td>0.629</td><td>0.336</td><td>0.530</td><td>0.300</td><td>0.534</td><td>0.313</td><td>0.425</td><td>0.298</td><td>0.436</td><td>0.296</td><td>0.621</td><td>0.383</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.337</td><td>0.777</td><td>0.420</td></tr><tr><td>720</td><td>0.430</td><td>0.281</td><td>0.432</td><td>0.286</td><td>0.640</td><td>0.350</td><td>0.573</td><td>0.313</td><td>0.577</td><td>0.325</td><td>0.520</td><td>0.353</td><td>0.466</td><td>0.315</td><td>0.626</td><td>0.382</td><td>0.653</td><td>0.355</td><td>0.660</td><td>0.408</td><td>0.864</td><td>0.472</td></tr><tr><td></td><td>Avg</td><td>0.387</td><td>0.262</td><td>0.391</td><td>0.264</td><td>0.620</td><td>0.336</td><td>0.542</td><td>0.283</td><td>0.541</td><td>0.315</td><td>0.442</td><td>0.308</td><td>0.434</td><td>0.295</td><td>0.609</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td><td>0.764</td><td>0.415</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.361</td><td>0.390</td><td>0.370</td><td>0.400</td><td>0.384</td><td>0.402</td><td>0.418</td><td>0.438</td><td>0.421</td><td>0.431</td><td>0.422</td><td>0.432</td><td>0.375</td><td>0.399</td><td>0.376</td><td>0.419</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td><td>0.865</td><td>0.713</td></tr><tr><td>192</td><td>0.409</td><td>0.414</td><td>0.413</td><td>0.429</td><td>0.436</td><td>0.429</td><td>0.539</td><td>0.517</td><td>0.474</td><td>0.487</td><td>0.462</td><td>0.458</td><td>0.405</td><td>0.416</td><td>0.420</td><td>0.448</td><td>0.534</td><td>0.504</td><td>0.500</td><td>0.482</td><td>1.008</td><td>0.792</td></tr><tr><td>336</td><td>0.430</td><td>0.429</td><td>0.422</td><td>0.440</td><td>0.638</td><td>0.469</td><td>0.709</td><td>0.638</td><td>0.569</td><td>0.551</td><td>0.501</td><td>0.483</td><td>0.439</td><td>0.443</td><td>0.459</td><td>0.465</td><td>0.588</td><td>0.535</td><td>0.521</td><td>0.496</td><td>1.107</td><td>0.809</td></tr><tr><td>720</td><td>0.445</td><td>0.460</td><td>0.447</td><td>0.468</td><td>0.521</td><td>0.500</td><td>0.733</td><td>0.636</td><td>0.770</td><td>0.672</td><td>0.544</td><td>0.526</td><td>0.472</td><td>0.490</td><td>0.506</td><td>0.507</td><td>0.643</td><td>0.616</td><td>0.514</td><td>0.512</td><td>1.181</td><td>0.865</td></tr><tr><td></td><td>Avg</td><td>0.411</td><td>0.423</td><td>0.413</td><td>0.434</td><td>0.458</td><td>0.450</td><td>0.600</td><td>0.557</td><td>0.558</td><td>0.535</td><td>0.482</td><td>0.475</td><td>0.423</td><td>0.437</td><td>0.440</td><td>0.460</td><td>0.57</td><td>0.536</td><td>0.496</td><td>0.487</td><td>1.040</td><td>0.795</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.271</td><td>0.330</td><td>0.274</td><td>0.337</td><td>0.340</td><td>0.374</td><td>0.425</td><td>0.463</td><td>0.299</td><td>0.364</td><td>0.323</td><td>0.370</td><td>0.289</td><td>0.353</td><td>0.346</td><td>0.388</td><td>0.476</td><td>0.458</td><td>0.358</td><td>0.397</td><td>3.755</td><td>1.525</td></tr><tr><td>192</td><td>0.317</td><td>0.402</td><td>0.314</td><td>0.382</td><td>0.231</td><td>0.322</td><td>0.473</td><td>0.500</td><td>0.441</td><td>0.454</td><td>0.391</td><td>0.415</td><td>0.383</td><td>0.418</td><td>0.429</td><td>0.439</td><td>0.512</td><td>0.493</td><td>0.456</td><td>0.452</td><td>5.602</td><td>1.931</td></tr><tr><td>336</td><td>0.332</td><td>0.396</td><td>0.329</td><td>0.384</td><td>0.452</td><td>0.452</td><td>0.581</td><td>0.562</td><td>0.654</td><td>0.567</td><td>0.415</td><td>0.440</td><td>0.448</td><td>0.465</td><td>0.496</td><td>0.487</td><td>0.552</td><td>0.551</td><td>0.482</td><td>0.486</td><td>4.721</td><td>1.835</td></tr><tr><td>720</td><td>0.342</td><td>0.408</td><td>0.379</td><td>0.422</td><td>0.462</td><td>0.468</td><td>0.775</td><td>0.665</td><td>0.956</td><td>0.716</td><td>0.441</td><td>0.459</td><td>0.605</td><td>0.551</td><td>0.463</td><td>0.474</td><td>0.562</td><td>0.560</td><td>0.515</td><td>0.511</td><td>3.647</td><td>1.625</td></tr><tr><td></td><td>Avg</td><td>0.316</td><td>0.384</td><td>0.324</td><td>0.381</td><td>0.371</td><td>0.404</td><td>0.564</td><td>0.548</td><td>0.588</td><td>0.525</td><td>0.393</td><td>0.421</td><td>0.431</td><td>0.447</td><td>0.433</td><td>0.447</td><td>0.526</td><td>0.516</td><td>0.453</td><td>0.462</td><td>4.431</td><td>1.729</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.291</td><td>0.340</td><td>0.293</td><td>0.346</td><td>0.338</td><td>0.375</td><td>0.361</td><td>0.403</td><td>0.316</td><td>0.362</td><td>0.302</td><td>0.345</td><td>0.299</td><td>0.343</td><td>0.379</td><td>0.419</td><td>0.386</td><td>0.398</td><td>0.505</td><td>0.475</td><td>0.672</td><td>0.571</td></tr><tr><td>192</td><td>0.327</td><td>0.365</td><td>0.333</td><td>0.370</td><td>0.374</td><td>0.387</td><td>0.387</td><td>0.422</td><td>0.363</td><td>0.390</td><td>0.338</td><td>0.368</td><td>0.335</td><td>0.365</td><td>0.426</td><td>0.441</td><td>0.459</td><td>0.444</td><td>0.553</td><td>0.496</td><td>0.795</td><td>0.669</td></tr><tr><td>336</td><td>0.360</td><td>0.381</td><td>0.369</td><td>0.392</td><td>0.410</td><td>0.411</td><td>0.605</td><td>0.572</td><td>0.408</td><td>0.426</td><td>0.373</td><td>0.388</td><td>0.369</td><td>0.386</td><td>0.445</td><td>0.459</td><td>0.495</td><td>0.464</td><td>0.621</td><td>0.537</td><td>1.212</td><td>0.871</td></tr><tr><td>720</td><td>0.415</td><td>0.417</td><td>0.416</td><td>0.420</td><td>0.478</td><td>0.450</td><td>0.703</td><td>0.645</td><td>0.481</td><td>0.476</td><td>0.420</td><td>0.420</td><td>0.425</td><td>0.421</td><td>0.543</td><td>0.490</td><td>0.585</td><td>0.516</td><td>0.671</td><td>0.561</td><td>1.166</td><td>0.823</td></tr><tr><td></td><td>Avg</td><td>0.348</td><td>0.375</td><td>0.353</td><td>0.382</td><td>0.353</td><td>0.382</td><td>0.514</td><td>0.510</td><td>0.392</td><td>0.413</td><td>0.358</td><td>0.38</td><td>0.357</td><td>0.379</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td><td>0.961</td><td>0.733</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.164</td><td>0.254</td><td>0.166</td><td>0.256</td><td>0.187</td><td>0.267</td><td>0.275</td><td>0.358</td><td>0.179</td><td>0.275</td><td>0.165</td><td>0.256</td><td>0.167</td><td>0.260</td><td>0.203</td><td>0.287</td><td>0.192</td><td>0.274</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td></tr><tr><td>192</td><td>0.223</td><td>0.295</td><td>0.223</td><td>0.296</td><td>0.249</td><td>0.309</td><td>0.345</td><td>0.400</td><td>0.307</td><td>0.376</td><td>0.222</td><td>0.296</td><td>0.224</td><td>0.303</td><td>0.269</td><td>0.328</td><td>0.280</td><td>0.339</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td></tr><tr><td>336</td><td>0.279</td><td>0.330</td><td>0.274</td><td>0.329</td><td>0.321</td><td>0.351</td><td>0.657</td><td>0.528</td><td>0.325</td><td>0.388</td><td>0.277</td><td>0.333</td><td>0.281</td><td>0.342</td><td>0.325</td><td>0.366</td><td>0.334</td><td>0.361</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td></tr><tr><td>720</td><td>0.359</td><td>0.383</td><td>0.362</td><td>0.385</td><td>0.408</td><td>0.403</td><td>1.208</td><td>0.753</td><td>0.502</td><td>0.490</td><td>0.371</td><td>0.389</td><td>0.397</td><td>0.421</td><td>0.421</td><td>0.415</td><td>0.417</td><td>0.413</td><td>0.422</td><td>0.419</td><td>3.379</td><td>1.388</td></tr><tr><td></td><td>Avg</td><td>0.256</td><td>0.315</td><td>0.256</td><td>0.317</td><td>0.291</td><td>0.333</td><td>0.621</td><td>0.510</td><td>0.328</td><td>0.382</td><td>0.259</td><td>0.319</td><td>0.267</td><td>0.332</td><td>0.304</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.324</td><td>0.368</td><td>1.410</td><td>0.823</td></tr></table>

## F FULL ABLATIONS

Here we provide the complete results of ablations and alternative designs for TimeMixer. 

## F.1 ABLATIONS OF EACH DESIGN IN TIMEMIXER

To verify the effectiveness of our design in TimeMixer, we conduct comprehensive ablations for all benchmarks. All the results are provided in Table 15, 16, 17 as a supplement to Table 5 of main text. 


Table 15: Ablations on both Past-Decompose-Mixing (Decompose, Season Mixing, and Trend Mixing) and Future-Multipredictor-Mixing blocks in predict-336 setting for all long-term benchmarks. $\nearrow$ indicates the bottom-up mixing while $\swarrow$ indicates top-down. A check mark $\checkmark$ and a wrong mark × indicate with and without certain components respectively. ① is the official design in TimeMixer.


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Weather</td><td colspan="2">Solar</td><td colspan="2">Electricity</td><td colspan="2">Traffic</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>1</td><td>✓</td><td>↗</td><td>√</td><td>✓</td><td>0.484</td><td>0.458</td><td>0.386</td><td>0.414</td><td>0.390</td><td>0.404</td><td>0.298</td><td>0.340</td><td>0.251</td><td>0.287</td><td>0.231</td><td>0.292</td><td>0.185</td><td>0.277</td><td>0.498</td><td>0.296</td></tr><tr><td>2</td><td>✓</td><td>↗</td><td>√</td><td>×</td><td>0.493</td><td>0.472</td><td>0.399</td><td>0.426</td><td>0.402</td><td>0.415</td><td>0.311</td><td>0.357</td><td>0.262</td><td>0.308</td><td>0.267</td><td>0.339</td><td>0.198</td><td>0.301</td><td>0.518</td><td>0.337</td></tr><tr><td>3</td><td>✓</td><td>×</td><td>√</td><td>✓</td><td>0.507</td><td>0.490</td><td>0.408</td><td>0.437</td><td>0.411</td><td>0.427</td><td>0.322</td><td>0.366</td><td>0.273</td><td>0.321</td><td>0.274</td><td>0.355</td><td>0.207</td><td>0.304</td><td>0.532</td><td>0.348</td></tr><tr><td>4</td><td>✓</td><td>↗</td><td>×</td><td>✓</td><td>0.491</td><td>0.483</td><td>0.397</td><td>0.424</td><td>0.405</td><td>0.414</td><td>0.317</td><td>0.351</td><td>0.269</td><td>0.311</td><td>0.268</td><td>0.341</td><td>0.200</td><td>0.299</td><td>0.525</td><td>0.339</td></tr><tr><td>5</td><td>✓</td><td>√</td><td>√</td><td>✓</td><td>0.488</td><td>0.466</td><td>0.393</td><td>0.426</td><td>0.392</td><td>0.413</td><td>0.309</td><td>0.349</td><td>0.257</td><td>0.293</td><td>0.252</td><td>0.330</td><td>0.191</td><td>0.293</td><td>0.520</td><td>0.331</td></tr><tr><td>6</td><td>✓</td><td>↗</td><td>↗</td><td>✓</td><td>0.493</td><td>0.484</td><td>0.401</td><td>0.432</td><td>0.396</td><td>0.415</td><td>0.319</td><td>0.361</td><td>0.271</td><td>0.322</td><td>0.281</td><td>0.363</td><td>0.214</td><td>0.307</td><td>0.541</td><td>0.351</td></tr><tr><td>7</td><td>✓</td><td>√</td><td>↗</td><td>✓</td><td>0.498</td><td>0.491</td><td>0.421</td><td>0.436</td><td>0.412</td><td>0.429</td><td>0.321</td><td>0.369</td><td>0.277</td><td>0.332</td><td>0.298</td><td>0.375</td><td>0.221</td><td>0.319</td><td>0.564</td><td>0.357</td></tr><tr><td>8</td><td>×</td><td></td><td>↗</td><td>✓</td><td>0.494</td><td>0.488</td><td>0.396</td><td>0.421</td><td>0.395</td><td>0.408</td><td>0.313</td><td>0.360</td><td>0.259</td><td>0.308</td><td>0.260</td><td>0.321</td><td>0.199</td><td>0.303</td><td>0.522</td><td>0.340</td></tr><tr><td>9</td><td>×</td><td></td><td>√</td><td>✓</td><td>0.487</td><td>0.462</td><td>0.394</td><td>0.419</td><td>0.393</td><td>0.406</td><td>0.307</td><td>0.354</td><td>0.261</td><td>0.327</td><td>0.257</td><td>0.334</td><td>0.196</td><td>0.310</td><td>0.526</td><td>0.339</td></tr><tr><td>10</td><td>×</td><td></td><td>×</td><td>✓</td><td>0.502</td><td>0.489</td><td>0.411</td><td>0.427</td><td>0.405</td><td>0.412</td><td>0.319</td><td>0.358</td><td>0.273</td><td>0.331</td><td>0.295</td><td>0.336</td><td>0.217</td><td>0.318</td><td>0.558</td><td>0.347</td></tr></table>


Table 16: Ablations on both Past-Decompose-Mixing (Decompose, Season Mixing, and Trend Mixing) and Future-Multipredictor-Mixing blocks in the M4 short-term forecasting benchmark. Case notations are same as Table 5 and 15. ① is the official design in TimeMixer.


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">M4</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>SMAPE</td><td>MASE</td><td>OWA</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>↙</td><td>√</td><td>11.723</td><td>1.559</td><td>0.840</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>↙</td><td>×</td><td>12.503</td><td>1.634</td><td>0.925</td></tr><tr><td>3</td><td>√</td><td>×</td><td>↙</td><td>√</td><td>13.051</td><td>1.676</td><td>0.962</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>12.911</td><td>1.655</td><td>0.941</td></tr><tr><td>5</td><td>√</td><td>↙</td><td>↙</td><td>√</td><td>12.008</td><td>1.628</td><td>0.871</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>11.978</td><td>1.626</td><td>0.859</td></tr><tr><td>7</td><td>√</td><td>↙</td><td>↗</td><td>√</td><td>13.012</td><td>1.657</td><td>0.954</td></tr><tr><td>8</td><td>×</td><td colspan="2">↗</td><td>√</td><td>11.975</td><td>1.617</td><td>0.851</td></tr><tr><td>9</td><td>×</td><td colspan="2">↙</td><td>√</td><td>11.973</td><td>1.622</td><td>0.850</td></tr><tr><td>10</td><td>×</td><td colspan="2">×</td><td>√</td><td>12.468</td><td>1.671</td><td>0.916</td></tr></table>


Table 17: Ablations on both Past-Decompose-Mixing (Decompose, Season Mixing, and Trend Mixing) and Future-Multipredictor-Mixing blocks in the PEMS short-term forecasting benchmarks. Case notations are same as Table 5 and 15. ① is the official design in TimeMixer.


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">PEMS03</td><td colspan="3">PEMS04</td><td colspan="3">PEMS07</td><td colspan="3">PEMS08</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MAE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>↘</td><td>√</td><td>14.63</td><td>14.54</td><td>23.28</td><td>19.21</td><td>12.53</td><td>30.92</td><td>20.57</td><td>8.62</td><td>33.59</td><td>15.22</td><td>9.67</td><td>24.26</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>↘</td><td>×</td><td>15.66</td><td>15.81</td><td>25.77</td><td>21.67</td><td>13.45</td><td>34.89</td><td>22.78</td><td>9.52</td><td>35.57</td><td>17.48</td><td>10.91</td><td>27.84</td></tr><tr><td>3</td><td>√</td><td>×</td><td>↘</td><td>√</td><td>18.90</td><td>17.33</td><td>30.75</td><td>24.49</td><td>16.28</td><td>38.79</td><td>25.27</td><td>10.74</td><td>40.06</td><td>19.02</td><td>11.71</td><td>30.05</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>17.67</td><td>17.58</td><td>28.48</td><td>22.91</td><td>15.02</td><td>37.14</td><td>24.81</td><td>10.02</td><td>38.68</td><td>18.29</td><td>12.21</td><td>28.62</td></tr><tr><td>5</td><td>√</td><td>↘</td><td>↘</td><td>√</td><td>15.46</td><td>15.73</td><td>24.91</td><td>20.78</td><td>13.02</td><td>32.47</td><td>22.57</td><td>9.33</td><td>35.87</td><td>16.54</td><td>9.97</td><td>26.88</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>15.32</td><td>15.41</td><td>24.83</td><td>21.09</td><td>13.78</td><td>33.11</td><td>21.94</td><td>9.41</td><td>35.40</td><td>17.01</td><td>10.82</td><td>26.93</td></tr><tr><td>7</td><td>√</td><td>↘</td><td>↗</td><td>√</td><td>18.81</td><td>17.29</td><td>29.78</td><td>22.27</td><td>15.14</td><td>34.67</td><td>25.11</td><td>10.60</td><td>39.74</td><td>18.74</td><td>12.09</td><td>28.67</td></tr><tr><td>8</td><td>×</td><td></td><td>↗</td><td>√</td><td>15.57</td><td>15.62</td><td>24.98</td><td>21.51</td><td>13.47</td><td>34.81</td><td>22.94</td><td>9.81</td><td>35.49</td><td>18.17</td><td>11.02</td><td>28.14</td></tr><tr><td>9</td><td>×</td><td></td><td>↘</td><td>√</td><td>15.48</td><td>15.55</td><td>24.83</td><td>21.79</td><td>14.03</td><td>35.23</td><td>21.93</td><td>9.91</td><td>36.02</td><td>17.71</td><td>10.88</td><td>27.91</td></tr><tr><td>10</td><td>×</td><td></td><td>×</td><td>√</td><td>19.01</td><td>18.58</td><td>30.06</td><td>24.87</td><td>16.66</td><td>39.48</td><td>24.72</td><td>9.97</td><td>37.18</td><td>19.18</td><td>12.21</td><td>30.79</td></tr></table>

Implementations We implement the following 10 types of ablations: 

• Offical design in TimeMixer (case ①). 

• Ablations on Future Mixing (case ②): In this case, we only adopt a single predictor to the finest scale features, that is xˆ = Predictor<sub>0</sub>(x<sup>L</sup>). 

• Ablations on Past Mixing (case ③-⑦): Firstly, we remove the mixing operation of TimeMixer in seasonal and trend parts respectively (case ③-④), that is removing Bottom-Up-Mixing or Top-Down-Mixing layer. Then, we reverse the mixing directions for seasonal and trend parts (case ⑤-⑦), which means adopting Bottom-Up-Mixing layer to trend and Top-Down-Mixing layer to seasonal part. 

• Ablations on Decomposition (case ⑧-⑩): In these cases, we do not adopt the decomposition, which means that there is only one single feature for each scale. Thus, we can only try one single mixing direction for these features, that is bottom-up mixing in case ⑧, top-down mixing in case ⑨. Besides, we also test the case that is without mixing in ⑩, where the interactions among multiscale features are removed. 

Analysis In all ablations, we can find that the official design in TimeMixer performs best, which provides solid support to our insights in special mixing approaches. Notably, it is observed that completely reversing mixing directions for seasonal and trend parts (case ⑦) leads to a seriously per formance drop. This may come from that the essential microscopic information in finer-scale seasons and macroscopic information in coarser-scale trends are ruined by unsuitable mixing approaches. 

## F.2 ALTERNATIVE DECOMPOSITION METHODS

In this paper, we adopt the moving-average-based season-trend decomposition, which is widely used in previous work, such as Autoformer (Wu et al., 2021), FEDformer (Zhou et al., 2022b) and DLinear (Zeng et al., 2023). It is notable that Discrete Fourier Transformer (DFT) have been widely recognized in time series analysis. Thus, we also try the DFT-based decomposition as a substitute. Here we present two types of experiments. 

The first one is DFT-based high- and low-frequency decomposition. We treat the high-frequency part like the seasonal part in TimeMixer and the low-frequency part like the trend part. The results are shown in Table 18. It observed that DFT-based decomposition performs worse than our design in TimeMixer. Since we only explore the proper mixing approach for decomposed seasonal and trend parts in the paper, the bottom-up and top-down mixing strategies may be not suitable for high- and low-frequency parts. New visualizations like Figure 3 and 4 are expected to provide insights to the model design. Thus, we would like to leave the exploration of DFT-based high- and low-frequency decomposition methods as the future work. 

The second one is to enhance season-trend decomposition with DFT. Here we present the DFT-based season-trend decomposition. Firstly, we transform the raw series into a frequency domain by DFT and then extract the most significant frequencies. After transforming the selected frequencies by inverse DFT, we obtain the seasonal part of the time series. Then the trend part is the raw series minus the seasonal part. We can find that this superior decomposition method surpasses the moving-average design. However, since moving average is quite simple and easy to implement with PyTorch, we eventually chose the moving-average-based season-trend decomposition in TimeMixer, which can also achieve a favorable balance between performance and efficiency. 


Table 18: Alternative decomposition methods in M4, PEMS04 and predict-336 setting of ETTm1.


<table><tr><td rowspan="2">Decomposition methods</td><td colspan="2">ETTm1</td><td colspan="3">M4</td><td colspan="3">PEMS04</td></tr><tr><td>MSE</td><td>MAE</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>MAE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>DFT-based high- and low-frequency decomposition</td><td>0.392</td><td>0.404</td><td>12.054</td><td>1.632</td><td>0.862</td><td>19.83</td><td>12.74</td><td>31.48</td></tr><tr><td>DFT-based season-trend decomposition</td><td>0.383</td><td>0.399</td><td>11.673</td><td>1.536</td><td>0.824</td><td>18.91</td><td>12.27</td><td>29.47</td></tr><tr><td>Moving-average-based season-trend decomposition (TimeMixer)</td><td>0.390</td><td>0.404</td><td>11.723</td><td>1.559</td><td>0.840</td><td>19.21</td><td>12.53</td><td>30.92</td></tr></table>

## F.3 ALTERNATIVE DOWNSAMPLING METHODS

As we stated in Section 3.1, we adopt the average pooling to obtain the multiscale series. Here we replace this operation with 1D convolutions. From Table 19, we can find that the complicated 1D-convolution-based outperforms average pooling slightly. But considering both performance and efficiency, we eventually use average pooling in TimeMixer. 


Table 19: Alternative downsampling methods in M4, PEMS04 and predict-336 setting of ETTm1.


<table><tr><td rowspan="2">Downsampling methods</td><td colspan="2">ETTm1</td><td colspan="3">M4</td><td colspan="3">PEMS04</td></tr><tr><td>MSE</td><td>MAE</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>MAE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>Moving average</td><td>0.390</td><td>0.404</td><td>11.723</td><td>1.559</td><td>0.840</td><td>19.21</td><td>12.53</td><td>30.92</td></tr><tr><td>1D convolutions with stride as 2</td><td>0.387</td><td>0.401</td><td>11.682</td><td>1.542</td><td>0.831</td><td>19.04</td><td>12.17</td><td>29.88</td></tr></table>

## F.4 ALTERNATIVE ENSEMBLE STRATEGIES

In the main text, we sum the outputs from multiple predictors towards the final result (Eq. 6). Here we also try the average strategy. Note that in TimeMixer, the loss is calculated based on the ensemble results, not for each predictor, that is $\begin{array} { r } { \| \mathbf { x } - \hat { \mathbf { x } } \| = \| \mathbf { x } - \sum _ { m = 0 } ^ { M } \hat { \mathbf { x } } _ { m } \| } \end{array}$ . When we change the ensemble strategy as average, the loss will be $\begin{array} { r } { \| \mathbf { x } - \hat { \mathbf { x } } \| = \| \mathbf { x } - \frac { 1 } { M + 1 } \sum _ { m = 0 } ^ { M } \hat { \mathbf { x } } _ { m } \| } \end{array}$ . Obviously, the difference between average and mean strategies is only a constant multiple. 

It is common sense in the deep learning community that deep models can easily fit constant multiple. For example, if we replace the “sum” with “average”, under the same supervision, the deep model can easily fit this change by learning the parameters of each predictor equal to the $\frac { 1 } { M + 1 }$ of the “sum” case, which means these two designs are equivalent in learning the final prediction under the deep model aspect. Besides, we also provide the experiment results in Table 20, where we can find that the performances of these two strategies are almost the same. 


Table 20: Alternative ensemble strategies in M4, PEMS04 and predict-336 setting of ETTm1.


<table><tr><td rowspan="2">Ensemble strategies</td><td colspan="2">ETTm1</td><td colspan="3">M4</td><td colspan="3">PEMS04</td></tr><tr><td>MSE</td><td>MAE</td><td>SMAPE</td><td>MASE</td><td>OWA</td><td>MAE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>Sum ensemble</td><td>0.390</td><td>0.404</td><td>11.723</td><td>1.559</td><td>0.840</td><td>19.21</td><td>12.53</td><td>30.92</td></tr><tr><td>Average ensemble</td><td>0.391</td><td>0.407</td><td>11.742</td><td>1.573</td><td>0.851</td><td>19.17</td><td>12.45</td><td>30.88</td></tr></table>

## F.5 ABLATIONS ON LARGER SCALES AND LARGER INPUT LENGTH SETTINGS

In the previous section (Table 15, 16, 17), we have conducted comprehensive ablations under the unified configuration presented in Table 7, which M is set to 1 for short-term forecasting and input length is set to 96 for short-term forecasting. To further evaluate the effectiveness of our proposed module, we also provide additional ablations on larger scales for short-term forecasting and larger input length settings in Table 21 as a supplement to Table 5 of the main text. Besides, we also provide a detailed analysis of relative promotion (Table 22), where we can find the following observations: 

• All the designs of TimeMixer are effective in both hyperparameter settings. Especially, the seasonal mixing (case ③) and proper mixing directions (case ⑦) are essential. 

• As shown in Table 22, in large M and longer input length situations, the relative promotions brought by seasonal and trend mixing in PDM and FMM are more significant in most cases, which further verifies the effectiveness of our design. 

• It is observed that the seasonal mixing direction contribution (case ⑤) is much more significant in the longer-input setting on the ETTm1 dataset. This may come from that input-96 only corresponds to one day in 15-minutely sampled ETTm1, while input-336 maintains 3.5 days of information (around 3.5 periods). Thus, the bottom-up mixing direction will benefit from sufficient microscopic seasonal information under the longer-input setting. 


Table 21: A supplement to Table 5 of the main text with ablations on both PDM (Decompose, Season Mixing, Trend Mixing) and FMM blocks in PEMS04 with M = 3 and predict-336 setting of ETTm1 with input-336. Since the input length of M4 is fixed to a small value, larger M may result in a meaningless configuration. We only experiment on PEMS04 here.


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">PEMS04 with M = 3</td><td colspan="2">ETTm1 with input 336</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MSE</td><td>MAE</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>√</td><td>√</td><td>18.10</td><td>11.73</td><td>28.51</td><td>0.360</td><td>0.381</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>√</td><td>×</td><td>21.49</td><td>13.12</td><td>33.48</td><td>0.375</td><td>0.398</td></tr><tr><td>3</td><td>√</td><td>×</td><td>√</td><td>√</td><td>23.68</td><td>16.01</td><td>37.42</td><td>0.390</td><td>0.415</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>22.44</td><td>14.81</td><td>36.54</td><td>0.386</td><td>0.410</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>20.41</td><td>13.08</td><td>31.92</td><td>0.371</td><td>0.389</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>21.28</td><td>13.19</td><td>32.84</td><td>0.370</td><td>0.388</td></tr><tr><td>7</td><td>√</td><td>√</td><td>↗</td><td>√</td><td>22.16</td><td>14.60</td><td>35.42</td><td>0.384</td><td>0.409</td></tr><tr><td>8</td><td>×</td><td colspan="2">↗</td><td>√</td><td>20.98</td><td>13.12</td><td>33.94</td><td>0.372</td><td>0.396</td></tr><tr><td>9</td><td>×</td><td colspan="2">√</td><td>√</td><td>20.66</td><td>13.06</td><td>32.74</td><td>0.374</td><td>0.398</td></tr><tr><td>10</td><td>×</td><td colspan="2">×</td><td>√</td><td>24.16</td><td>16.21</td><td>38.04</td><td>0.401</td><td>0.414</td></tr></table>


Table 22: Relative promotion analysis on ablations under different hyperparameter configurations. For example, the relative promotion is calculated by (1−①/②) in case ②.


<table><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">PEMS04 with M = 3</td><td colspan="2">ETTm1 with input 336</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MSE</td><td>MAE</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>√</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>√</td><td>×</td><td>15.8%</td><td>10.6%</td><td>14.9%</td><td>4.1%</td><td>4.2%</td></tr><tr><td>3</td><td>√</td><td>×</td><td>√</td><td>√</td><td>23.6%</td><td>26.7%</td><td>23.8%</td><td>7.7%</td><td>8.2%</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>19.2%</td><td>20.1%</td><td>22.0%</td><td>6.8%</td><td>7.1%</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>11.4%</td><td>10.4%</td><td>10.7%</td><td>3.0%</td><td>2.1%</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>15.0%</td><td>11.1%</td><td>13.2%</td><td>2.7%</td><td>1.8%</td></tr><tr><td>7</td><td>√</td><td>√</td><td>↗</td><td>√</td><td>18.4%</td><td>19.7%</td><td>19.5%</td><td>6.2%</td><td>6.9%</td></tr><tr><td>8</td><td>×</td><td>↗</td><td></td><td>√</td><td>13.7%</td><td>10.6%</td><td>15.9%</td><td>3.2%</td><td>3.8%</td></tr><tr><td>9</td><td>×</td><td>√</td><td></td><td>√</td><td>12.4%</td><td>10.2%</td><td>13.0%</td><td>3.7%</td><td>4.2%</td></tr><tr><td>10</td><td>×</td><td>×</td><td></td><td>√</td><td>25.1%</td><td>27.6%</td><td>25.0%</td><td>10.2%</td><td>8.0%</td></tr><tr><td rowspan="2">Case</td><td rowspan="2">Decompose</td><td colspan="2">Past mixing</td><td>Future mixing</td><td colspan="3">PEMS04 with M = 1</td><td colspan="2">ETTm1 with input 96</td></tr><tr><td>Seasonal</td><td>Trend</td><td>Multipredictor</td><td>MAE</td><td>MAPE</td><td>RMSE</td><td>MSE</td><td>MAE</td></tr><tr><td>1</td><td>√</td><td>↗</td><td>√</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>2</td><td>√</td><td>↗</td><td>√</td><td>×</td><td>11.4%</td><td>6.8%</td><td>11.2%</td><td>2.9%</td><td>2.6%</td></tr><tr><td>3</td><td>√</td><td>×</td><td>√</td><td>√</td><td>21.6%</td><td>23.1%</td><td>20.2%</td><td>5.1%</td><td>5.4%</td></tr><tr><td>4</td><td>√</td><td>↗</td><td>×</td><td>√</td><td>16.2%</td><td>16.6%</td><td>16.7%</td><td>3.7%</td><td>2.4%</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>7.6%</td><td>3.7%</td><td>4.7%</td><td>0.5%</td><td>2.1%</td></tr><tr><td>6</td><td>√</td><td>↗</td><td>↗</td><td>√</td><td>8.9%</td><td>9.0%</td><td>6.6%</td><td>1.5%</td><td>2.6%</td></tr><tr><td>7</td><td>√</td><td>√</td><td>↗</td><td>√</td><td>13.7%</td><td>17.2%</td><td>10.8%</td><td>5.3%</td><td>5.8%</td></tr><tr><td>8</td><td>×</td><td>↗</td><td></td><td>√</td><td>10.6%</td><td>6.9%</td><td>11.7%</td><td>1.2%</td><td>1.0%</td></tr><tr><td>9</td><td>×</td><td>√</td><td></td><td>√</td><td>11.8%</td><td>10.6%</td><td>12.2%</td><td>0.7%</td><td>0.5%</td></tr><tr><td>10</td><td>×</td><td>×</td><td></td><td>√</td><td>22.7%</td><td>24.7%</td><td>21.6%</td><td>3.9%</td><td>2.1%</td></tr></table>

## G ADDITIONAL BASELINES

Due to the limitation of the main text, we also include three advanced baselines here: the general multiscale framework Scaleformer (Amin Shabani & Sylvain., 2023), two concurrent MLP-based model MTSMixer (Li et al., 2023) and TSMixer (Chen et al., 2023). Since the latter two baselines were not officially published during our submission, we adopted their public code and reproduced them with both unified hyperparameter setting and the hyperparameter searching settings. As presented in Table 23, 24, 25, TimeMixer still performs best in comparison with these baselines. Showcases of these additional baselines are also provided in Appendix I for an intuitive comparison. 

## H SPECTRAL ANALYSIS OF MODEL PREDICTIONS

To demonstrate the advancement of TimeMixer, we plot the spectrum of ground truth and model predictions. It is observed that TimeMixer captures different frequency parts precisely. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/75066d11d7d06fdbf8801469c53c9fd44bb712be1a8d33e6a1bdf455214b832f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/fd939d5555ec9c1ab39c9a31400c2bdbb06776fca53fd645340029163446ae7c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/c2128bcbeb0a0c2cec6551db717435fa713e8ae806502a94f14498459e54f3e6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/3497641599e1c0cb41c1d26d78234aff529b72e259b0fa6440338cdd7f116705.jpg)



(a) Ground Truth spectrogram


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/6f950eee43733f586b18b602db865f0cdf4821efab3fb4a26dda8f0e6e9211cd.jpg)



(b) TimeMixer spectrogram


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/81fd59eee4dc0152ca79d079fed6a7dfd259cba6928ff2783e9a97573ffe056a.jpg)



(c) PatchTST spectrogram



Figure 8: Prediction spectrogram cases from ETTh1 by ground truth and different models under the input-96-predict-96 settings.


## I SHOWCASES

In order to evaluate the performance of different models, we conduct the qualitative comparison by plotting the final dimension of forecasting results from the test set of each dataset (Figures 9, 10, 11, 12, 13, 17, 18). Among the various models, TimeMixer exhibits superior performance. 

## J LIMITATIONS AND FUTURE WORK

TimeMixer has shown favorable efficiency in GPU memory and running time as we presented in the main text. However, it should be noted that as the input length increases, the linear mixing layer may result in a larger number of model parameters, which is inefficient for mobile applications. To address this issue and improve TimeMixer’s parameter efficiency, we plan to investigate alternative mixing designs, such as attention-based or CNN-based in future research. In addition, we only focus on the temporal dimension mixing in this paper, and also plan to incorporate the variate dimension mixing into model design in our future work. Furthermore, as an application-oriented model, we made a great effort to verify the effectiveness of our design with experiments and ablations. The theoretical analysis to verify the optimality and completeness of our design is also a promising direction. 


Table 23: Unified and searched hyperparameter results for additional baselines in the long-term forecasting task. We compare extensive competitive models under different prediction lengths. Avg is averaged from all four prediction lengths, that is 96, 192, 336, 720. All these baselines are reproduced by their official code. For the searched hyperparameter setting, we follow the searching strategy described in Appendix E. Especially for TSMixer, we reproduced it in Pytorch (Paszke et al., 2019).


<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="8">Unified Hyperparameter</td><td colspan="8">Searched Hyperparameter</td></tr><tr><td colspan="2">TimeMixer (Ours)</td><td colspan="2">Scaleformer (2023)</td><td colspan="2">MTSMixer (2023)</td><td colspan="2">TSMixer (2023)</td><td colspan="2">TimeMixer (Ours)</td><td colspan="2">Scaleformer (2023)</td><td colspan="2">MTSMixer (2023)</td><td colspan="2">TSMixer (2023)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.163</td><td>0.209</td><td>0.220</td><td>0.289</td><td>0.173</td><td>0.224</td><td>0.175</td><td>0.247</td><td>0.147</td><td>0.197</td><td>0.192</td><td>0.241</td><td>0.167</td><td>0.221</td><td>0.149</td><td>0.198</td></tr><tr><td>192</td><td>0.208</td><td>0.250</td><td>0.341</td><td>0.385</td><td>0.219</td><td>0.261</td><td>0.224</td><td>0.294</td><td>0.189</td><td>0.239</td><td>0.220</td><td>0.288</td><td>0.208</td><td>0.250</td><td>0.201</td><td>0.251</td></tr><tr><td>336</td><td>0.251</td><td>0.287</td><td>0.463</td><td>0.455</td><td>0.274</td><td>0.300</td><td>0.262</td><td>0.326</td><td>0.241</td><td>0.280</td><td>0.288</td><td>0.324</td><td>0.298</td><td>0.302</td><td>0.287</td><td>0.291</td></tr><tr><td>720</td><td>0.339</td><td>0.341</td><td>0.640</td><td>0.565</td><td>0.365</td><td>0.359</td><td>0.349</td><td>0.348</td><td>0.310</td><td>0.330</td><td>0.321</td><td>0.360</td><td>0.344</td><td>0.339</td><td>0.320</td><td>0.336</td></tr><tr><td>Avg</td><td>0.240</td><td>0.271</td><td>0.416</td><td>0.423</td><td>0.258</td><td>0.286</td><td>0.253</td><td>0.304</td><td>0.222</td><td>0.262</td><td>0.248</td><td>0.304</td><td>0.254</td><td>0.278</td><td>0.240</td><td>0.269</td></tr><tr><td rowspan="5">Solar-Energy</td><td>96</td><td>0.189</td><td>0.259</td><td>0.271</td><td>0.331</td><td>0.217</td><td>0.272</td><td>0.216</td><td>0.294</td><td>0.167</td><td>0.220</td><td>0.224</td><td>0.328</td><td>0.199</td><td>0.251</td><td>0.190</td><td>0.272</td></tr><tr><td>192</td><td>0.222</td><td>0.283</td><td>0.288</td><td>0.332</td><td>0.258</td><td>0.299</td><td>0.294</td><td>0.359</td><td>0.187</td><td>0.249</td><td>0.247</td><td>0.312</td><td>0.221</td><td>0.275</td><td>0.207</td><td>0.278</td></tr><tr><td>336</td><td>0.231</td><td>0.292</td><td>0.358</td><td>0.412</td><td>0.278</td><td>0.310</td><td>0.302</td><td>0.367</td><td>0.200</td><td>0.258</td><td>0.274</td><td>0.308</td><td>0.231</td><td>0.281</td><td>0.246</td><td>0.304</td></tr><tr><td>720</td><td>0.223</td><td>0.285</td><td>0.377</td><td>0.437</td><td>0.293</td><td>0.321</td><td>0.311</td><td>0.372</td><td>0.215</td><td>0.250</td><td>0.335</td><td>0.384</td><td>0.270</td><td>0.301</td><td>0.274</td><td>0.308</td></tr><tr><td>Avg</td><td>0.216</td><td>0.280</td><td>0.323</td><td>0.378</td><td>0.261</td><td>0.300</td><td>0.280</td><td>0.348</td><td>0.192</td><td>0.244</td><td>0.270</td><td>0.333</td><td>0.230</td><td>0.277</td><td>0.229</td><td>0.283</td></tr><tr><td rowspan="5">Electricity</td><td>96</td><td>0.153</td><td>0.247</td><td>0.182</td><td>0.297</td><td>0.173</td><td>0.270</td><td>0.190</td><td>0.299</td><td>0.129</td><td>0.224</td><td>0.162</td><td>0.274</td><td>0.154</td><td>0.267</td><td>0.142</td><td>0.234</td></tr><tr><td>192</td><td>0.166</td><td>0.256</td><td>0.188</td><td>0.300</td><td>0.186</td><td>0.280</td><td>0.216</td><td>0.323</td><td>0.140</td><td>0.220</td><td>0.171</td><td>0.284</td><td>0.168</td><td>0.272</td><td>0.154</td><td>0.248</td></tr><tr><td>336</td><td>0.185</td><td>0.277</td><td>0.210</td><td>0.324</td><td>0.204</td><td>0.297</td><td>0.226</td><td>0.334</td><td>0.161</td><td>0.255</td><td>0.192</td><td>0.304</td><td>0.182</td><td>0.281</td><td>0.161</td><td>0.262</td></tr><tr><td>720</td><td>0.225</td><td>0.310</td><td>0.232</td><td>0.339</td><td>0.241</td><td>0.326</td><td>0.250</td><td>0.353</td><td>0.194</td><td>0.287</td><td>0.238</td><td>0.332</td><td>0.212</td><td>0.321</td><td>0.209</td><td>0.304</td></tr><tr><td>Avg</td><td>0.182</td><td>0.272</td><td>0.203</td><td>0.315</td><td>0.201</td><td>0.293</td><td>0.220</td><td>0.327</td><td>0.156</td><td>0.246</td><td>0.191</td><td>0.298</td><td>0.179</td><td>0.286</td><td>0.167</td><td>0.262</td></tr><tr><td rowspan="5">Traffic</td><td>96</td><td>0.462</td><td>0.285</td><td>0.564</td><td>0.351</td><td>0.523</td><td>0.357</td><td>0.499</td><td>0.344</td><td>0.360</td><td>0.249</td><td>0.409</td><td>0.281</td><td>0.514</td><td>0.338</td><td>0.398</td><td>0.272</td></tr><tr><td>192</td><td>0.473</td><td>0.296</td><td>0.570</td><td>0.349</td><td>0.535</td><td>0.367</td><td>0.540</td><td>0.370</td><td>0.375</td><td>0.250</td><td>0.418</td><td>0.294</td><td>0.519</td><td>0.351</td><td>0.402</td><td>0.281</td></tr><tr><td>336</td><td>0.498</td><td>0.296</td><td>0.576</td><td>0.349</td><td>0.566</td><td>0.379</td><td>0.557</td><td>0.378</td><td>0.385</td><td>0.270</td><td>0.427</td><td>0.294</td><td>0.557</td><td>0.361</td><td>0.412</td><td>0.294</td></tr><tr><td>720</td><td>0.506</td><td>0.313</td><td>0.602</td><td>0.360</td><td>0.608</td><td>0.397</td><td>0.586</td><td>0.397</td><td>0.430</td><td>0.281</td><td>0.518</td><td>0.356</td><td>0.569</td><td>0.362</td><td>0.448</td><td>0.311</td></tr><tr><td>Avg</td><td>0.484</td><td>0.297</td><td>0.578</td><td>0.352</td><td>0.558</td><td>0.375</td><td>0.546</td><td>0.372</td><td>0.387</td><td>0.262</td><td>0.443</td><td>0.307</td><td>0.539</td><td>0.354</td><td>0.415</td><td>0.290</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.375</td><td>0.400</td><td>0.401</td><td>0.428</td><td>0.418</td><td>0.437</td><td>0.387</td><td>0.411</td><td>0.361</td><td>0.390</td><td>0.381</td><td>0.412</td><td>0.397</td><td>0.428</td><td>0.370</td><td>0.402</td></tr><tr><td>192</td><td>0.429</td><td>0.421</td><td>0.471</td><td>0.478</td><td>0.463</td><td>0.460</td><td>0.441</td><td>0.437</td><td>0.409</td><td>0.414</td><td>0.445</td><td>0.441</td><td>0.452</td><td>0.466</td><td>0.406</td><td>0.414</td></tr><tr><td>336</td><td>0.484</td><td>0.458</td><td>0.527</td><td>0.498</td><td>0.516</td><td>0.478</td><td>0.507</td><td>0.467</td><td>0.430</td><td>0.429</td><td>0.501</td><td>0.484</td><td>0.487</td><td>0.462</td><td>0.424</td><td>0.434</td></tr><tr><td>720</td><td>0.498</td><td>0.482</td><td>0.578</td><td>0.547</td><td>0.532</td><td>0.549</td><td>0.527</td><td>0.548</td><td>0.445</td><td>0.460</td><td>0.544</td><td>0.528</td><td>0.510</td><td>0.506</td><td>0.471</td><td>0.479</td></tr><tr><td>Avg</td><td>0.447</td><td>0.440</td><td>0.495</td><td>0.488</td><td>0.482</td><td>0.481</td><td>0.466</td><td>0.467</td><td>0.411</td><td>0.423</td><td>0.468</td><td>0.466</td><td>0.461</td><td>0.464</td><td>0.418</td><td>0.432</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.289</td><td>0.341</td><td>0.368</td><td>0.398</td><td>0.343</td><td>0.378</td><td>0.308</td><td>0.357</td><td>0.271</td><td>0.330</td><td>0.340</td><td>0.394</td><td>0.328</td><td>0.367</td><td>0.271</td><td>0.339</td></tr><tr><td>192</td><td>0.372</td><td>0.392</td><td>0.431</td><td>0.446</td><td>0.422</td><td>0.425</td><td>0.395</td><td>0.404</td><td>0.317</td><td>0.402</td><td>0.401</td><td>0.414</td><td>0.404</td><td>0.426</td><td>0.344</td><td>0.397</td></tr><tr><td>336</td><td>0.386</td><td>0.414</td><td>0.486</td><td>0.474</td><td>0.462</td><td>0.460</td><td>0.428</td><td>0.434</td><td>0.332</td><td>0.396</td><td>0.437</td><td>0.448</td><td>0.406</td><td>0.434</td><td>0.360</td><td>0.400</td></tr><tr><td>720</td><td>0.412</td><td>0.434</td><td>0.517</td><td>0.522</td><td>0.476</td><td>0.475</td><td>0.443</td><td>0.451</td><td>0.342</td><td>0.408</td><td>0.469</td><td>0.471</td><td>0.448</td><td>0.463</td><td>0.428</td><td>0.461</td></tr><tr><td>Avg</td><td>0.364</td><td>0.395</td><td>0.451</td><td>0.460</td><td>0.426</td><td>0.435</td><td>0.394</td><td>0.412</td><td>0.316</td><td>0.384</td><td>0.412</td><td>0.432</td><td>0.397</td><td>0.422</td><td>0.350</td><td>0.399</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.320</td><td>0.357</td><td>0.383</td><td>0.408</td><td>0.344</td><td>0.378</td><td>0.331</td><td>0.378</td><td>0.291</td><td>0.340</td><td>0.338</td><td>0.375</td><td>0.316</td><td>0.362</td><td>0.288</td><td>0.336</td></tr><tr><td>192</td><td>0.361</td><td>0.381</td><td>0.417</td><td>0.421</td><td>0.397</td><td>0.408</td><td>0.386</td><td>0.399</td><td>0.327</td><td>0.365</td><td>0.392</td><td>0.406</td><td>0.374</td><td>0.391</td><td>0.332</td><td>0.374</td></tr><tr><td>336</td><td>0.390</td><td>0.404</td><td>0.437</td><td>0.448</td><td>0.429</td><td>0.430</td><td>0.426</td><td>0.421</td><td>0.360</td><td>0.381</td><td>0.410</td><td>0.426</td><td>0.408</td><td>0.411</td><td>0.358</td><td>0.381</td></tr><tr><td>720</td><td>0.454</td><td>0.441</td><td>0.512</td><td>0.481</td><td>0.489</td><td>0.460</td><td>0.489</td><td>0.465</td><td>0.415</td><td>0.417</td><td>0.481</td><td>0.476</td><td>0.472</td><td>0.454</td><td>0.420</td><td>0.417</td></tr><tr><td>Avg</td><td>0.381</td><td>0.395</td><td>0.438</td><td>0.440</td><td>0.415</td><td>0.419</td><td>0.408</td><td>0.416</td><td>0.348</td><td>0.375</td><td>0.406</td><td>0.421</td><td>0.393</td><td>0.405</td><td>0.350</td><td>0.377</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.175</td><td>0.258</td><td>0.201</td><td>0.297</td><td>0.191</td><td>0.278</td><td>0.179</td><td>0.282</td><td>0.164</td><td>0.254</td><td>0.192</td><td>0.274</td><td>0.187</td><td>0.268</td><td>0.160</td><td>0.249</td></tr><tr><td>192</td><td>0.237</td><td>0.299</td><td>0.261</td><td>0.324</td><td>0.258</td><td>0.320</td><td>0.244</td><td>0.305</td><td>0.223</td><td>0.295</td><td>0.248</td><td>0.322</td><td>0.237</td><td>0.301</td><td>0.228</td><td>0.299</td></tr><tr><td>336</td><td>0.298</td><td>0.340</td><td>0.328</td><td>0.366</td><td>0.319</td><td>0.357</td><td>0.320</td><td>0.357</td><td>0.279</td><td>0.330</td><td>0.301</td><td>0.348</td><td>0.299</td><td>0.352</td><td>0.269</td><td>0.328</td></tr><tr><td>720</td><td>0.391</td><td>0.396</td><td>0.424</td><td>0.417</td><td>0.417</td><td>0.412</td><td>0.419</td><td>0.432</td><td>0.359</td><td>0.383</td><td>0.411</td><td>0.398</td><td>0.413</td><td>0.419</td><td>0.421</td><td>0.426</td></tr><tr><td>Avg</td><td>0.275</td><td>0.323</td><td>0.303</td><td>0.351</td><td>0.296</td><td>0.342</td><td>0.290</td><td>0.344</td><td>0.256</td><td>0.315</td><td>0.288</td><td>0.336</td><td>0.284</td><td>0.335</td><td>0.270</td><td>0.326</td></tr></table>


Table 24: Short-term forecasting results in the PEMS datasets with multiple variates. All input lengths are 96 and prediction lengths are 12. A lower MAE, MAPE or RMSE indicates a better prediction.


<table><tr><td colspan="2">Models</td><td>TimeMixer (Ours)</td><td>Scaleformer (2023)</td><td>MTSMixer (2023)</td><td>TSMixer (2023)</td></tr><tr><td rowspan="3">PEMS03</td><td>MAE</td><td>14.63</td><td>17.66</td><td>18.63</td><td>15.71</td></tr><tr><td>MAPE</td><td>14.54</td><td>17.58</td><td>19.35</td><td>15.28</td></tr><tr><td>RMSE</td><td>23.28</td><td>27.51</td><td>28.85</td><td>25.88</td></tr><tr><td rowspan="3">PEMS04</td><td>MAE</td><td>19.21</td><td>22.68</td><td>25.57</td><td>20.86</td></tr><tr><td>MAPE</td><td>12.53</td><td>14.81</td><td>17.79</td><td>12.97</td></tr><tr><td>RMSE</td><td>30.92</td><td>35.61</td><td>39.92</td><td>32.68</td></tr><tr><td rowspan="3">PEMS07</td><td>MAE</td><td>20.57</td><td>27.62</td><td>25.69</td><td>22.97</td></tr><tr><td>MAPE</td><td>8.62</td><td>12.68</td><td>11.57</td><td>9.93</td></tr><tr><td>RMSE</td><td>33.59</td><td>42.27</td><td>39.82</td><td>35.68</td></tr><tr><td rowspan="3">PEMS08</td><td>MAE</td><td>15.22</td><td>20.74</td><td>24.22</td><td>18.79</td></tr><tr><td>MAPE</td><td>9.67</td><td>12.81</td><td>14.98</td><td>10.69</td></tr><tr><td>RMSE</td><td>24.26</td><td>32.77</td><td>37.21</td><td>26.74</td></tr></table>


Table 25: Short-term forecasting results in the M4 dataset with a single variate. All prediction lengths are in [6, 48]. A lower SMAPE, MASE or OWA indicates a better prediction.


<table><tr><td colspan="2">Models</td><td>TimeMixer (Ours)</td><td>Scaleformer (2023)</td><td>MTSMixer (2023)</td><td>TSMixer (2023)</td></tr><tr><td rowspan="3">Yearly</td><td>SMAPE</td><td>13.206</td><td>13.778</td><td>20.071</td><td>19.845</td></tr><tr><td>MASE</td><td>2.916</td><td>3.176</td><td>4.537</td><td>4.439</td></tr><tr><td>OWA</td><td>0.776</td><td>0.871</td><td>1.185</td><td>1.166</td></tr><tr><td rowspan="3">Quarterly</td><td>SMAPE</td><td>9.996</td><td>10.727</td><td>16.371</td><td>16.322</td></tr><tr><td>MASE</td><td>1.166</td><td>1.291</td><td>2.216</td><td>2.21</td></tr><tr><td>OWA</td><td>0.825</td><td>0.954</td><td>1.551</td><td>1.543</td></tr><tr><td rowspan="3">Monthly</td><td>SMAPE</td><td>12.605</td><td>13.378</td><td>18.947</td><td>19.248</td></tr><tr><td>MASE</td><td>0.919</td><td>1.104</td><td>1.725</td><td>1.774</td></tr><tr><td>OWA</td><td>0.869</td><td>0.972</td><td>1.468</td><td>1.501</td></tr><tr><td rowspan="3">Others</td><td>SMAPE</td><td>4.564</td><td>4.972</td><td>7.493</td><td>7.494</td></tr><tr><td>MASE</td><td>3.115</td><td>3.311</td><td>5.457</td><td>5.463</td></tr><tr><td>OWA</td><td>0.982</td><td>1.112</td><td>1.649</td><td>1.651</td></tr><tr><td rowspan="3">Weighted Average</td><td>SMAPE</td><td>11.723</td><td>12.978</td><td>18.041</td><td>18.095</td></tr><tr><td>MASE</td><td>1.559</td><td>1.764</td><td>2.677</td><td>2.674</td></tr><tr><td>OWA</td><td>0.840</td><td>0.921</td><td>1.364</td><td>1.336</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/037682877e2efb809b27f092252c3c20fe85ffad0b9c98a71f135aeba4d83f18.jpg)



(a) TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/2717f5dd8ac91b54bb2ae0702a02cbb733dd9c18042729c661655591b9fd7e91.jpg)



(b) PatchTST


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/7d1020d27c2352bc10384d84a0df7f6cc2fa53f45ff877908f5d5509116da761.jpg)



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/377fa5a382aed2b4b117dbf08c34cd940d5d7d6b18e7fa42b6c996db303e31db.jpg)



(d) DLinear


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/0e6c09672a9a71a84daf1876a656b5b0215a1b8ffb02f45ad2375b2204cafd5d.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/44b77da4aded8a6457b80677cb7a4cec8baf03efd4bb8244f18e8f34aa9e9098.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/9acc5d49f56949b995136f0b4b1e95234d06dd7c43af3c480997b57b24ca5476.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/fed5247f9ae5fbc69248ab0a8d2bea2cd267203a277449f5a53001ddd6d0946d.jpg)



(h) TSMixer



Figure 9: Prediction cases from ETTh1 by different models under the input-96-predict-96 settings. Blue lines are the ground truths and orange lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/3f816de3954410ac10825c1d176a116a7b4998c4226401bd9dcb04dc9c71f83d.jpg)



(a) TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d8cc33615c446587cf8f50344b3b7d4b319b1bcbd66e6c68a7e77b56242748d8.jpg)



(b) PatchTST


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ca59d861ab8ef29f480186b46b3fd33bda8b0c37c0dd9368915227526c876dbd.jpg)



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/a6d87b650b5735f3aceed01be9e95d6c015da07886ce778d7e9106701440e1a5.jpg)



(d) DLinear


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d7c69baa9c1551c5ca8b5e5ec139021cbb667256477a7f9cc40b37fbf5845db8.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/7d246d6e5a02dcc23433016fc124a93888e26bdc21d0bffa71c7909f44290e29.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/90b1b567fcce011c1bade09ff4fbdc2146682bedc9f53a913cc3d83408b27669.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ec6f73d699172f647d24dc9742d3690f507cf5c24987ac9ac26741a6387efad4.jpg)



(h) TSMixer



Figure 10: Prediction cases from Electricity by different models under input-96-predict-96 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/8748967e91006278cbf8608703034c83ce49bb6f34b472940bb2981f30cf1c8f.jpg)



(a) TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/9867959d416e984a64ed34f2a39a5ee191833cbb5f9e0aa8a95cc93d931862bf.jpg)



(b) PatchTST


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/b817919b1c47ff03a1dfa08d88d26785b97ceb1d61acbc71c4d94c31f9755b31.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/0d7ba02f94776ec7cb1831d40206c0526c6d8f8301b089504760d8155c7a32bd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/62b7ba1379d6495c962992784a306cdccd8f291d773673f9a077a293eb4296de.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/bc89ce8c30418753392b9ec2b4ea714abce3cf1d1d62c5a6037ce9fab57ec613.jpg)



(c) TimesNet



(f) Scaleformer



(d) DLinear


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/b509eea64d51b0c3475a0f6e78099b458826f8430628ae7d7dd936c56c333a41.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/6f27605d139ca0948991d2cff5e979f22f0045ef803d16414be4c34dd6a08260.jpg)



(h) TSMixer



Figure 11: Prediction cases from Traffic by different models under the input-96-predict-96 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d8cd7c59e75eb808d73330c050f98539e8514cc97094ac21997d3402f5588f68.jpg)



(a) TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ead8e001252d7bc4973febda3a508df12cba4329d12c066977c50a164529b742.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/6942cf48bd91662c25141f5d86b821ad04049a6516816a0fb7b89e04740c6654.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/36dc73aa38b481b3b9f173451883a283c697507c712b038ea42e69936b9c5453.jpg)



(b) PatchTST



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/0481ba363025abb87c33be8547a4486cf1f07eb06d8e911a6221ebfbb7364459.jpg)



(c) TimesNet



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/1a7df7081546d65b9ca32f26a3431901144d80a4c2598acc9af8c5ce7c102fea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f0c6477d7cf3549535dfa860420900c6cb5aa83f1445224e3bf14e0ad7abce85.jpg)



(g) MTSMixer



(d) DLinear


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/652bd9835166b79d93214fd141f498d6af3e384e8bdc98c343b65f42b6eb6711.jpg)



(h) TSMixer



Figure 12: Prediction cases from Weather by different models under the input-96-predict-96 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f7dfc8ff668207f4bba41a2a2fc2e9c2118421ada1e17848455179f3540bba6c.jpg)



(a) TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/30b0b13ebf885cef22517fd8077a33d1c44c1ae380dfcc529a4d526443bc73dc.jpg)



(b) PatchTST


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/45d50c71020548bcfdad53b30f0b91b11d035fcdad61e17a41d1db52e6462bea.jpg)



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/46bcaa2619763fd8ec8095afa95ec0b380643a802e9d27bbde321c32b027ec5d.jpg)



(d) DLinear


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f965a45df4f0f1ba51ba06d7a79ce3012283def9df1c99e9238eb06cc4a5ad14.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/475cfddb7256831b42bcc3f4716e7d363ef3f9ef50b1edcf894494620cf58d90.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/a593172b3a3e5ad69a0213ac8e581e54f7de3bc11fd49664d1345aceb309f297.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/75a1a41414238cc4ab7381dd686184a5f3a536d4738d73d6145add5227c85975.jpg)



(h) TSMixer



Figure 13: Showcases from Solar-Energy by different models under the input-96-predict-96 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/23188d4ed228843924f89028b311283284a2636c0287e428beff1e7bfdff64f0.jpg)



(a)TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/9cb9383590e9b1cae312f3d652111980b53576d564bc17e75a589ead99e7b41f.jpg)



(b)SCINet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/80f056f3357fd4006a056a706e18ac9313f709c2e144f14c5fd2a782af3cc4c7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ea02ca8219a38a74ea761c8e70b8b5a636fa41cb4f3ead0112d690abfdf81d06.jpg)



(d) Crossformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/074107b30ce097fdf2c03f503df1502ea757b42fc718c6a90cb5d7af74ab9ac3.jpg)



(e) Autoformer



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/99a59d5404ca2dff67e8af2dfa077a0a2f9831c13b7919f604d181d300815f42.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f7be1a3304480959c5102cc19ee0fe3b82dcc5c4fdca95fc23eb766955772eec.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/9bce553e8a199d62040dd3afbd3544b1fefe907f6f35affa8f2f86d928c1de40.jpg)



(h) TSMixer



Figure 14: Showcases from PEMS03 by different models under the input-96-predict-12 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/eacbc203f3aff4f83f720d13c46a1ef6607d3c9fe51bf9e41e6d9c77372fa8ad.jpg)



(a)TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/62f386bddf5af7b847e5d0f15c5fc81a09fac881e0c5cb6a57dff54408b888f5.jpg)



(b)SCINet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/3226aa09823c1026886464755ebdc9225168142f48beec7bad5b6a91a5bf596d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f3d59110228bc416db94296e556f86a60c8ec3c93193366ce63de52b195a1353.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/f3372f3004d4f592cf9dc988740bb342674e3f7e9c9f683934dc5d8041f99d07.jpg)



(e) Autoformer



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/65f3175f45380b5518977193106b3d2e69e858f6daf37711d347a12d9a1aef62.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/61c47253633642694c9437711028e86a26cf052ce8938e0a620b8c42b59e8ac2.jpg)



(g) MTSMixer



(d) Crossformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/7941eb4eed40accf3cc493a39d57474a80b1a798d36054cef5cbb2195dd66661.jpg)



(h) TSMixer



Figure 15: Showcases from PEMS04 by different models under the input-96-predict-12 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/2fc448787a0d25140024206a08cbb1c55d8ba669aa69e7c8a57d959273cea4de.jpg)



(a)TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/5b0ce76baf67d92920533aa39c34ad7c68bb6567c7abf1a02a52ec50de0ea28e.jpg)



(b)SCINet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/a6b7a739ebfa11dc75d7c52ca7535d971b0c876fe7a72b7dd8533c0cdf7ab8b4.jpg)



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/c64b55aedbc78915f37ea53aaa86d6c101d27410362d4ccecce94866bb30741a.jpg)



(d) Crossformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/a61caf2e2ffd230d924024624a9b6b2c4b2ad63d21b1c6ec2f06234823d18c75.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/e9c5adb12dbaec7c201933dcf49c29a4b681ecf23b5c08b69af8339ddcc09019.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/a082b932701c7f47ad7e5151fc17642f0361a2e5f66c5c4978a6f13c8bc83c0c.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/55b1f88e7b88ee82c1243e9c28160cf9e7c5132b35d1fa71135efcf69d24199d.jpg)



(h) TSMixer



Figure 16: Showcases from PEMS07 by different models under the input-96-predict-12 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d5f375b9c0d026b80aa1ec0d8515db72baf8f52bccd440e2d673461de9d866ae.jpg)



(a)TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/dfa74607f0ab43eaa8b0c3328bb70fcd948225529be27b40d179d036bce5a369.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/fb1c7554b440171ccb0b4bc42eb2599740a80d3307957a08e2c73de1a94ebd70.jpg)



(b)SCINet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ab0f07f0947365ae394c49cd851ce85fface817de058216c92c9809de53a6e45.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/fa0a23b55150a2d02f801f4d63a498fc6fb04b2db85aad412c945aa6e101ecf6.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/3ce74bea2df1583dcd6865b1aaa747cf4ed50f1ea1b38faee6006e8e50795742.jpg)



(c) TimesNet



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/968aaa68bf047b32459640c54768b52769983123504ad06997a2abfdb87abab0.jpg)



(d) Crossformer



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/ec9b99112ba58597afac2e7bdd6005dd5921f10e2f563757dd89a02d629b28f3.jpg)



(h) TSMixer



Figure 17: Showcases from PEMS08 by different models under the input-96-predict-12 settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/963ec9319c3e46c33aab231f882888c2b800ae02e597d72503bc86ea00801c90.jpg)



(a)TimeMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/8639c5635af2d9e9f948171f5e55569407be1f07604aaf10d854a85ff377da9d.jpg)



(b)SCINet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/337f944a539faca14a0d55ad4dc850b9c19b3d9dcd82416f127cf751275e7f0c.jpg)



(c) TimesNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/7a7c63d5c4846ee7d1dccd38366a642a4c7076b29f23a5c66c992bebdb54dda6.jpg)



(d) PatchTST


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/35570bdb6c2fdc00d45bcc0ee8329be1e54ce682484fbdd2c6dc092f4653162d.jpg)



(e) Autoformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/4ed06e3da150ec17ed14a2595e22d9ed125673345cf0679d603053aeed6a305d.jpg)



(f) Scaleformer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/6e0d3c4d2d645f099783d7ca62277e4aac693347b1c6ad7d201a91f47f967991.jpg)



(g) MTSMixer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/e4724628-ddb0-4e18-9293-335d971ec839/d4bc93b172c2832882e908f1b37f8e4f6ad88fa473317029e9cbd2e7e2a3dc6f.jpg)



(h) TSMixer



Figure 18: Showcases from the M4 dataset by different models under the input-36-predict-18 settings.
