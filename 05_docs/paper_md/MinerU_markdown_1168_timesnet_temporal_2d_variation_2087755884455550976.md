# TIMESNET: TEMPORAL 2D-VARIATION MODELING FOR GENERAL TIME SERIES ANALYSIS

Haixu Wu∗, Tengge Hu∗, Yong Liu∗, Hang Zhou, Jianmin Wang, Mingsheng Long<sup>B</sup> School of Software, BNRist, Tsinghua University, Beijing 100084, China {whx20,liuyong21,htg21,h-zhou18}@mails.tsinghua.edu.cn {jimwang,mingsheng}@tsinghua.edu.cn 

## ABSTRACT

Time series analysis is of immense importance in extensive applications, such as weather forecasting, anomaly detection, and action recognition. This paper focuses on temporal variation modeling, which is the common key problem of extensive analysis tasks. Previous methods attempt to accomplish this directly from the 1D time series, which is extremely challenging due to the intricate temporal patterns. Based on the observation of multi-periodicity in time series, we ravel out the complex temporal variations into the multiple intraperiod- and interperiod-variations. To tackle the limitations of 1D time series in representation capability, we extend the analysis of temporal variations into the 2D space by transforming the 1D time series into a set of 2D tensors based on multiple periods. This transformation can embed the intraperiod- and interperiod-variations into the columns and rows of the 2D tensors respectively, making the 2D-variations to be easily modeled by 2D kernels. Technically, we propose the TimesNet with TimesBlock as a task-general backbone for time series analysis. TimesBlock can discover the multi-periodicity adaptively and extract the complex temporal variations from transformed 2D tensors by a parameter-efficient inception block. Our proposed TimesNet achieves consistent state-of-the-art in five mainstream time series analysis tasks, including short- and long-term forecasting, imputation, classification, and anomaly detection. Code is available at this repository: https://github.com/thuml/TimesNet. 

## 1 INTRODUCTION

Time series analysis is widely used in extensive real-world applications, such as the forecasting of meteorological factors for weather prediction (Wu et al., 2021), imputation of missing data for data mining (Friedman, 1962), anomaly detection of monitoring data for industrial maintenance (Xu et al., 2021) and classification of trajectories for action recognition (Franceschi et al., 2019). Because of its immense practical value, time series analysis has received great interest (Lim & Zohren, 2021). 

Different from other types of sequential data, such as language or video, time series is recorded continuously and each time point only saves some scalars. Since one single time point usually cannot provide sufficient semantic information for analysis, many works focus on the temporal variation, which is more informative and can reflect the inherent properties of time series, such as the continuity, periodicity, trend and etc. However, the variations of real-world time series always involve intricate temporal patterns, where multiple variations (e.g. rising, falling, fluctuation and etc.) mix and overlap with each other, making the temporal variation modeling extremely challenging. 

Especially in the deep learning communities, benefiting from the powerful non-linear modeling capacity of deep models, many works have been proposed to capture the complex temporal variations in real-world time series. One category of methods adopts recurrent neural networks (RNN) to model the successive time points based on the Markov assumption (Hochreiter & Schmidhuber, 1997; Lai et al., 2018; Shen et al., 2020). However, these methods usually fail in capturing the longterm dependencies and their efficiency suffers from the sequential computation paradigm. Another category of methods utilizes the convolutional neural network along the temporal dimension (TCN) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/08bb69a4cdb53e717f0da47559a42af29b499c722147265ff47634f120c769b1.jpg)



Figure 1: Multi-periodicity and temporal 2D-variation of time series. Each period involves the intraperiod-variation and interperiod-variation. We transform the original 1D time series into a set of 2D tensors based on multiple periods, which can unify the intraperiod- and interperiod-variations.


to extract the variation information (Franceschi et al., 2019; He & Zhao, 2019). Also, because of the locality property of the one-dimension convolution kernels, they can only model the variations among adjacent time points, thereby still failing in long-term dependencies. Recently, Transformers with attention mechanism have been widely used in sequential modeling (Brown et al., 2020; Dosovitskiy et al., 2021; Liu et al., 2021b). In time series analysis, many Transformer-based models adopt the attention mechanism or its variants to capture the pair-wise temporal dependencies among time points (Li et al., 2019; Kitaev et al., 2020; Zhou et al., 2021; 2022). But it is hard for attention mechanism to find out reliable dependencies directly from scattered time points, since the temporal dependencies can be obscured deeply in intricate temporal patterns (Wu et al., 2021). 

In this paper, to tackle the intricate temporal variations, we analyze the time series from a new dimension of multi-periodicity. Firstly, we observe that real-world time series usually present multi-periodicity, such as daily and yearly variations for weather observations, weekly and quarterly variations for electricity consumption. These multiple periods overlap and interact with each other, making the variation modeling intractable. Secondly, for each period, we find out that the variation of each time point is not only affected by the temporal pattern of its adjacent area but also highly related to the variations of its adjacent periods. For clearness, we name these two types of temporal variations as intraperiod-variation and interperiod-variation respectively. The former indicates short-term temporal patterns within a period. The latter can reflect long-term trends of consecutive different periods. Note that for the time series without clear periodicity, the variations will be dominated by the intraperiod-variation and is equivalent to the ones with infinite period length. 

Since different periods will lead to different intraperiod- and interperiod-variations, the multiperiodicity can naturally derive a modular architecture for temporal variation modeling, where we can capture the variations derived by a certain period in one module. Besides, this design makes the intricate temporal patterns disentangled, benefiting the temporal variation modeling. However, it is notable that the 1D time series is hard to explicitly present two different types of variations simultaneously. To tackle this obstacle, we extend the analysis of temporal variations into the 2D space. Concretely, as shown in Figure 1, we can reshape the 1D time series into a 2D tensor, where each column contains the time points within a period and each row involves the time points at the same phase among different periods. Thus, by transforming 1D time series into a set of 2D tensors, we can break the bottleneck of representation capability in the original 1D space and successfully unify the intraperiod- and interperiod-variations in 2D space, obtaining the temporal 2D-variations. 

Technically, based on above motivations, we go beyond previous backbones and propose the TimesNet as a new task-general model for time series analysis. Empowering by TimesBlock, TimesNet can discover the multi-periodicity of time series and capture the corresponding temporal variations in a modular architecture. Concretely, TimesBlock can adaptively transform the 1D time series into a set of 2D tensors based on learned periods and further capture intraperiod- and interperiod-variations in the 2D space by a parameter-efficient inception block. Experimentally, TimesNet achieves the consistent state-of-the-art in five mainstream analysis tasks, including short- and long-term forecasting, imputation, classification and anomaly detection. Our contributions are summarized in three folds: 

• Motivated by multi-periodicity and complex interactions within and between periods, we find out a modular way for temporal variation modeling. By transforming the 1D time series into 2D space, we can present the intraperiod- and interperiod-variations simultaneously. 

• We propose the TimesNet with TimesBlock to discover multiple periods and capture temporal 2D-variations from the transformed 2D tensors by a parameter-efficient inception block. 

• As a task-general foundation model, TimesNet achieves the consistent state-of-the-art in five mainstream time series analysis tasks. Detailed and insightful visualizations are included. 

## 2 RELATED WORK

As a key problem of time series analysis, temporal variation modeling has been well explored. 

Many classical methods assume that the temporal variations follow the pre-defined patterns, such as ARIMA (Anderson & Kendall, 1976), Holt-Winter (Hyndman & Athanasopoulos, 2018) and Prophet (Taylor & Letham, 2018). However, the variations of real-world time series are usually too complex to be covered by pre-defined patterns, limiting the practical applicability of these classical methods. 

In recent years, many deep models have been proposed for temporal modeling, such as MLP, TCN, RNN-based models (Hochreiter & Schmidhuber, 1997; Lai et al., 2018; Franceschi et al., 2019). Technically, MLP-based methods (Oreshkin et al., 2019; Challu et al., 2022; Zeng et al., 2023; Zhang et al., 2022) adopt the MLP along the temporal dimension and encode the temporal dependencies into the fixed parameter of MLP layers. The TCN-based (2019) methods capture the temporal variations by convolutional kernels that slide along the temporal dimension. The RNN-based methods (Hochreiter & Schmidhuber, 1997; Lai et al., 2018; Gu et al., 2022) utilize the recurrent structure and capture temporal variations implicitly by state transitions among time steps. Note that none of these methods consider the temporal 2D-variations derived by periodicity, which is proposed in this paper. 

Besides, Transformers have shown great performance in time series forecasting (Zhou et al., 2021; Liu et al., 2021a; Wu et al., 2021; Zhou et al., 2022). With attention mechanism, they can discover the temporal dependencies among time points. Especially, Wu et al. present the Autoformer with Auto-Correlation mechanism to capture the series-wise temporal dependencies based on the learned periods. In addition, to tackle the intricate temporal patterns, Autoformer also presents a deep decomposition architecture to obtain the seasonal and trend parts of input series. Afterward, FEDformer (Zhou et al., 2022) employs the mixture-of-expert design to enhance the seasonal-trend decomposition and presents a sparse attention within the frequency domain. Unlike previous methods, we ravel out the intricate temporal patterns by exploring the multi-periodicity of time series and capture the temporal 2D-variations in 2D space by well-acknowledged computer vision backbones for the first time. 

It is also notable that, different from previous methods, we no longer limit to a specific analysis task and attempt to propose a task-general foundation model for time series analysis. 

## 3 TIMESNET

As aforementioned, based on the multi-periodicity of time series, we propose the TimesNet with a modular architecture to capture the temporal patterns derived from different periods. For each period, to capture the corresponding intraperiod- and interperiod-variations, we design a TimesBlock within the TimesNet, which can transform the 1D time series into 2D space and simultaneously model the two types of variations by a parameter-efficient inception block. 

## 3.1 TRANSFORM 1D-VARIATIONS INTO 2D-VARIATIONS

As shown in Figure 1, each time point involves two types of temporal variations with its adjacent area and with the same phase among different periods simultaneously, namely intraperiod- and interperiod variations. However, this original 1D structure of time series can only present the variations among adjacent time points. To tackle this limitation, we explore the two-dimension structure for temporal variations, which can explicitly present variations within and between periods, thereby with more advantages in representation capability and benefiting the subsequent representation learning. 

Concretely, for the length-T time series with C recorded variates, the original 1D organization is $\mathbf { X } _ { \mathrm { 1 D } } \in \mathbb { R } ^ { \tilde { T } \times C }$ . To represent the interperiod-variation, we need to discover periods first. Technically, we analyze the time series in the frequency domain by Fast Fourier Transform (FFT) as follows: 

$$
\mathbf {A} = \operatorname{Avg} \left(\operatorname{Amp} \left(\operatorname{FFT} (\mathbf {X} _ {\mathrm{1D}})\right)\right), \left\{f _ {1}, \dots , f _ {k} \right\} = \underset {f _ {*} \in \{1, \dots , [ \frac {T}{2} ] \}} {\arg \operatorname{Topk}} (\mathbf {A}), p _ {i} = \left\lceil \frac {T}{f _ {i}} \right\rceil , i \in \{1, \dots , k \}.\tag{1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/26638d3cf1feface5a7dd6e1de402d7184c99b70e4f1afc503d196397721a3db.jpg)



Figure 2: A univariate example to illustrate 2D structure in time series. By discovering the periodicity, we can transform the original 1D time series into structured 2D tensors, which can be processed by 2D kernels conveniently. By conducting the same reshape operation to all variates of time series, we can extend the above process to multivariate time series.


Here, $\mathrm { F F T } ( \cdot )$ and $\operatorname { A m p } ( \cdot )$ denote the FFT and the calculation of amplitude values. $\mathbf { A } \in \mathbb { R } ^ { T }$ represents the calculated amplitude of each frequency, which is averaged from $C$ dimensions by $\mathrm { A v g ( \cdot ) }$ . Note that the $j \cdot$ -th value $\mathbf { A } _ { j }$ represents the intensity of the frequency-j periodic basis function, corresponding to the period length $\textstyle \lceil { \frac { T } { j } } \rceil$ . Considering the sparsity of frequency domain and to avoid the noises brought by meaningless high frequencies (Chatfield, 1981; Zhou et al., 2022), we only select the top-k amplitude values and obtain the most significant frequencies $\{ f _ { 1 } , \cdots , f _ { k } \}$ with the unnormalized amplitudes $\{ \mathbf { A } _ { f _ { 1 } } , \dotsc , \mathbf { A } _ { f _ { k } } \}$ , where k is the hyper-parameter. These selected frequencies also correspond to k period lengths $\{ p _ { 1 } , \cdots , p _ { k } \}$ . Due to the conjugacy of frequency domain, we only consider the frequencies within $\{ 1 , \cdots , [ \frac { T } { 2 } ] \}$ . We summarize Equation 1 as follows: 

$$
\mathbf {A}, \left\{f _ {1}, \dots , f _ {k} \right\}, \left\{p _ {1}, \dots , p _ {k} \right\} = \operatorname{Period} \left(\mathbf {X} _ {\mathrm{1D}}\right).\tag{2}
$$

Based on the selected frequencies $\{ f _ { 1 } , \cdots , f _ { k } \}$ and corresponding period lengths $\{ p _ { 1 } , \cdots , p _ { k } \}$ , we can reshape the 1D time series $\mathbf { X } _ { \mathrm { 1 D } } \mathbf { \Psi } \in \mathbb { R } ^ { T \times C }$ into multiple 2D tensors by the following equations: 

$$
\mathbf {X} _ {\mathrm{2D}} ^ {i} = \operatorname{Reshape} _ {p _ {i}, f _ {i}} \left(\operatorname{Padding} \left(\mathbf {X} _ {\mathrm{1D}}\right)\right), i \in \{1, \dots , k \},\tag{3}
$$

where Padding( ) is to extend the time series by zeros along temporal dimension to make it compati ble for Resha $\mathrm { { o e } } _ { p _ { i } , f _ { i } } ( \cdot )$ , where $p _ { i }$ and $f _ { i }$ represent the number of rows and columns of the transformed 2D tensors respectively. Note that $\mathbf { X } _ { 2 \mathrm { D } } ^ { i } \in \mathbb { R } ^ { p _ { i } \times f _ { i } \times C }$ denotes the i-th reshaped time series based on frequency- $\cdot f _ { i } ,$ , whose columns and rows represent the intraperiod-variation and interperiod-variation under the corresponding period length $p _ { i }$ respectively. Eventually, as shown in Figure 2, based on the selected frequencies and estimated periods, we obtain a set of 2D tensors $\{ \mathbf { X } _ { \mathrm { 2 D } } ^ { 1 } , \cdot \cdot \cdot , \mathbf { X } _ { \mathrm { 2 D } } ^ { k } \}$ , which indicates k different temporal 2D-variations derived by different periods. 

It is also notable that, this transformation brings two types of localities to the transformed 2D tensors, that is localities among adjacent time points (columns, intraperiod-variation) and adjacent periods (rows, interperiod-variation). Thus, the temporal 2D-variations can be easily processed by 2D kernels. 

## 3.2 TIMESBLOCK

As shown in Figure 3, we organize the TimesBlock in a residual way (He et al., 2016). Concretely, for the length-T 1D input time series $\mathbf { X } _ { \mathrm { 1 D } } \in \mathbb { R } ^ { T \times C }$ , we project the raw inputs into the deep features $\mathbf { X } _ { \mathrm { 1 D } } ^ { 0 } \in \mathbb { R } ^ { \breve { T } \times d _ { \mathrm { m o d e l } } }$ by the embedding layer $\mathbf { X } _ { \mathrm { 1 D } } ^ { 0 } = \mathrm { E m b e d } ( \mathbf { \bar { X } } _ { \mathrm { 1 D } } )$ at the very beginning. For the l-th layer of TimesNet, the input is $\mathbf { X } _ { \mathrm { 1 D } } ^ { l - 1 } \in \mathbb { R } ^ { T \times d _ { \mathrm { \mathrm { \mathrm { n } } } } }$ model and the process can be formalized as: 

$$
\mathbf {X} _ {\mathrm{1D}} ^ {l} = \text { TimesBlock } \left(\mathbf {X} _ {\mathrm{1D}} ^ {l - 1}\right) + \mathbf {X} _ {\mathrm{1D}} ^ {l - 1}.\tag{4}
$$

As shown in Figure 3, for the l-th TimesBlock, the whole process involves two successive parts: capturing temporal 2D-variations and adaptively aggregating representations from different periods. 

Capturing temporal 2D-variations Similar to Equation 1, we can estimate period lengths for deep features $\bar { \mathbf X } _ { 1 \mathrm { D } } ^ { l - 1 }$ by Period( ). Based on estimated period lengths, we can transform the 1D time series processed 2D feature into <sub>X</sub> l,i back to 1D tensor <sub>X</sub> l,i <sub>R</sub>T ⇥dmodel for aggregation by <sub>Trunc( )</sub> tosign of TimesBloc<sup>(Liu</sup> <sup>et</sup> <sup>al.,</sup> <sup>2022b)</sup> into 2D space and obtain a set of 2D tensors, from which we can obtain informative representations b btruncate the padding series into original length. Note that benefiting from the transformanalby parameter-efficient inception block conveniently. The process is formalized as follows: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/3849119b6bdd3584e23dc77fd8c349afab3a27a6ea13f2cc619bba3aedd2d612.jpg)



X 2D = Inception ⇣X2D⌘ , i 2 {1, · · · , k} sign of TimesBlocuch as the widely-Figure 3: Overall architecture of TimesNet. TimesNet is stacked by TimesBlocks in a residual way. X <sup>l,i</sup> = Trunc ⇣Reshape ⇣X <sup>l,i</sup>⌘⌘ , i  1,  , k<sup>simultaneously,</sup> <sup>ma</sup>    design also bridgesTimesBlocks can capture various temporal 2D-variations from k different reshaped tensors by a l,i <sub>(fi pi) dmodel</sub> Generality in 2D vanalysis take advantparameter-efficient inception block in 2D space and fuse them based on normalized amplitude values.


$$
\begin{array}{r l} \mathbf {A} ^ {l - 1}, \{f _ {1}, \dots , f _ {k} \}, \{p _ {1}, \dots , p _ {k} \} & = \text {Period} (\mathbf {X} _ {\mathrm{1D}} ^ {l - 1}), \\ & \mathbf {X} _ {\mathrm{2D}} ^ {l, i} = \text {Reshape} _ {p _ {i}, f _ {i}} (\text {Padding} (\mathbf {X} _ {\mathrm{1D}} ^ {l - 1}))  ,   i \in \{1, \dots , k \} \\ & \widehat {\mathbf {X}} _ {\mathrm{2D}} ^ {l, i} = \text {Inception} (\mathbf {X} _ {\mathrm{2D}} ^ {l, i})  ,   i \in \{1, \dots , k \} \\ & \widehat {\mathbf {X}} _ {\mathrm{1D}} ^ {l, i} = \text {Trunc} (\text {Reshape} _ {1, (p _ {i} \times f _ {i})} (\widehat {\mathbf {X}} _ {\mathrm{2D}} ^ {l, i}))  ,   i \in \{1, \dots , k \}, \end{array}\tag{5}
$$

where $\mathbf { X } _ { 2 \mathrm { D } } ^ { l , i } \in \mathbb { R } ^ { p _ { i } \times f _ { i } \times d _ { \mathrm { m o d e l } } }$ b b <sub>(6)</sub>is the i-th transformed 2D tensor. After the transformation, we process X<sup>l</sup>1D = XA <sup>l</sup> <sup>1</sup>fi ⇥ X 1D.the 2D tensor by a parameter-efficient inception block (Szegedy et al., 2015) as Inception( ), which <sub>Since the temporal variations are involved in the multiple highly-structured 2D tensors, the de-</sub>involves multi-scale 2D kernels and is one of the most well-acknowledged vision backbones. Then <sup>sign</sup> <sup>of</sup> <sup>Tim</sup>we transform the learned 2D representations $\widehat { \mathbf { X } } _ { 2 \mathrm { { D } } } ^ { l , i }$ <sup>can</sup> <sup>fully</sup> <sup>capture</sup> <sup>the</sup> <sup>multi-</sup>back to 1D space $\widehat { \mathbf { X } } _ { 1 \mathrm { D } } ^ { l , i } \in \mathbb { R } ^ { T \times d _ { \mathrm { m o d e l } } }$ <sup>s</sup> <sup>from</sup> <sup>multiple</sup> <sup>views</sup>for aggregation, bwhere we employ Trunc( ) to truncate the padded series with length $( p _ { i } \times f _ { i } )$ into original length T. 

<sup>also</sup> <sup>choose</sup> <sup>various</sup> <sup>vision</sup> <sup>backbones</sup> <sup>to</sup> <sup>replace</sup> <sup>the</sup> <sup>inception</sup> <sup>module</sup> <sup>for</sup> <sup>representation</sup> <sup>learning,</sup>Note that benefiting from the transformation of 1D time series, the 2D kernels in the inception (Liu et al., 2022b) and attention-based models (Liu et al., 2021b). Thus, our temporal 2D-variationblock can aggregate the multi-scale intraperiod-variation (columns) and interperiod-variation (rows) analysis take advantage of the development of vision community. For efficiency, we conduct the mainsimultaneously, covering both adjacent time points and adjacent periods. Besides, we adopt a shared inception block for different reshaped 2D tensors $\{ \mathbf { X } _ { \mathrm { 2 D } } ^ { l , 1 } , \cdot \cdot \cdot , \bar { \mathbf { X } _ { \mathrm { 2 D } } ^ { l , k } } \}$ to improve parameter efficiency, <sub>5</sub>which can make the model size invariant to the selection of hyper-parameter k. 

Adaptive aggregation Finally, we need to fuse k different 1D-representations $\{ \widehat { \mathbf { X } } _ { \mathrm { 1 D } } ^ { l , 1 } , \cdot \cdot \cdot , \widehat { \mathbf { X } } _ { \mathrm { 1 D } } ^ { l , k } \}$ b bfor the next layer. Inspired by Auto-Correlation (Wu et al., 2021), the amplitudes A can reflect the relative importance of selected frequencies and periods, thereby corresponding to the importance of each transformed 2D tensor. Thus, we aggregate the 1D-representations based on the amplitudes: 

$$
\begin{array}{c} \widehat {\mathbf {A}} _ {f _ {1}} ^ {l - 1}, \dots , \widehat {\mathbf {A}} _ {f _ {k}} ^ {l - 1} = \text {Softmax} \left(\mathbf {A} _ {f _ {1}} ^ {l - 1}, \dots , \mathbf {A} _ {f _ {k}} ^ {l - 1}\right) \\ \mathbf {X} _ {\mathrm{1D}} ^ {l} = \sum_ {i = 1} ^ {k} \widehat {\mathbf {A}} _ {f _ {i}} ^ {l - 1} \times \widehat {\mathbf {X}} _ {\mathrm{1D}} ^ {l, i}. \end{array}\tag{6}
$$

Since the variations within and between periods are already involved in multiple highly-structured 2D tensors, TimesBlock can fully capture multi-scale temporal 2D-variations simultaneously. Thus, TimesNet can achieve a more effective representation learning than directly from 1D time series. 

Generality in 2D vision backbones Benefiting from the transformation of 1D time series into temporal 2D-variations, we can choose various computer vision backbones to replace the inception block for representation learning, such as the widely-used ResNet (He et al., 2016) and ResNeXt (Xie et al., 2017), advanced ConvNeXt (Liu et al., 2022b) and attention-based models (Liu et al., 2021b). Thus, our temporal 2D-variation design also bridges the 1D time series to the booming 2D vision backbones, making the time series analysis take advantage of the development of computer vision community. In general, more powerful 2D backbones for representation learning will bring better performance. Considering both performance and efficiency (Figure 4 right), we conduct the main experiments based on the parameter-efficient inception block as shown in Equation 5. 

## 4 EXPERIMENTS

To verify the generality of TimesNet, we extensively experiment on five mainstream analysis tasks, including short- and long-term forecasting, imputation, classification and anomaly detection. 

Implementation Table 1 is a summary of benchmarks. More details about the dataset, experiment implementation and model configuration can be found in Appendix A. 


Table 1: Summary of experiment benchmarks.


<table><tr><td>Tasks</td><td>Benchmarks</td><td>Metrics</td><td>Series Length</td></tr><tr><td rowspan="2">Forecasting</td><td>Long-term: ETT (4 subsets), Electricity, Traffic, Weather, Exchange, ILI</td><td>MSE, MAE</td><td>96~720(ILI: 24~60)</td></tr><tr><td>Short-term: M4 (6 subsets)</td><td>SMAPE, MASE, OWA</td><td>6~48</td></tr><tr><td>Imputation</td><td>ETT (4 subsets), Electricity, Weather</td><td>MSE, MAE</td><td>96</td></tr><tr><td>Classification</td><td>UEA (10 subsets)</td><td>Accuracy</td><td>29~1751</td></tr><tr><td>Anomaly Detection</td><td>SMD, MSL, SMAP, SWaT, PSM</td><td>Precision, Recall, F1-Socre</td><td>100</td></tr></table>

Baselines Since we attempt to propose a foundation model for time series analysis, we extensively compare the well-acknowledged and advanced models in all five tasks, including the RNN-based models: LSTM (1997), LSTNet (2018) and LSSL (2022); CNN-based Model: TCN (2019); MLP-based models: LightTS (2022) and DLinear (2023); Transformer-based models: Reformer (2020), Informer (2021), Pyraformer (2021a), Autoformer (2021), FEDformer (2022), Non-stationary Transformer (2022a) and ETSformer (2022). Besides, we also compare the state-of-the-art models for each specific task, such as N-HiTS (2022) and N-BEATS (2019) for short-term forecasting, Anomaly Transformer (2021) for anomaly detection, Rocket (2020) and Flowformer (2022) for classification and etc. Overall, more than 15 baselines are included for a comprehensive comparison. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/92de3379af53c6b5c2adaecca38c39c7e146207e8ecaff387f597d0d4b7d491b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/c3cec3a50b87fc7dd8fe84421c61c14ad2fd25bc1103bfbe28f39945bc85f5d5.jpg)



Figure 4: Model performance comparison (left) and generality in different vision backbones (right).


## 4.1 MAIN RESULTS

As a foundation model, TimesNet achieves consistent state-of-the-art performance on five mainstream analysis tasks compared with other customized models (Figure 4 left). The full efficiency comparison is provided in Table 11 of Appendix. Besides, by replacing the inception block with more powerful vision backbones, we can further promote the performance of TimesNet (Figure 4 right), confirming that our design can make time series analysis take advantage of booming vision backbones. 

## 4.2 SHORT- AND LONG-TERM FORECASTING

Setups Time series forecasting is essential in weather forecasting, traffic and energy consumption planning. To fully evaluate the model performance in forecasting, we adopt two types of benchmarks, including long-term and short-term forecasting. Especially for the long-term setting, we follow the benchmarks used in Autoformer (2021), including ETT (Zhou et al., 2021), Electricity (UCI), Traffic (PeMS), Weather (Wetterstation), Exchange (Lai et al., 2018) and ILI (CDC), covering five real-world applications. For the short-term dataset, we adopt the M4 (Spyros Makridakis, 2018), which contains the yearly, quarterly and monthly collected univariate marketing data. Note that each dataset in the long-term setting only contains one continuous time series, where we obtain samples by sliding window, while M4 involves 100,000 different time series collected in different frequencies. 


Table 2: Long-term forecasting task. The past sequence length is set as 36 for ILI and 96 for the others. All the results are averaged from 4 different prediction lengths, that is 24, 36, 48, 60 for ILI and 96, 192, 336, 720 for the others. See Table 13 in Appendix for the full results.


<table><tr><td>Models</td><td colspan="2">TimesNet(Ours)</td><td colspan="2">ETSformer(2022)</td><td colspan="2">LightTS(2022)</td><td colspan="2">DLinear(2023)</td><td colspan="2">FEDformer(2022)</td><td colspan="2">Stationary(2022a)</td><td colspan="2">Autoformer(2021)</td><td colspan="2">Pyraformer(2021a)</td><td colspan="2">Informer(2021)</td><td colspan="2">LogTrans(2019)</td><td colspan="2">Reformer(2020)</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.400</td><td>0.406</td><td>0.429</td><td>0.425</td><td>0.435</td><td>0.437</td><td>0.403</td><td>0.407</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td><td>0.691</td><td>0.607</td><td>0.961</td><td>0.734</td><td>0.929</td><td>0.725</td><td>0.799</td><td>0.671</td></tr><tr><td>ETTm2</td><td>0.291</td><td>0.333</td><td>0.293</td><td>0.342</td><td>0.409</td><td>0.436</td><td>0.350</td><td>0.401</td><td>0.305</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.327</td><td>0.371</td><td>1.498</td><td>0.869</td><td>1.410</td><td>0.810</td><td>1.535</td><td>0.900</td><td>1.479</td><td>0.915</td></tr><tr><td>ETTh1</td><td>0.458</td><td>0.450</td><td>0.542</td><td>0.510</td><td>0.491</td><td>0.479</td><td>0.456</td><td>0.452</td><td>0.440</td><td>0.460</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td><td>0.827</td><td>0.703</td><td>1.040</td><td>0.795</td><td>1.072</td><td>0.837</td><td>1.029</td><td>0.805</td></tr><tr><td>ETTh2</td><td>0.414</td><td>0.427</td><td>0.439</td><td>0.452</td><td>0.602</td><td>0.543</td><td>0.559</td><td>0.515</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td>0.459</td><td>0.826</td><td>0.703</td><td>4.431</td><td>1.729</td><td>2.686</td><td>1.494</td><td>6.736</td><td>2.191</td></tr><tr><td>Electricity</td><td>0.192</td><td>0.295</td><td>0.208</td><td>0.323</td><td>0.229</td><td>0.329</td><td>0.212</td><td>0.300</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td><td>0.379</td><td>0.445</td><td>0.311</td><td>0.397</td><td>0.272</td><td>0.370</td><td>0.338</td><td>0.422</td></tr><tr><td>Traffic</td><td>0.620</td><td>0.336</td><td>0.621</td><td>0.396</td><td>0.622</td><td>0.392</td><td>0.625</td><td>0.383</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td><td>0.878</td><td>0.469</td><td>0.764</td><td>0.416</td><td>0.705</td><td>0.395</td><td>0.741</td><td>0.422</td></tr><tr><td>Weather</td><td>0.259</td><td>0.287</td><td>0.271</td><td>0.334</td><td>0.261</td><td>0.312</td><td>0.265</td><td>0.317</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td><td>0.946</td><td>0.717</td><td>0.634</td><td>0.548</td><td>0.696</td><td>0.602</td><td>0.803</td><td>0.656</td></tr><tr><td>Exchange</td><td>0.416</td><td>0.443</td><td>0.410</td><td>0.427</td><td>0.385</td><td>0.447</td><td>0.354</td><td>0.414</td><td>0.519</td><td>0.500</td><td>0.461</td><td>0.454</td><td>0.613</td><td>0.539</td><td>1.913</td><td>1.159</td><td>1.550</td><td>0.998</td><td>1.402</td><td>0.968</td><td>1.280</td><td>0.932</td></tr><tr><td>ILI</td><td>2.139</td><td>0.931</td><td>2.497</td><td>1.004</td><td>7.382</td><td>2.003</td><td>2.616</td><td>1.090</td><td>2.847</td><td>1.144</td><td>2.077</td><td>0.914</td><td>3.006</td><td>1.161</td><td>7.635</td><td>2.050</td><td>5.137</td><td>1.544</td><td>4.839</td><td>1.485</td><td>4.724</td><td>1.445</td></tr></table>


Table 3: Short-term forecasting task on M4. The prediction lengths are in [6, 48] and results are weighted averaged from several datasets under different sample intervals. See Table 14 for full results.


<table><tr><td>Models</td><td>TimesNet (Ours)</td><td>N-HiTS (2022)</td><td>N-BEATS (2019)</td><td>ETSformer (2022)</td><td>LightTS (2022)</td><td>DLinear (2023)</td><td>FEDformer (2022)</td><td>Stationary (2022a)</td><td>Autoformer (2021)</td><td>Pyraformer (2021a)</td><td>Informer (2021)</td><td>LogTrans (2019)</td><td>Reformer (2020)</td></tr><tr><td>SMAPE</td><td>11.829</td><td>11.927</td><td>11.851</td><td>14.718</td><td>13.525</td><td>13.639</td><td>12.840</td><td>12.780</td><td>12.909</td><td>16.987</td><td>14.086</td><td>16.018</td><td>18.200</td></tr><tr><td>MASE</td><td>1.585</td><td>1.613</td><td>1.599</td><td>2.408</td><td>2.111</td><td>2.095</td><td>1.701</td><td>1.756</td><td>1.771</td><td>3.265</td><td>2.718</td><td>3.010</td><td>4.223</td></tr><tr><td>OWA</td><td>0.851</td><td>0.861</td><td>0.855</td><td>1.172</td><td>1.051</td><td>1.051</td><td>0.918</td><td>0.930</td><td>0.939</td><td>1.480</td><td>1.230</td><td>1.378</td><td>1.775</td></tr></table>


Results TimesNet shows great performance in both long-term and short-term settings (Table 2–3). Concretely, TimesNet achieves state-of-the-art in more than 80% of cases in long-term forecasting (Table 13). For the M4 dataset, since the time series are collected from different sources, the temporal variations can be quite diverse, making forecasting much more challenging. Our model still performs best in this task, surpassing extensive advanced MLP-based and Transformer-based models.


## 4.3 IMPUTATION

Setups Real-world systems always work continuously and are monitored by automatic observation equipment. However, due to malfunctions, the collected time series can be partially missing, making the downstream analysis difficult. Thus, imputation is widely-used in practical applications. In this paper, we select the datasets from the electricity and weather scenarios as our benchmarks, including ETT (Zhou et al., 2021), Electricity (UCI) and Weather (Wetterstation), where the data-missing problem happens commonly. To compare the model capacity under different proportions of missing data, we randomly mask the time points in the ratio of 12.5%, 25%, 37.5%, 50% . 

Results Due to the missing time points, the imputation task requires the model to discover underlying temporal patterns from the irregular and partially observed time series. As shown in Table 4, our proposed TimesNet still achieves the consistent state-of-the-art in this difficult task, verifying the model capacity in capturing temporal variation from extremely complicated time series. 

## 4.4 CLASSIFICATION

Setups Time series classification can be used in recognition and medical diagnosis (Moody et al., 2011). We adopt the sequence-level classification to verify the model capacity in high-level representation learning. Concretely, we select 10 multivariate datasets from UEA Time Series Classification Archive (Bagnall et al., 2018), covering the gesture, action and audio recognition, medical diagnosis by heartbeat monitoring and other practical tasks. Then, we pre-process the datasets following the descriptions in (Zerveas et al., 2021), where different subsets have different sequence lengths. 


Table 4: Imputation task. We randomly mask 12.5%, 25%, 37.5%, 50% time points in length-96 time series. The results are averaged from 4 different mask ratios. See Table 16 for full results.


<table><tr><td>Models</td><td>TimesNet(Ours)</td><td>ETSformer(2022)</td><td>LightTS(2022)</td><td>DLinear(2023)</td><td>FEDformer(2022)</td><td>Stationary(2022a)</td><td>Autoformer(2021)</td><td>Pyraformer(2021a)</td><td>Informer(2021)</td><td>LogTrans(2019)</td><td>Reformer(2020)</td></tr><tr><td>Metric</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>ETTm1</td><td>0.027 0.107</td><td>0.120 0.253</td><td>0.104 0.218</td><td>0.093 0.206</td><td>0.062 0.177</td><td>0.036 0.126</td><td>0.051 0.150</td><td>0.717 0.570</td><td>0.071 0.188</td><td>0.050 0.154</td><td>0.055 0.166</td></tr><tr><td>ETTm2</td><td>0.022 0.088</td><td>0.208 0.327</td><td>0.046 0.151</td><td>0.096 0.208</td><td>0.101 0.215</td><td>0.026 0.099</td><td>0.029 0.105</td><td>0.465 0.508</td><td>0.156 0.292</td><td>0.119 0.246</td><td>0.157 0.280</td></tr><tr><td>ETTh1</td><td>0.078 0.187</td><td>0.202 0.329</td><td>0.284 0.373</td><td>0.201 0.306</td><td>0.117 0.246</td><td>0.094 0.201</td><td>0.103 0.214</td><td>0.842 0.682</td><td>0.161 0.279</td><td>0.219 0.332</td><td>0.122 0.245</td></tr><tr><td>ETTh2</td><td>0.049 0.146</td><td>0.367 0.436</td><td>0.119 0.250</td><td>0.142 0.259</td><td>0.163 0.279</td><td>0.053 0.152</td><td>0.055 0.156</td><td>1.079 0.792</td><td>0.337 0.452</td><td>0.186 0.318</td><td>0.234 0.352</td></tr><tr><td>Electricity</td><td>0.092 0.210</td><td>0.214 0.339</td><td>0.131 0.262</td><td>0.132 0.260</td><td>0.130 0.259</td><td>0.100 0.218</td><td>0.101 0.225</td><td>0.297 0.382</td><td>0.222 0.328</td><td>0.175 0.303</td><td>0.200 0.313</td></tr><tr><td>Weather</td><td>0.030 0.054</td><td>0.076 0.171</td><td>0.055 0.117</td><td>0.052 0.110</td><td>0.099 0.203</td><td>0.032 0.059</td><td>0.031 0.057</td><td>0.152 0.235</td><td>0.045 0.104</td><td>0.039 0.076</td><td>0.038 0.087</td></tr></table>

Results As shown in Figure 5, TimesNet achieves the best performance with an average accuracy of 73.6%, surpassing the previous state-of-the-art classical method Rocket (72.5%) and deep model Flowformer (73.0%). It is also notable that the MLP-based model DLinear fails in this classification task (67.5%), which performs well in some time series forecasting datasets. This is because DLinear only adopts a one-layer MLP model on the temporal dimension, which might be suitable for some autoregressive tasks with fixed temporal 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/1159ba497885ef77b2f29aa2b973188af686582fc460241da13424d82bf8a146.jpg)



Figure 5: Model comparison in classification. “∗.” in the Transform ers indicates the name of ∗former. The results are averaged from 10 subsets of UEA. See Table 17 in Appendix for full results.


dependencies but will degenerate a lot in learning high-level representations. In contrast, TimesNet unifies the temporal 2D-variation in 2D space, which is convenient to learn informative representation by 2D kernels, thereby benefiting the classification task that requires hierarchical representations. 

## 4.5 ANOMALY DETECTION

Setups Detecting anomalies from monitoring data is vital to industrial maintenance. Since the anomalies are usually hidden in the large-scale data, making the data labeling hard, we focus on unsupervised time series anomaly detection, which is to detect the abnormal time points. We compare models on five widely-used anomaly detection benchmarks: SMD (Su et al., 2019), MSL (Hundman et al., 2018), SMAP (Hundman et al., 2018), SWaT (Mathur & Tippenhauer, 2016), PSM (Abdulaal et al., 2021), covering service monitoring, space & earth exploration, and water treatment applications. Following the pre-processing methods in Anomaly Transformer (2021), we split the dataset into consecutive non-overlapping segments by sliding window. In previous works, the reconstruction is a classical task for unsupervised point-wise representation learning, where the reconstruction error is a natural anomaly criterion. For a fair comparison, we only change the base models for reconstruction and use the classical reconstruction error as the shared anomaly criterion for all experiments. 

Results Table 5 demonstrates that TimesNet still achieves the best performance in anomaly detection, outperforming the advanced Transformer-based models FEDformer (2022) and Autoformer (2021). The canonical Transformer performs worse in this task (averaged F1-score 76.88%). This may come from that anomaly detection requires the model to find out the rare abnormal temporal patterns (Lai et al., 2021), while the vanilla attention mechanism calculates the similarity between each pair of time points, which can be distracted by the dominant normal time points. Besides, by taking the periodicity into consideration, TimesNet, FEDformer and Autoformer all achieve great performance. Thus, these results also demonstrate the importance of periodicity analysis, which can highlight variations that violate the periodicity implicitly, further benefiting the anomaly detection. 


Table 5: Anomaly detection task. We calculate the F1-score (as %) for each dataset. *. means the1.5<sup>1.5</sup> *former. A higher value of F1-score indicates a better performance. See Table 15 for full results.


<table><tr><td>Models</td><td>TimesNet (ResNeXt)</td><td>TimesNet (Inception)</td><td>ETS. (2022)</td><td>FED. (2022)</td><td>LightTS (2022)</td><td>DLinear (2023)</td><td>Stationary (2022a)</td><td>Auto. (2021)</td><td>Pyra. (2021a)</td><td>Anomaly* (2021)</td><td>In. (2021)</td><td>Re. (2020)</td><td>LogTrans (2019)</td><td>Trans. (2017)</td></tr><tr><td>SMD</td><td>85.81</td><td>85.12</td><td>83.13</td><td>85.08</td><td>82.53</td><td>77.10</td><td>84.72</td><td>85.11</td><td>83.04</td><td>85.49</td><td>81.65</td><td>75.32</td><td>76.21</td><td>79.56</td></tr><tr><td>MSL</td><td>85.15</td><td>84.18</td><td>85.03</td><td>78.57</td><td>78.95</td><td>84.88</td><td>77.50</td><td>79.05</td><td>84.86</td><td>83.31</td><td>84.06</td><td>84.40</td><td>79.57</td><td>78.68</td></tr><tr><td>SMAP</td><td>71.52</td><td>70.85</td><td>69.50</td><td>70.76</td><td>69.21</td><td>69.26</td><td>71.09</td><td>71.12</td><td>71.09</td><td>71.18</td><td>69.92</td><td>70.40</td><td>69.97</td><td>69.70</td></tr><tr><td>SWaT</td><td>91.74</td><td>92.10</td><td>84.91</td><td>93.19</td><td>93.33</td><td>87.52</td><td>79.88</td><td>92.74</td><td>91.78</td><td>83.10</td><td>81.43</td><td>82.80</td><td>80.52</td><td>80.37</td></tr><tr><td>PSM</td><td>97.47</td><td>95.21</td><td>91.76</td><td>97.23</td><td>97.15</td><td>93.55</td><td>97.29</td><td>93.29</td><td>82.08</td><td>79.40</td><td>77.10</td><td>73.61</td><td>76.74</td><td>76.07</td></tr><tr><td>Avg F1</td><td>86.34</td><td>85.49</td><td>82.87</td><td>84.97</td><td>84.23</td><td>82.46</td><td>82.08</td><td>84.26</td><td>82.57</td><td>80.50</td><td>78.83</td><td>77.31</td><td>76.60</td><td>76.88</td></tr></table>


∗ We replace the joint criterion in Anomaly Transformer (2021) with reconstruction error for fair comparison.


## 4.6 MODEL ANALYSIS

Representation analysis We attempt to explain model performance from the representation learning aspect. From Figure 6, we can find that the better performance in forecasting and anomaly detection corresponds to the higher CKA similarity (2019), which is opposite to the imputation and classification tasks. Note that the lower CKA similarity means that the representations are distinguishing among1<sup>0</sup><sub>0</sub> 20 40 60 80 different layers, namely hierarchical representations. Thus, these results also indicate the property0 of representations that each task requires. As shown in Figure 6, TimesNet can learn appropriate0 50 100 <sup>°1</sup> representations for different tasks, such as low-level representations for forecasting and reconstruction40 in anomaly detection, and hierarchical representations for imputation and classification. In contrast,20 FEDformer (2022) performs well in forecasting and anomaly detection tasks but fails in learning0 hierarchical representations, resulting in poor performance in imputation and classification. These results also verify the task-generality of our proposed TimesNet as a foundation model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/1580d8f7bd9bd455c0e48f87a459ae28bcba758bbc2d35fb28ca9717b1ddc04e.jpg)



(a) Forecasting (Weather input-96-predict-336)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/bd872fb72ab399291e9f9c87a1eac6c9c14f3c9b1000074171fbc26e8780d6bd.jpg)



(b) Imputation (Electricity Mask 37.5%)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/ed95e75115808f0ff388823ab6453d849dc55dfa0ecde02ec52efce8aa0fe4ec.jpg)



(c) Classification (PEMS-SF)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/7b501eaa49761c5624b09c97f6f1ebf4aca1b89f2142df031718ba94ea64f31f.jpg)



(d) Anomaly Detection (SMD)



Figure 6: Representation analysis in four tasks. For each model, we calculate the centered kernel alignment (CKA) similarity (2019) between representations from the first and the last layers. A5 higher CKA similarity indicates more similar representations. TimesNet is marked by red stars.


Temporal 2D-variations We provide a case study of temporal 2D-variations in Figure 7. We can find that TimesNet can capture the multi-periodicities precisely. Besides, the transformed 2D tensor is highly structured and informative, where the columns and rows can reflect the localities between time points and periods respectively, supporting our motivation in adopting 2D kernels for representation learning. See Appendix D for more visualizations. 

## 5 CONCLUSION AND FUTURE WORK

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/37d9b54d9bf517224ba7fe91aa320d6f6518b020088809009ec42cac3d0af213.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/13492edfeedc9633a506d28e6d6674601e7259d322ad38a9836f2f8b6c61aa7a.jpg)



Figure 7: A case of temporal 2D-variations.


This paper presents the TimesNet as a task-general foundation model for time series analysis. Motivated by the multi-periodicity, TimesNet can ravel out intricate temporal variations by a modular architecture and capture intraperiod- and interperiod-variations in 2D space by a parameter-efficient inception block. Experimentally, TimesNet shows great generality and performance in five mainstream analysis tasks. In the future, we will further explore large-scale pre-training methods in time series, which utilize TimesNet as the backbone and can generally benefit extensive downstream tasks. 

## ACKNOWLEDGMENTS

This work was supported by the National Key Research and Development Plan (2020AAA0109201), National Natural Science Foundation of China (62022050 and 62021002), Civil Aircraft Research Project (MZJ3-2N21), Beijing Nova Program (Z201100006820041), and CCF-Ant Group Green Computing Fund. 

## REFERENCES



Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. KDD, 2021. 





O. Anderson and M. Kendall. Time-series. 2nd edn. J. R. Stat. Soc. (Series D), 1976. 





Anthony J. Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron George Bostrom, Paul Southam, and Eamonn J. Keogh. The uea multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075, 2018. 





Donald J. Berndt and James Clifford. Using dynamic time warping to find patterns in time series. In KDD Workshop, 1994. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. 





CDC. Illness. https://gis.cdc.gov/grasp/fluview/fluportaldashboard. html. 





Cristian Challu, Kin G Olivares, Boris N Oreshkin, Federico Garza, Max Mergenthaler, and Artur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. arXiv preprint arXiv:2201.12886, 2022. 





Chris Chatfield. The analysis of time series: an introduction. 1981. 





Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. KDD, 2016. 





Angus Dempster, Franccois Petitjean, and Geoffrey I. Webb. Rocket: exceptionally fast and accurate time series classification using random convolutional kernels. Data Min. Knowl. Discov., 2020. 





Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 





Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jaggi. Unsupervised scalable representation learning for multivariate time series. In NeurIPS, 2019. 





Milton Friedman. The interpolation of time series by related series. J. Amer. Statist. Assoc, 1962. 





Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´ state spaces. In ICLR, 2022. 





Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR, 2016. 





Yangdong He and Jiabao Zhao. Temporal convolutional networks for anomaly detection in time series. J. Phys. Conf. Ser, 2019. 





S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 1997. 





Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstr¨ om.¨ Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. KDD, 2018. 





Rob J Hyndman and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2018. 





Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. URL http://arxiv.org/abs/1412.6980. 





Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2020. 





Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. Similarity of neural network representations revisited. In ICML, 2019. 





Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In SIGIR, 2018. 





Kwei-Herng Lai, D. Zha, Junjie Xu, and Yue Zhao. Revisiting time series outlier detection: Definitions and benchmarks. In NeurIPS Dataset and Benchmark Track, 2021. 





Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In NeurIPS, 2019. 





Bryan Lim and Stefan Zohren. Time-series forecasting with deep learning: a survey. Philos. Trans. Royal Soc. A, 2021. 





Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In ICLR, 2021a. 





Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Rethinking the stationarity in time series forecasting. In NeurIPS, 2022a. 





Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021b. 





Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. CVPR, 2022b. 





Aditya P. Mathur and Nils Ole Tippenhauer. Swat: a water treatment testbed for research and training on ICS security. In CySWATER, 2016. 





George B. Moody, Roger G. Mark, and Ary L. Goldberger. Physionet: Physiologic signals, time series and related open source software for basic, clinical, and applied research. EMBC, 2011. 





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR, 2019. 





Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito,¨ Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 





PeMS. Traffic. http://pems.dot.ca.gov/. 





Lifeng Shen, Zhuocong Li, and James T. Kwok. Timeseries anomaly detection using temporal hierarchical one-class network. In NeurIPS, 2020. 





Spyros Makridakis. M4 dataset, 2018. URL https://github.com/M4Competition/ M4-methods/tree/master/Dataset. 





Ya Su, Y. Zhao, Chenhao Niu, Rong Liu, W. Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. KDD, 2019. 





Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, D. Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. CVPR, 2015. 





Sean J Taylor and Benjamin Letham. Forecasting at scale. Am. Stat., 2018. 





UCI. Electricity. https://archive.ics.uci.edu/ml/datasets/ ElectricityLoadDiagrams20112014. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 





Wetterstation. Weather. https://www.bgc-jena.mpg.de/wetter/. 





Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven C. H. Hoi. Etsformer: Exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:2202.01381, 2022. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with Auto-Correlation for long-term series forecasting. In NeurIPS, 2021. 





Haixu Wu, Jialong Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Flowformer: Linearizing transformers with conservation flows. In ICML, 2022. 





Saining Xie, Ross B. Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual ´ transformations for deep neural networks. CVPR, 2017. 





Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In ICLR, 2021. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023. 





G. Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty, and Carsten Eickhoff. A transformer-based framework for multivariate time series representation learning. KDD, 2021. 





T. Zhang, Yizhuo Zhang, Wei Cao, J. Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022. 





Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI, 2021. 





Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In ICML, 2022. 



## A IMPLEMENTATION DETAILS

We provide the dataset descriptions and experiment configurations in Table 6 and 7. All experiments are repeated three times, implemented in PyTorch (Paszke et al., 2019) and conducted on a single NVIDIA TITAN RTX 24GB GPU. To make the model handle various dimensions of input series from different datasets, we select the $d _ { \mathrm { m o d e l } }$ based on the input series dimension C by min $\{ \operatorname* { m a x } \{ 2 ^ { \lceil \log C \rceil } , d _ { \operatorname* { m i n } } \} , d _ { \operatorname* { m a x } } \}$ (see Table 7 for details of $d _ { \mathrm { m i n } }$ and $d _ { \operatorname* { m a x } } )$ . This protocol can make the model powerful enough for multiple variates and also keep the model size compact. 

All the baselines that we reproduced are implemented based on configurations of the original paper or official code. It is also notable that none of the previous methods are proposed for general time series analysis. For a fair comparison, we keep the input embedding and the final projection layer the same among different base models and only evaluate the capability of base models. Especially for the forecasting task, we use a MLP on temporal dimension to get the initialization of predicted future. Since we focus on the temporal variation modeling, we also adopt the Series Stationarization from Non-stationary Transformer (Liu et al., 2022a) to eliminate the affect the distribution shift. 

For the metrics, we adopt the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting and imputations. For anomaly detection, we adopt the F1-score, which is the harmonic mean of precision and recall. For the short-term forecasting, following the N-BEATS (Oreshkin et al., 2019), we adopt the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE) and overall weighted average (OWA) as the metrics, where OWA is a special metric used in M4 competition. These metrics can be calculated as follows: 

$$
\begin{array}{l l} \text {SMAPE} = \frac {2 0 0}{H} \sum_ {i = 1} ^ {H} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} | + | \widehat {\mathbf {X}} _ {i} |}, & \text {MAPE} = \frac {1 0 0}{H} \sum_ {i = 1} ^ {H} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{| \mathbf {X} _ {i} |}, \\ \text {MASE} = \frac {1}{H} \sum_ {i = 1} ^ {H} \frac {| \mathbf {X} _ {i} - \widehat {\mathbf {X}} _ {i} |}{\frac {1}{H - m} \sum_ {j = m + 1} ^ {H} | \mathbf {X} _ {j} - \mathbf {X} _ {j - m} |}, & \text {OWA} = \frac {1}{2} \left[ \frac {\text {SMAPE}}{\text {SMAPE} _ {\text {Naïve2}}} + \frac {\text {MASE}}{\text {MASE} _ {\text {Naïve2}}} \right], \end{array}
$$

where m is the periodicity of the data. $\mathbf { X } , \widehat { \mathbf { X } } \in \mathbb { R } ^ { H \times C }$ are the ground truth and prediction results of bthe future with H time pints and C dimensions. Xi means the i-th future time point. 

## B HYPER-PARAMETER SENSITIVITY

We introduce a hyper-parameter in Equation 1 to select the most significant frequencies. We provide the sensitivity analysis for these two hyper-parameters in Figure 8. We can find that our proposed TimesNet can present a stable performance under different choices of k in all four tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/68a564c860b2aeceb5d53723c6b4d1ff7a94dd254c3f6f95fbf98e1e78edccce.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/9f6ada4010ca18bf547da472662afdc8decfb63bc8b2f658672f781b2f7560bc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/5a66c915a20dcbc61e47ca9c494b89b9b638b2b5b6aa9ad5d0e4e1e0f44d02ec.jpg)



k in Equation (1) for Classification Task


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/52b8f2e2976213a54beabb1f6369e54c2b2ce06694efb61c6ca45a0e832aab72.jpg)



k in Equation (1) for Short-term Forecasting Task



Figure 8: Sensitivity analysis of hyper-parameters k in imputation and anomaly detection task. Especially, we select the 37.5% mask ratio for hyper-parameter experiments. For classification, we choose the two largest subsets: SpokenArobicDigits and FaceDetection for evaluation. For short-term forecasting, we adopt the weighted average for sensitivity analysis.


Besides, from Figure 8, we can also find out the following observations: 

• For the low-level modeling tasks, such as forecasting and anomaly detection, the selection of k will affect the final performance more. This may come from that k will directly affect the amount of information in deep representations. 


Table 6: Dataset descriptions. The dataset size is organized in (Train, Validation, Test).


<table><tr><td>Tasks</td><td>Dataset</td><td>Dim</td><td>Series Length</td><td>Dataset Size</td><td>Information (Frequency)</td></tr><tr><td rowspan="7">Forecasting(Long-term)</td><td>ETTm1, ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>Electricity (15 mins)</td></tr><tr><td>ETTh1, ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Electricity (15 mins)</td></tr><tr><td>Electricity</td><td>321</td><td>{96, 192, 336, 720}</td><td>(18317, 2633, 5261)</td><td>Electricity (Hourly)</td></tr><tr><td>Traffic</td><td>862</td><td>{96, 192, 336, 720}</td><td>(12185, 1757, 3509)</td><td>Transportation (Hourly)</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>Weather (10 mins)</td></tr><tr><td>Exchange</td><td>8</td><td>{96, 192, 336, 720}</td><td>(5120, 665, 1422)</td><td>Exchange rate (Daily)</td></tr><tr><td>ILI</td><td>7</td><td>{24, 36, 48, 60}</td><td>(617, 74, 170)</td><td>Illness (Weekly)</td></tr><tr><td rowspan="6">Forecasting(short-term)</td><td>M4-Yearly</td><td>1</td><td>6</td><td>(23000, 0, 23000)</td><td>Demographic</td></tr><tr><td>M4-Quarterly</td><td>1</td><td>8</td><td>(24000, 0, 24000)</td><td>Finance</td></tr><tr><td>M4-Monthly</td><td>1</td><td>18</td><td>(48000, 0, 48000)</td><td>Industry</td></tr><tr><td>M4-Weakly</td><td>1</td><td>13</td><td>(359, 0, 359)</td><td>Macro</td></tr><tr><td>M4-Daily</td><td>1</td><td>14</td><td>(4227, 0, 4227)</td><td>Micro</td></tr><tr><td>M4-Hourly</td><td>1</td><td>48</td><td>(414, 0, 414)</td><td>Other</td></tr><tr><td rowspan="4">Imputation</td><td>ETTm1, ETTm2</td><td>7</td><td>96</td><td>(34465, 11521, 11521)</td><td>Electricity (15 mins)</td></tr><tr><td>ETTh1, ETTh2</td><td>7</td><td>96</td><td>(8545, 2881, 2881)</td><td>Electricity (15 mins)</td></tr><tr><td>Electricity</td><td>321</td><td>96</td><td>(18317, 2633, 5261)</td><td>Electricity (15 mins)</td></tr><tr><td>Weather</td><td>21</td><td>96</td><td>(36792, 5271, 10540)</td><td>Weather (10 mins)</td></tr><tr><td rowspan="10">Classification(UEA)</td><td>EthanolConcentration</td><td>3</td><td>1751</td><td>(261, 0, 263)</td><td>Alcohol Industry</td></tr><tr><td>FaceDetection</td><td>144</td><td>62</td><td>(5890, 0, 3524)</td><td>Face (250Hz)</td></tr><tr><td>Handwriting</td><td>3</td><td>152</td><td>(150, 0, 850)</td><td>Handwriting</td></tr><tr><td>Heartbeat</td><td>61</td><td>405</td><td>(204, 0, 205)</td><td>Heart Beat</td></tr><tr><td>JapaneseVowels</td><td>12</td><td>29</td><td>(270, 0, 370)</td><td>Voice</td></tr><tr><td>PEMS-SF</td><td>963</td><td>144</td><td>(267, 0, 173)</td><td>Transportation (Daily)</td></tr><tr><td>SelfRegulationSCP1</td><td>6</td><td>896</td><td>(268, 0, 293)</td><td>Health (256Hz)</td></tr><tr><td>SelfRegulationSCP2</td><td>7</td><td>1152</td><td>(200, 0, 180)</td><td>Health (256Hz)</td></tr><tr><td>SpokenArabicDigits</td><td>13</td><td>93</td><td>(6599, 0, 2199)</td><td>Voice (11025Hz)</td></tr><tr><td>UWaveGestureLibrary</td><td>3</td><td>315</td><td>(120, 0, 320)</td><td>Gesture</td></tr><tr><td rowspan="5">Anomaly Detection</td><td>SMD</td><td>38</td><td>100</td><td>(566724, 141681, 708420)</td><td>Server Machine</td></tr><tr><td>MSL</td><td>55</td><td>100</td><td>(44653, 11664, 73729)</td><td>Spacecraft</td></tr><tr><td>SMAP</td><td>25</td><td>100</td><td>(108146, 27037, 427617)</td><td>Spacecraft</td></tr><tr><td>SWaT</td><td>51</td><td>100</td><td>(396000, 99000, 449919)</td><td>Infrastructure</td></tr><tr><td>PSM</td><td>25</td><td>100</td><td>(105984, 26497, 87841)</td><td>Server Machine</td></tr></table>


Table 7: Experiment configuration of TimesNet. All the experiments use the ADAM (2015) optimizer with the default hyperparameter configuration for $( \beta _ { 1 } , \beta _ { 2 } )$ as (0.9, 0.999).


<table><tr><td rowspan="2">Tasks / Configurations</td><td colspan="4">Model Hyper-parameter</td><td colspan="4">Training Process</td></tr><tr><td>k (Equ. 1)</td><td>Layers</td><td><eq>d_{\text{min}} \dagger</eq></td><td><eq>d_{\text{max}} \dagger</eq></td><td>LR*</td><td>Loss</td><td>Batch Size</td><td>Epochs</td></tr><tr><td>Long-term Forecasting</td><td>5</td><td>2</td><td>32</td><td>512</td><td><eq>10^{-4}</eq></td><td>MSE</td><td>32</td><td>10</td></tr><tr><td>Short-term Forecasting</td><td>5</td><td>2</td><td>16</td><td>64</td><td><eq>10^{-3}</eq></td><td>SMAPE</td><td>16</td><td>10</td></tr><tr><td>Imputation</td><td>3</td><td>2</td><td>64</td><td>128</td><td><eq>10^{-3}</eq></td><td>MSE</td><td>16</td><td>10</td></tr><tr><td>Classification</td><td>3</td><td>2</td><td>32</td><td>64</td><td><eq>10^{-3}</eq></td><td>Cross Entropy</td><td>16</td><td>30</td></tr><tr><td>Anomaly Detection</td><td>3</td><td>3</td><td>32</td><td>128</td><td><eq>10^{-4}</eq></td><td>MSE</td><td>128</td><td>10</td></tr></table>


$\mathrm { ~ \# ~ } d _ { \mathrm { m o d e l } } = \operatorname* { m i n } \{ \operatorname* { m a x } \{ 2 ^ { \lceil \log C \rceil } , d _ { \operatorname* { m i n } } \} , d _ { \operatorname* { m a x } } \}$ , where C is input series dimension. 



∗ LR means the initial learning rate. 


• For the high-level modeling tasks, such as classification and imputation, the model performance will be more robust to the selection of k, since the key to these tasks is to extract hierarchical representations. 

Giving consideration to both efficiency and performance, we set k = 3 for imputation, classification and anomaly detection and k = 5 for short-term forecasting. 

## C ABLATION STUDIES

To elaborate the property of our proposed TimesNet, we conduct detailed ablations on the representa tion leaning in 2D space, model architecture and adaptive aggregation. 

2D space As shown in Table 8, replacing the inception block with more powerful blocks will bring further performance promotion, such as ResNeXt (Xie et al., 2017), Swin Transformer (Liu et al., 2021b) and ConvNeXt (Liu et al., 2022b). It is also notable that using the independent parameters will also bring improvement, while this will cause the model size related to the selection of hyperparameter k. Considering the efficiency and model performance, we choose the parameter-efficient inception block as our final solution. These results also verify that our design bridges the 1D time series analysis with 2D computer vision backbones. 


Table 8: Ablations on the representation leaning in 2D space, where we replace the parameter-efficient inception with other well-acknowledged vision backbones. See Figure 4 for efficiency comparison.


<table><tr><td rowspan="2">DatasetsMetrics</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td><td rowspan="2">Avg F1(%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>ResNet</td><td>87.64</td><td>80.33</td><td>83.82</td><td>83.99</td><td>85.42</td><td>84.70</td><td>92.09</td><td>57.90</td><td>71.10</td><td>86.59</td><td>95.81</td><td>90.97</td><td>98.24</td><td>89.12</td><td>93.45</td><td>84.81</td></tr><tr><td>ResNeXt</td><td>88.66</td><td>83.14</td><td>85.81</td><td>83.92</td><td>86.42</td><td>85.15</td><td>92.52</td><td>58.29</td><td>71.52</td><td>86.76</td><td>97.32</td><td>91.74</td><td>98.19</td><td>96.76</td><td>97.47</td><td>86.34</td></tr><tr><td>Swin Transformer</td><td>88.51</td><td>82.22</td><td>85.25</td><td>87.36</td><td>86.93</td><td>87.14</td><td>92.12</td><td>57.60</td><td>70.88</td><td>89.02</td><td>95.81</td><td>92.29</td><td>98.45</td><td>91.80</td><td>95.01</td><td>86.11</td></tr><tr><td>ConvNext</td><td>87.89</td><td>84.67</td><td>86.25</td><td>87.31</td><td>86.93</td><td>87.12</td><td>92.42</td><td>59.19</td><td>72.16</td><td>89.05</td><td>95.81</td><td>92.31</td><td>97.99</td><td>95.28</td><td>96.62</td><td>86.89</td></tr><tr><td>Inception (Ind*)</td><td>87.54</td><td>81.04</td><td>84.17</td><td>87.44</td><td>86.93</td><td>87.18</td><td>91.92</td><td>57.69</td><td>70.89</td><td>88.98</td><td>96.00</td><td>92.36</td><td>98.11</td><td>89.13</td><td>93.41</td><td>85.60</td></tr><tr><td>Inception (Shared*)</td><td>87.76</td><td>82.63</td><td>85.12</td><td>82.97</td><td>85.42</td><td>84.18</td><td>91.50</td><td>57.80</td><td>70.85</td><td>88.31</td><td>96.24</td><td>92.10</td><td>98.22</td><td>92.21</td><td>95.21</td><td>85.49</td></tr></table>


∗ In this paper, we adopt a parameter-efficient design that uses the same parameters for k different transformed 2D tensors, namely Shared. For comparison, we also compare with the independent design, that uses different parameters for different transformed 2D tensors, namely Ind. 


Model architecture We also conduct experiments on different architectures. Surprisingly, as shown in Table 9, we find that combining with the deep decomposition architecture in Autoformer (Wu et al., 2021) cannot bring further promotion. These results may come from that in the case that the input series already present clear periodicity, our design can capture the 2D-variations effectively. As for the case that they are without clear periodicity, the model will learn the most significant frequency as 1, where the trend of time series be covered by the intraperiod-variation modeling. These results also verify that our proposed TimesNet can handle the analysis for time series without clear periodicity. 

Besides, in this paper, to take advantages of the deep representations, we place the transformation from 1D-variations to 2D-variations in every TimesBlock. Here, we compare our design with the case that only conducts the transformation on the raw input series. From Table 9, we can find that the performance of Transform raw data degenerates a lot (Avg F1-score: 85.49% 84.85%), indicating the advantages of our design. 


Table 9: Ablations on model architecture. + decomposition is to combine the deep decomposition architecture proposed by Autoformer (Wu et al., 2021) with TimesNet. Transform raw data refers conducting the transformation on the original time series, instead of deep features.


<table><tr><td rowspan="2">DatasetsMetrics</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td><td rowspan="2">Avg F1(%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>TimesNet</td><td>87.76</td><td>82.63</td><td>85.12</td><td>82.97</td><td>85.42</td><td>84.18</td><td>91.50</td><td>57.80</td><td>70.85</td><td>88.31</td><td>96.24</td><td>92.10</td><td>98.22</td><td>92.21</td><td>95.21</td><td>85.49</td></tr><tr><td>+ decomposition</td><td>87.44</td><td>78.49</td><td>82.72</td><td>83.48</td><td>86.47</td><td>84.95</td><td>91.64</td><td>57.34</td><td>70.54</td><td>89.68</td><td>95.60</td><td>92.54</td><td>98.42</td><td>93.12</td><td>95.69</td><td>85.29</td></tr><tr><td>Transform raw data</td><td>86.83</td><td>79.17</td><td>82.82</td><td>85.23</td><td>86.47</td><td>85.84</td><td>91.92</td><td>57.60</td><td>70.82</td><td>87.68</td><td>95.81</td><td>91.57</td><td>97.64</td><td>89.14</td><td>93.20</td><td>84.85</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/5c8ab9aacf096c76f9a386fb1cc3b12b9d3d71cbe7b33c3759188e9961f0487f.jpg)


Adaptive aggregation As shown in Equation 6, following the design in Autoformer (2021), we adopt the amplitudes after the Softmax function as aggregation weights of processed tensors $\{ \widehat { \mathbf { X } } _ { \mathrm { 1 D } } ^ { l , 1 } , \cdot \cdot \cdot , \widehat { \mathbf { X } } _ { \mathrm { 1 D } } ^ { l , k } \}$ . Here we include two variants for comparison. The first is directly-sum, namely $\textstyle \sum _ { i = 1 } ^ { k } { \widehat { \mathbf { X } } } _ { \mathrm { 1 D } } ^ { l , i } .$ bThe second is to remove the Softmax function, that is $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \mathbf { A } _ { f _ { i } } ^ { l - 1 } \times \widehat { \mathbf { X } } _ { \mathrm { 1 D } } ^ { l , i } } \end{array}$ . Table 10 bdemonstrates that our design in adaptive aggregation performs the best. 


Table 10: Ablations on adaptive aggregation.


<table><tr><td rowspan="2">DatasetsMetrics</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td><td rowspan="2">Avg F1(%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>TimesNet</td><td>87.76</td><td>82.63</td><td>85.12</td><td>82.97</td><td>85.42</td><td>84.18</td><td>91.50</td><td>57.80</td><td>70.85</td><td>88.31</td><td>96.24</td><td>92.10</td><td>98.22</td><td>92.21</td><td>95.21</td><td>85.49</td></tr><tr><td>Directly-sum</td><td>87.10</td><td>78.93</td><td>82.81</td><td>85.81</td><td>85.42</td><td>85.62</td><td>91.35</td><td>57.58</td><td>70.64</td><td>87.28</td><td>96.00</td><td>91.43</td><td>98.13</td><td>87.22</td><td>92.35</td><td>84.57</td></tr><tr><td>Removing-Softmax</td><td>87.27</td><td>79.31</td><td>83.10</td><td>83.91</td><td>86.47</td><td>85.17</td><td>91.93</td><td>58.57</td><td>71.55</td><td>87.13</td><td>95.81</td><td>91.27</td><td>98.00</td><td>92.48</td><td>95.16</td><td>85.25</td></tr></table>

## D MORE REPRESENTATION ANALYSIS

To give an intuitive understanding of 2D-variations, we visualize the transformed 2D tensors in Figure 9. From the visualization, we can obtain the following observations: 

• The interperiod-variation can present the long-term trends of time series. For example, in the first case of Exchange, the values in each row decrease from left to right, indicating the downtrend of the raw series. And for the ETTh1 dataset, the values in each row are similar to each other, reflecting the global stable variation of the raw series. 

• For the time series without clear periodicity, the temporal 2D-variations can still present informative 2D structure. If the frequency is one, the intraperiod-variation is just the original variation of raw series. Besides, the interperiod-variation can also present the long-termID Series 2D Variations trend, benefiting the temporal variation modeling.0 

• The transformed 2D-variations demonstrate two types of localities. Firstly, for each column (intraperiod-variation), the adjacent values are close to each other, presenting the locality20 among adjacent time points. Secondly, for each row (interperiod-variation), the adjacent10 <sup>15 0.8</sup> 30 values are also close, corresponding to the locality among adjacent periods. Note that the40 non-adjacent periods can be quite different from each other, which can be caused by global<sup>top_3</sup> <sup>freq=5,9,15 15 25</sup> <sub>50</sub> trend, such as the case from the Exchange dataset. These observations of localities also<sup>30</sup> motivate us to adopt the 2D kernel for representation learning.<sup>35</sup> 


Figure 9: More showcases for temporal 2D-variations.


## E MULTI-PERIODICITY OF TIME SERIES

As shown in Figure 10, we calculate the density of each period length for different datasets. We can find that real-world time series present multi-periodicity to some extent. For example, the Electricity dataset contains the periods with length-12 and length-24. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/aa04ad6f43b993ca10612e8d5531fdd829b376d66d9156f125fb79f525f5071e.jpg)



(a) Electricity


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/ed9c7667d8838bb0fc29da86942950d2b4abc213331cb5221fd290bec1d805e9.jpg)



(b) ETTh1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/01dc2819b097563a03c4a2e3192bbed60da7e3071f50a0c8dd604b65ea137022.jpg)



(c) Exchange


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/beb2c857ab8dbea40ee586a3812054839bf2d8017faae51e11b09edbdcb6cbca.jpg)



(d) Weather



Figure 10: Statistics of period length in experimental datasets. We conduct FFT to the raw data and select the top-6 significant frequencies for each length-96 segment. Then, we record the corresponding period lengths and plot the normalized density for each period length.


## F SHOWCASES

To provide a clear comparison among different models, we provide showcases to the regression tasks, including the imputation (Figure 11), long-term forecasting (Figure 12) and short-term forecasting (Figure 13). Especially in the imputation task, the MLP-based models degenerate a lot. This is because the input series has been randomly masked. However, the MLP-based models adopt the fixed model parameter to model the temporal dependencies among time points, thereby failing in this task. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/2c7f068642b73a382f0c623e58bf6905309ed02253cd24cbe90a1fc0f62e9dc7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/2ff37ded447037d6441b536a608ace8c18e0517f16ace93dd2696780acc43c77.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/9e7c38fc54e5739c8a92a2fcc3c9592af2f4107546fc6292a649c57e5907c9f7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/d110d06b67cf8b847651b4a191f171d55d4755c83c2385bd25a1853f9574b084.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/c059e01f3faee3d14215ca60f44355eadd0b7083cac2365167d32766f55812f2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/9f9e2fd3af5f68805667df765e3bc5c8d38db80b012ec84b98a216fa5bfb4cf7.jpg)



Figure 11: Visualization of ETTm1 imputation results given by models under the 50% mask ratio setting. The black lines stand for the ground truth and the orange lines stand for predicted values.


## G MODEL EFFICIENCY ANALYSIS

To summarize the model performance and efficiency, we calculate the relative performance rankings for comparing baselines. The rankings are compared from the common models used in all five tasks: LSTM (1997) and LSSL (2022); TCN (2019); LightTS (2022) and DLinear (2023); Reformer (2020), Informer (2021), Pyraformer (2021a), Autoformer (2021), FEDformer (2022), Non-stationary Transformer (2022a), ETSformer (2022) and our proposed TimesNet, namely 13 models in total. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/6f3e6335c4f85b763f198359d61dea18158da6725d523b38ce7c3b0074a6e341.jpg)



Figure 12: Visualization of ETTm2 predictions by different models under the input-96-predict-336 setting. The black lines stand for the ground truth and the orange lines stand for predicted values.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-05/33a016ef-08ef-47e1-bc4c-530538bb4e5f/e302ecf43a13c64a6b615a470b498056c974482922de9b144e1361cf5d3cf172.jpg)



Figure 13: Visualization of M4 predictions by different models. The black lines stand for the ground truth and the orange lines stand for predicted values.


As shown in Table 11, our proposed TimesNet achieves the best performance in all five tasks. Among the top three models, TimesNet also achieves the greatest efficiency. Compared to MLP-based models, our proposed TimesNet shows a significant advantage in performance. And benefiting from the utilization of 2D kernels and parameter-efficient design, the parameter size is invariant when the input series changes. Compared to Transformer-based models, TimesNet is with great efficiency in GPU memory, which is essential in long sequence modeling. 


Table 11: Model efficiency comparison and their rankings in five tasks. The efficiency measurements are recorded on the imputation task of ETTh1 dataset. The rankings are organized in the order of long- and short-term forecasting, imputation, classification and anomaly detection. “/” indicates the out-of-memory situation. A smaller ranking means better performance.


<table><tr><td rowspan="2" colspan="2">Models Series Length</td><td rowspan="2">Parameter (MB)</td><td rowspan="2">GPU Memory (MiB)</td><td rowspan="2">Running Time (s / iter)</td><td colspan="2">Ranking</td></tr><tr><td>Five tasks</td><td>Avg Ranking</td></tr><tr><td rowspan="4">TimesNet (ours)</td><td>384</td><td>0.067</td><td>1245</td><td>0.024</td><td rowspan="4">(1, 1, 1, 1, 1)</td><td rowspan="4">1.0</td></tr><tr><td>768</td><td>0.067</td><td>1585</td><td>0.040</td></tr><tr><td>1536</td><td>0.067</td><td>2491</td><td>0.045</td></tr><tr><td>3072</td><td>0.067</td><td>2353</td><td>0.073</td></tr><tr><td rowspan="4">Non-stationary Transformer</td><td>384</td><td>1.884</td><td>2321</td><td>0.046</td><td rowspan="4">(3, 2, 2, 2, 8)</td><td rowspan="4">3.4</td></tr><tr><td>768</td><td>1.910</td><td>4927</td><td>0.118</td></tr><tr><td>1536</td><td>1.961</td><td>/</td><td>/</td></tr><tr><td>3072</td><td>/</td><td>/</td><td>/</td></tr><tr><td rowspan="4">Autoformer</td><td>384</td><td>1.848</td><td>2101</td><td>0.070</td><td rowspan="4">(7, 4, 3, 5, 3)</td><td rowspan="4">4.4</td></tr><tr><td>768</td><td>1.848</td><td>3209</td><td>0.071</td></tr><tr><td>1536</td><td>1.848</td><td>5395</td><td>0.129</td></tr><tr><td>3072</td><td>1.848</td><td>10043</td><td>0.255</td></tr><tr><td rowspan="4">FEDformer</td><td>384</td><td>2.901</td><td>5977</td><td>0.807</td><td rowspan="4">(4, 3, 6, 9, 2)</td><td rowspan="4">4.8</td></tr><tr><td>768</td><td>2.901</td><td>7111</td><td>1.055</td></tr><tr><td>1536</td><td>2.901</td><td>9173</td><td>1.482</td></tr><tr><td>3072</td><td>2.901</td><td>/</td><td>/</td></tr><tr><td rowspan="4">LightTS</td><td>384</td><td>0.163</td><td>1055</td><td>0.009</td><td rowspan="4">(6, 5, 4, 10, 4)</td><td rowspan="4">5.8</td></tr><tr><td>768</td><td>0.614</td><td>1077</td><td>0.013</td></tr><tr><td>1536</td><td>2.403</td><td>1127</td><td>0.015</td></tr><tr><td>3072</td><td>9.534</td><td>1311</td><td>0.030</td></tr><tr><td rowspan="4">DLinear</td><td>384</td><td>0.296</td><td>1057</td><td>0.006</td><td rowspan="4">(2, 6, 5, 12, 7)</td><td rowspan="4">6.4</td></tr><tr><td>768</td><td>1.181</td><td>1093</td><td>0.006</td></tr><tr><td>1536</td><td>4.722</td><td>1159</td><td>0.007</td></tr><tr><td>3072</td><td>18.881</td><td>1433</td><td>0.026</td></tr><tr><td rowspan="4">ETSformer</td><td>384</td><td>1.123</td><td>1831</td><td>0.042</td><td rowspan="4">(5, 9, 9, 6, 5)</td><td rowspan="4">6.8</td></tr><tr><td>768</td><td>1.123</td><td>2565</td><td>0.047</td></tr><tr><td>1536</td><td>1.123</td><td>4081</td><td>0.072</td></tr><tr><td>3072</td><td>1.123</td><td>7065</td><td>0.143</td></tr><tr><td rowspan="4">Informer</td><td>384</td><td>1.903</td><td>1577</td><td>0.044</td><td rowspan="4">(10, 8, 8, 3, 9)</td><td rowspan="4">7.6</td></tr><tr><td>768</td><td>1.903</td><td>2125</td><td>0.047</td></tr><tr><td>1536</td><td>1.903</td><td>3153</td><td>0.088</td></tr><tr><td>3072</td><td>1.903</td><td>5194</td><td>0.165</td></tr><tr><td rowspan="4">Reformer</td><td>384</td><td>1.157</td><td>1681</td><td>0.030</td><td rowspan="4">(11, 11, 7, 4, 11)</td><td rowspan="4">8.8</td></tr><tr><td>768</td><td>1.157</td><td>2301</td><td>0.046</td></tr><tr><td>1536</td><td>1.157</td><td>5793</td><td>0.102</td></tr><tr><td>3072</td><td>1.157</td><td>/</td><td>/</td></tr><tr><td rowspan="4">Pyraformer</td><td>384</td><td>1.308</td><td>2047</td><td>0.046</td><td rowspan="4">(9, 10, 12, 8, 6)</td><td rowspan="4">9.0</td></tr><tr><td>768</td><td>1.996</td><td>6077</td><td>0.119</td></tr><tr><td>1536</td><td>3.372</td><td>/</td><td>/</td></tr><tr><td>3072</td><td>/</td><td>/</td><td>/</td></tr><tr><td rowspan="4">LSSL</td><td>384</td><td>0.121</td><td>1135</td><td>0.010</td><td rowspan="4">(8, 12, 10, 7, 13)</td><td rowspan="4">10.0</td></tr><tr><td>768</td><td>0.220</td><td>1139</td><td>0.011</td></tr><tr><td>1536</td><td>0.417</td><td>1147</td><td>0.013</td></tr><tr><td>3072</td><td>0.812</td><td>1197</td><td>0.032</td></tr><tr><td rowspan="4">TCN</td><td>384</td><td>0.372</td><td>1195</td><td>0.020</td><td rowspan="4">(12, 7, 11, 11, 12)</td><td rowspan="4">10.6</td></tr><tr><td>768</td><td>0.372</td><td>1333</td><td>0.020</td></tr><tr><td>1536</td><td>0.372</td><td>1533</td><td>0.025</td></tr><tr><td>3072</td><td>0.372</td><td>1983</td><td>0.061</td></tr><tr><td rowspan="4">LSTM</td><td>384</td><td>0.268</td><td>1201</td><td>0.064</td><td rowspan="4">(13, 13, 13, 13, 10)</td><td rowspan="4">12.4</td></tr><tr><td>768</td><td>0.268</td><td>1323</td><td>0.122</td></tr><tr><td>1536</td><td>0.268</td><td>1539</td><td>0.229</td></tr><tr><td>3072</td><td>0.268</td><td>2017</td><td>0.452</td></tr></table>

## H MODEL PERFORMANCE IN MIXED DATASET

To verify the model capacity in large-scale pre-training, we evaluate the model performance when it is trained from a mixed dataset. Concretely, we mixed the hourly-collected ETTh1, ETTh2 and the 15-minute collected ETTm1, ETTm2 as the mixed dataset. Note that this mixed dataset contains diverse temporal patterns and periodicities in different data instances, making the unified training challenging. From Table 12, we can find that TimesNet can handle this mixed dataset well and generally promote the model performance in four independent subsets. 

Besides, we can also find that except TimesNet, for other baselines, the mixed training may decrease the model performance in some subsets, indicating that other baselines cannot handle the complex periodicities in the mixed dataset. These results also verify the potential of TimesNet in performing as the general-purpose backbone for large-scale pre-training in time series. 


Table 12: Comparison between unified training and independent training for imputation task.


<table><tr><td rowspan="2"></td><td colspan="2">Datasets</td><td colspan="4">ETTm1</td><td colspan="4">ETTm2</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td></tr><tr><td colspan="2">Mask Ratio</td><td>12.5%</td><td>25%</td><td>37.5%</td><td>50%</td><td>12.5%</td><td>25%</td><td>37.5%</td><td>50%</td><td>12.5%</td><td>25%</td><td>37.5%</td><td>50%</td><td>12.5%</td><td>25%</td><td>37.5%</td><td>50%</td></tr><tr><td rowspan="4">Autoformer</td><td rowspan="2">Unified</td><td>MSE</td><td>0.034</td><td>0.048</td><td>0.060</td><td>0.078</td><td>0.023</td><td>0.027</td><td>0.030</td><td>0.034</td><td>0.066</td><td>0.086</td><td>0.114</td><td>0.133</td><td>0.042</td><td>0.049</td><td>0.055</td><td>0.065</td></tr><tr><td>MAE</td><td>0.122</td><td>0.146</td><td>0.163</td><td>0.185</td><td>0.091</td><td>0.102</td><td>0.109</td><td>0.117</td><td>0.174</td><td>0.200</td><td>0.229</td><td>0.247</td><td>0.135</td><td>0.147</td><td>0.157</td><td>0.171</td></tr><tr><td rowspan="2">Independent</td><td>MSE</td><td>0.034</td><td>0.046</td><td>0.057</td><td>0.067</td><td>0.023</td><td>0.026</td><td>0.030</td><td>0.035</td><td>0.074</td><td>0.090</td><td>0.109</td><td>0.137</td><td>0.044</td><td>0.050</td><td>0.060</td><td>0.068</td></tr><tr><td>MAE</td><td>0.124</td><td>0.144</td><td>0.161</td><td>0.174</td><td>0.092</td><td>0.101</td><td>0.108</td><td>0.119</td><td>0.182</td><td>0.203</td><td>0.222</td><td>0.248</td><td>0.138</td><td>0.149</td><td>0.163</td><td>0.173</td></tr><tr><td rowspan="4">FEDformer</td><td rowspan="2">Unified</td><td>MSE</td><td>0.041</td><td>0.057</td><td>0.073</td><td>0.099</td><td>0.060</td><td>0.089</td><td>0.125</td><td>0.172</td><td>0.077</td><td>0.101</td><td>0.130</td><td>0.164</td><td>0.087</td><td>0.125</td><td>0.161</td><td>0.214</td></tr><tr><td>MAE</td><td>0.143</td><td>0.169</td><td>0.192</td><td>0.224</td><td>0.166</td><td>0.205</td><td>0.244</td><td>0.287</td><td>0.196</td><td>0.228</td><td>0.258</td><td>0.289</td><td>0.204</td><td>0.246</td><td>0.283</td><td>0.326</td></tr><tr><td rowspan="2">Independent</td><td>MSE</td><td>0.035</td><td>0.052</td><td>0.069</td><td>0.089</td><td>0.056</td><td>0.080</td><td>0.110</td><td>0.156</td><td>0.070</td><td>0.106</td><td>0.124</td><td>0.165</td><td>0.095</td><td>0.137</td><td>0.187</td><td>0.232</td></tr><tr><td>MAE</td><td>0.135</td><td>0.166</td><td>0.191</td><td>0.218</td><td>0.159</td><td>0.195</td><td>0.231</td><td>0.276</td><td>0.190</td><td>0.236</td><td>0.258</td><td>0.299</td><td>0.212</td><td>0.258</td><td>0.304</td><td>0.341</td></tr><tr><td rowspan="4">TimesNet</td><td rowspan="2">Unified</td><td>MSE</td><td>0.019</td><td>0.023</td><td>0.028</td><td>0.037</td><td>0.018</td><td>0.020</td><td>0.022</td><td>0.025</td><td>0.035</td><td>0.046</td><td>0.057</td><td>0.075</td><td>0.032</td><td>0.036</td><td>0.040</td><td>0.047</td></tr><tr><td>MAE</td><td>0.091</td><td>0.099</td><td>0.109</td><td>0.123</td><td>0.075</td><td>0.081</td><td>0.086</td><td>0.095</td><td>0.126</td><td>0.144</td><td>0.159</td><td>0.181</td><td>0.112</td><td>0.119</td><td>0.129</td><td>0.140</td></tr><tr><td rowspan="2">Independent</td><td>MSE</td><td>0.019</td><td>0.023</td><td>0.029</td><td>0.037</td><td>0.018</td><td>0.020</td><td>0.023</td><td>0.026</td><td>0.057</td><td>0.069</td><td>0.084</td><td>0.102</td><td>0.040</td><td>0.046</td><td>0.052</td><td>0.060</td></tr><tr><td>MAE</td><td>0.092</td><td>0.101</td><td>0.111</td><td>0.124</td><td>0.080</td><td>0.085</td><td>0.091</td><td>0.098</td><td>0.159</td><td>0.178</td><td>0.196</td><td>0.215</td><td>0.130</td><td>0.141</td><td>0.151</td><td>0.162</td></tr></table>

## I FULL RESULTS

Due to the space limitation of the main text, we place the full results of all experiments in the following: long-term forecasting in Table 13, short-term forecasting in Table 14, imputation in Table 16, classification in Table 17 and anomaly detection in Table 15. 


Table 13: Full results for the long-term forecasting task. We compare extensive competitive models under different prediction lengths. The input sequence length is set to 36 for the ILI dataset and 96 for the others. Avg is averaged from all four prediction lengths.


<table><tr><td colspan="2">Models</td><td colspan="2">TimesNet(Ours)</td><td colspan="2">ETSformer(2022)</td><td colspan="2">LightTS*(2022)</td><td colspan="2">DLinear*(2023)</td><td colspan="2">FEDformer(2022)</td><td colspan="2">Stationary(2022a)</td><td colspan="2">Autoformer(2021)</td><td colspan="2">Pyraformer(2021a)</td><td colspan="2">Informer(2021)</td><td colspan="2">LogTrans(2019)</td><td colspan="2">Reformer(2020)</td><td colspan="2">LSSL(2022)</td><td colspan="2">LSTM(1997)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.338</td><td>0.375</td><td>0.375</td><td>0.398</td><td>0.374</td><td>0.400</td><td>0.345</td><td>0.372</td><td>0.379</td><td>0.419</td><td>0.386</td><td>0.398</td><td>0.505</td><td>0.475</td><td>0.543</td><td>0.510</td><td>0.672</td><td>0.571</td><td>0.600</td><td>0.546</td><td>0.538</td><td>0.528</td><td>0.450</td><td>0.477</td><td>0.863</td><td>0.664</td></tr><tr><td>192</td><td>0.374</td><td>0.387</td><td>0.408</td><td>0.410</td><td>0.400</td><td>0.407</td><td>0.380</td><td>0.389</td><td>0.426</td><td>0.441</td><td>0.459</td><td>0.444</td><td>0.553</td><td>0.496</td><td>0.557</td><td>0.537</td><td>0.795</td><td>0.669</td><td>0.837</td><td>0.700</td><td>0.658</td><td>0.592</td><td>0.469</td><td>0.481</td><td>1.113</td><td>0.776</td></tr><tr><td>336</td><td>0.410</td><td>0.411</td><td>0.435</td><td>0.428</td><td>0.438</td><td>0.438</td><td>0.413</td><td>0.413</td><td>0.445</td><td>0.459</td><td>0.495</td><td>0.464</td><td>0.621</td><td>0.537</td><td>0.754</td><td>0.655</td><td>1.212</td><td>0.871</td><td>1.124</td><td>0.832</td><td>0.898</td><td>0.721</td><td>0.583</td><td>0.574</td><td>1.267</td><td>0.832</td></tr><tr><td>720</td><td>0.478</td><td>0.450</td><td>0.499</td><td>0.462</td><td>0.527</td><td>0.502</td><td>0.474</td><td>0.453</td><td>0.543</td><td>0.490</td><td>0.585</td><td>0.516</td><td>0.671</td><td>0.561</td><td>0.908</td><td>0.724</td><td>1.166</td><td>0.823</td><td>1.153</td><td>0.820</td><td>1.102</td><td>0.841</td><td>0.632</td><td>0.596</td><td>1.324</td><td>0.858</td></tr><tr><td></td><td>Avg</td><td>0.400</td><td>0.406</td><td>0.429</td><td>0.425</td><td>0.435</td><td>0.437</td><td>0.403</td><td>0.407</td><td>0.448</td><td>0.452</td><td>0.481</td><td>0.456</td><td>0.588</td><td>0.517</td><td>0.691</td><td>0.607</td><td>0.961</td><td>0.734</td><td>0.929</td><td>0.725</td><td>0.799</td><td>0.671</td><td>0.533</td><td>0.532</td><td>1.142</td><td>0.782</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.187</td><td>0.267</td><td>0.189</td><td>0.280</td><td>0.209</td><td>0.308</td><td>0.193</td><td>0.292</td><td>0.203</td><td>0.287</td><td>0.192</td><td>0.274</td><td>0.255</td><td>0.339</td><td>0.435</td><td>0.507</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td><td>0.243</td><td>0.342</td><td>2.041</td><td>1.073</td></tr><tr><td>192</td><td>0.249</td><td>0.309</td><td>0.253</td><td>0.319</td><td>0.311</td><td>0.382</td><td>0.284</td><td>0.362</td><td>0.269</td><td>0.328</td><td>0.280</td><td>0.339</td><td>0.281</td><td>0.340</td><td>0.730</td><td>0.673</td><td>0.533</td><td>0.563</td><td>0.989</td><td>0.757</td><td>1.078</td><td>0.827</td><td>0.392</td><td>0.448</td><td>2.249</td><td>1.112</td></tr><tr><td>336</td><td>0.321</td><td>0.351</td><td>0.314</td><td>0.357</td><td>0.442</td><td>0.466</td><td>0.369</td><td>0.427</td><td>0.325</td><td>0.366</td><td>0.334</td><td>0.361</td><td>0.339</td><td>0.372</td><td>1.201</td><td>0.845</td><td>1.363</td><td>0.887</td><td>1.334</td><td>0.872</td><td>1.549</td><td>0.972</td><td>0.932</td><td>0.724</td><td>2.568</td><td>1.238</td></tr><tr><td>720</td><td>0.408</td><td>0.403</td><td>0.414</td><td>0.413</td><td>0.675</td><td>0.587</td><td>0.554</td><td>0.522</td><td>0.421</td><td>0.415</td><td>0.417</td><td>0.413</td><td>0.433</td><td>0.432</td><td>3.625</td><td>1.451</td><td>3.379</td><td>1.338</td><td>3.048</td><td>1.328</td><td>2.631</td><td>1.242</td><td>1.372</td><td>0.879</td><td>2.720</td><td>1.287</td></tr><tr><td></td><td>Avg</td><td>0.291</td><td>0.333</td><td>0.293</td><td>0.342</td><td>0.409</td><td>0.436</td><td>0.350</td><td>0.401</td><td>0.305</td><td>0.349</td><td>0.306</td><td>0.347</td><td>0.327</td><td>0.371</td><td>1.498</td><td>0.869</td><td>1.410</td><td>0.810</td><td>1.535</td><td>0.900</td><td>1.479</td><td>0.915</td><td>0.735</td><td>0.598</td><td>2.395</td><td>1.177</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.384</td><td>0.402</td><td>0.494</td><td>0.479</td><td>0.424</td><td>0.432</td><td>0.386</td><td>0.400</td><td>0.376</td><td>0.419</td><td>0.513</td><td>0.491</td><td>0.449</td><td>0.459</td><td>0.664</td><td>0.612</td><td>0.865</td><td>0.713</td><td>0.878</td><td>0.740</td><td>0.837</td><td>0.728</td><td>0.548</td><td>0.528</td><td>1.044</td><td>0.773</td></tr><tr><td>192</td><td>0.436</td><td>0.429</td><td>0.538</td><td>0.504</td><td>0.475</td><td>0.462</td><td>0.437</td><td>0.432</td><td>0.420</td><td>0.448</td><td>0.534</td><td>0.504</td><td>0.500</td><td>0.482</td><td>0.790</td><td>0.681</td><td>1.008</td><td>0.792</td><td>1.037</td><td>0.824</td><td>0.923</td><td>0.766</td><td>0.542</td><td>0.526</td><td>1.217</td><td>0.832</td></tr><tr><td>336</td><td>0.491</td><td>0.469</td><td>0.574</td><td>0.521</td><td>0.518</td><td>0.488</td><td>0.481</td><td>0.459</td><td>0.459</td><td>0.465</td><td>0.588</td><td>0.535</td><td>0.521</td><td>0.496</td><td>0.891</td><td>0.738</td><td>1.107</td><td>0.809</td><td>1.238</td><td>0.932</td><td>1.097</td><td>0.835</td><td>1.298</td><td>0.942</td><td>1.259</td><td>0.841</td></tr><tr><td>720</td><td>0.521</td><td>0.500</td><td>0.562</td><td>0.535</td><td>0.547</td><td>0.533</td><td>0.519</td><td>0.516</td><td>0.506</td><td>0.507</td><td>0.643</td><td>0.616</td><td>0.514</td><td>0.512</td><td>0.963</td><td>0.782</td><td>1.181</td><td>0.865</td><td>1.135</td><td>0.852</td><td>1.257</td><td>0.889</td><td>0.721</td><td>0.659</td><td>1.271</td><td>0.838</td></tr><tr><td></td><td>Avg</td><td>0.458</td><td>0.450</td><td>0.542</td><td>0.510</td><td>0.491</td><td>0.479</td><td>0.456</td><td>0.452</td><td>0.440</td><td>0.460</td><td>0.570</td><td>0.537</td><td>0.496</td><td>0.487</td><td>0.827</td><td>0.703</td><td>1.040</td><td>0.795</td><td>1.072</td><td>0.837</td><td>1.029</td><td>0.805</td><td>0.777</td><td>0.664</td><td>1.198</td><td>0.821</td></tr><tr><td rowspan="4">ETTh2</td><td>96</td><td>0.340</td><td>0.374</td><td>0.340</td><td>0.391</td><td>0.397</td><td>0.437</td><td>0.333</td><td>0.387</td><td>0.358</td><td>0.397</td><td>0.476</td><td>0.458</td><td>0.346</td><td>0.388</td><td>0.645</td><td>0.597</td><td>3.755</td><td>1.525</td><td>2.116</td><td>1.197</td><td>2.626</td><td>1.317</td><td>1.616</td><td>1.036</td><td>2.522</td><td>1.278</td></tr><tr><td>192</td><td>0.402</td><td>0.414</td><td>0.430</td><td>0.439</td><td>0.520</td><td>0.504</td><td>0.477</td><td>0.476</td><td>0.429</td><td>0.439</td><td>0.512</td><td>0.493</td><td>0.456</td><td>0.452</td><td>0.788</td><td>0.683</td><td>5.602</td><td>1.931</td><td>4.315</td><td>1.635</td><td>11.12</td><td>2.979</td><td>2.083</td><td>1.197</td><td>3.312</td><td>1.384</td></tr><tr><td>336</td><td>0.452</td><td>0.452</td><td>0.485</td><td>0.479</td><td>0.626</td><td>0.559</td><td>0.594</td><td>0.541</td><td>0.496</td><td>0.487</td><td>0.552</td><td>0.551</td><td>0.482</td><td>0.486</td><td>0.907</td><td>0.747</td><td>4.721</td><td>1.835</td><td>1.124</td><td>1.604</td><td>9.323</td><td>2.769</td><td>2.970</td><td>1.439</td><td>3.291</td><td>1.388</td></tr><tr><td>720</td><td>0.462</td><td>0.468</td><td>0.500</td><td>0.497</td><td>0.863</td><td>0.672</td><td>0.831</td><td>0.657</td><td>0.463</td><td>0.474</td><td>0.562</td><td>0.560</td><td>0.515</td><td>0.511</td><td>0.963</td><td>0.783</td><td>3.647</td><td>1.625</td><td>3.188</td><td>1.540</td><td>3.874</td><td>1.697</td><td>2.576</td><td>1.363</td><td>3.257</td><td>1.357</td></tr><tr><td></td><td>Avg</td><td>0.414</td><td>0.427</td><td>0.439</td><td>0.452</td><td>0.602</td><td>0.543</td><td>0.559</td><td>0.515</td><td>0.437</td><td>0.449</td><td>0.526</td><td>0.516</td><td>0.450</td><td>0.459</td><td>0.826</td><td>0.703</td><td>4.431</td><td>1.729</td><td>2.686</td><td>1.494</td><td>6.736</td><td>2.191</td><td>2.311</td><td>1.259</td><td>3.095</td><td>1.352</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.168</td><td>0.272</td><td>0.187</td><td>0.304</td><td>0.207</td><td>0.307</td><td>0.197</td><td>0.282</td><td>0.193</td><td>0.308</td><td>0.169</td><td>0.273</td><td>0.201</td><td>0.317</td><td>0.386</td><td>0.449</td><td>0.274</td><td>0.368</td><td>0.258</td><td>0.357</td><td>0.312</td><td>0.402</td><td>0.300</td><td>0.392</td><td>0.375</td><td>0.437</td></tr><tr><td>192</td><td>0.184</td><td>0.289</td><td>0.199</td><td>0.315</td><td>0.213</td><td>0.316</td><td>0.196</td><td>0.285</td><td>0.201</td><td>0.315</td><td>0.182</td><td>0.286</td><td>0.222</td><td>0.334</td><td>0.378</td><td>0.443</td><td>0.296</td><td>0.386</td><td>0.266</td><td>0.368</td><td>0.348</td><td>0.433</td><td>0.297</td><td>0.390</td><td>0.442</td><td>0.473</td></tr><tr><td>336</td><td>0.198</td><td>0.300</td><td>0.212</td><td>0.329</td><td>0.230</td><td>0.333</td><td>0.209</td><td>0.301</td><td>0.214</td><td>0.329</td><td>0.200</td><td>0.304</td><td>0.231</td><td>0.338</td><td>0.376</td><td>0.443</td><td>0.300</td><td>0.394</td><td>0.280</td><td>0.380</td><td>0.350</td><td>0.433</td><td>0.317</td><td>0.403</td><td>0.439</td><td>0.473</td></tr><tr><td>720</td><td>0.220</td><td>0.320</td><td>0.233</td><td>0.345</td><td>0.265</td><td>0.360</td><td>0.245</td><td>0.333</td><td>0.246</td><td>0.355</td><td>0.222</td><td>0.321</td><td>0.254</td><td>0.361</td><td>0.376</td><td>0.445</td><td>0.373</td><td>0.439</td><td>0.283</td><td>0.376</td><td>0.340</td><td>0.420</td><td>0.338</td><td>0.417</td><td>0.980</td><td>0.814</td></tr><tr><td></td><td>Avg</td><td>0.192</td><td>0.295</td><td>0.208</td><td>0.323</td><td>0.229</td><td>0.329</td><td>0.212</td><td>0.300</td><td>0.214</td><td>0.327</td><td>0.193</td><td>0.296</td><td>0.227</td><td>0.338</td><td>0.379</td><td>0.445</td><td>0.311</td><td>0.397</td><td>0.272</td><td>0.370</td><td>0.338</td><td>0.422</td><td>0.313</td><td>0.401</td><td>0.559</td><td>0.549</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.593</td><td>0.321</td><td>0.607</td><td>0.392</td><td>0.615</td><td>0.391</td><td>0.650</td><td>0.396</td><td>0.587</td><td>0.366</td><td>0.612</td><td>0.338</td><td>0.613</td><td>0.388</td><td>0.867</td><td>0.468</td><td>0.719</td><td>0.391</td><td>0.684</td><td>0.384</td><td>0.732</td><td>0.423</td><td>0.798</td><td>0.436</td><td>0.843</td><td>0.453</td></tr><tr><td>192</td><td>0.617</td><td>0.336</td><td>0.621</td><td>0.399</td><td>0.601</td><td>0.382</td><td>0.598</td><td>0.370</td><td>0.604</td><td>0.373</td><td>0.613</td><td>0.340</td><td>0.616</td><td>0.382</td><td>0.869</td><td>0.467</td><td>0.696</td><td>0.379</td><td>0.685</td><td>0.390</td><td>0.733</td><td>0.420</td><td>0.849</td><td>0.481</td><td>0.847</td><td>0.453</td></tr><tr><td>336</td><td>0.629</td><td>0.336</td><td>0.622</td><td>0.396</td><td>0.613</td><td>0.386</td><td>0.605</td><td>0.373</td><td>0.621</td><td>0.383</td><td>0.618</td><td>0.328</td><td>0.622</td><td>0.337</td><td>0.881</td><td>0.469</td><td>0.777</td><td>0.420</td><td>0.734</td><td>0.408</td><td>0.742</td><td>0.420</td><td>0.828</td><td>0.476</td><td>0.853</td><td>0.455</td></tr><tr><td>720</td><td>0.640</td><td>0.350</td><td>0.632</td><td>0.396</td><td>0.658</td><td>0.407</td><td>0.645</td><td>0.394</td><td>0.626</td><td>0.382</td><td>0.653</td><td>0.355</td><td>0.660</td><td>0.408</td><td>0.896</td><td>0.473</td><td>0.864</td><td>0.472</td><td>0.717</td><td>0.396</td><td>0.755</td><td>0.423</td><td>0.854</td><td>0.489</td><td>1.500</td><td>0.805</td></tr><tr><td></td><td>Avg</td><td>0.620</td><td>0.336</td><td>0.621</td><td>0.396</td><td>0.622</td><td>0.392</td><td>0.625</td><td>0.383</td><td>0.610</td><td>0.376</td><td>0.624</td><td>0.340</td><td>0.628</td><td>0.379</td><td>0.878</td><td>0.469</td><td>0.764</td><td>0.416</td><td>0.705</td><td>0.395</td><td>0.741</td><td>0.422</td><td>0.832</td><td>0.471</td><td>1.011</td><td>0.541</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.172</td><td>0.220</td><td>0.197</td><td>0.281</td><td>0.182</td><td>0.242</td><td>0.196</td><td>0.255</td><td>0.217</td><td>0.296</td><td>0.173</td><td>0.223</td><td>0.266</td><td>0.336</td><td>0.622</td><td>0.556</td><td>0.300</td><td>0.384</td><td>0.458</td><td>0.490</td><td>0.689</td><td>0.596</td><td>0.174</td><td>0.252</td><td>0.369</td><td>0.406</td></tr><tr><td>192</td><td>0.219</td><td>0.261</td><td>0.237</td><td>0.312</td><td>0.227</td><td>0.287</td><td>0.237</td><td>0.296</td><td>0.276</td><td>0.336</td><td>0.245</td><td>0.285</td><td>0.307</td><td>0.367</td><td>0.739</td><td>0.624</td><td>0.598</td><td>0.544</td><td>0.658</td><td>0.589</td><td>0.752</td><td>0.638</td><td>0.238</td><td>0.313</td><td>0.416</td><td>0.435</td></tr><tr><td>336</td><td>0.280</td><td>0.306</td><td>0.298</td><td>0.353</td><td>0.282</td><td>0.334</td><td>0.283</td><td>0.335</td><td>0.339</td><td>0.380</td><td>0.321</td><td>0.338</td><td>0.359</td><td>0.395</td><td>1.004</td><td>0.753</td><td>0.578</td><td>0.523</td><td>0.797</td><td>0.652</td><td>0.639</td><td>0.596</td><td>0.287</td><td>0.355</td><td>0.455</td><td>0.454</td></tr><tr><td>720</td><td>0.365</td><td>0.359</td><td>0.352</td><td>0.288</td><td>0.352</td><td>0.386</td><td>0.345</td><td>0.381</td><td>0.403</td><td>0.428</td><td>0.414</td><td>0.410</td><td>0.419</td><td>0.428</td><td>1.420</td><td>0.934</td><td>1.059</td><td>0.741</td><td>0.869</td><td>0.675</td><td>1.130</td><td>0.792</td><td>0.384</td><td>0.415</td><td>0.535</td><td>0.520</td></tr><tr><td></td><td>Avg</td><td>0.259</td><td>0.287</td><td>0.271</td><td>0.334</td><td>0.261</td><td>0.312</td><td>0.265</td><td>0.317</td><td>0.309</td><td>0.360</td><td>0.288</td><td>0.314</td><td>0.338</td><td>0.382</td><td>0.946</td><td>0.717</td><td>0.634</td><td>0.548</td><td>0.696</td><td>0.602</td><td>0.803</td><td>0.656</td><td>0.271</td><td>0.334</td><td>0.444</td><td>0.454</td></tr><tr><td rowspan="2">Exchange</td><td>96</td><td>0.107</td><td>0.234</td><td>0.085</td><td>0.204</td><td>0.116</td><td>0.262</td><td>0.088</td><td>0.218</td><td>0.148</td><td>0.278</td><td>0.111</td><td>0.237</td><td>0.197</td><td>0.323</td><td>1.748</td><td>1.105</td><td>0.847</td><td>0.752</td><td>0.968</td><td>0.812</td><td>1.065</td><td>0.829</td><td>0.395</td><td>0.474</td><td>1.453</td><td>1.049</td></tr><tr><td>192</td><td>0.226</td><td>0.344</td><td>0.182</td><td>0.303</td><td>0.215</td><td>0.359</td><td>0.176</td><td>0.315</td><td>0.271</td><td>0.380</td><td>0.219</td><td>0.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>


∗ means that there are some mismatches between our input-output setting and their papers. We adopt their official codes and onl change the length of input and output sequences for a fair comparison. 



Table 14: Full results for the short-term forecasting task in the M4 dataset. . in the Transformers indicates the name of former. Stationary means the Non-stationary Transformer.


<table><tr><td colspan="2">Models</td><td>TimesNet (Ours)</td><td>N-HiTS (2022)</td><td>N-BEATS* (2019)</td><td>ETS. (2022)</td><td>LightTS (2022)</td><td>DLinear (2023)</td><td>FED. (2022)</td><td>Stationary (2022a)</td><td>Auto. (2021)</td><td>Pyra. (2021a)</td><td>In. (2021)</td><td>LogTrans (2019)</td><td>Re. (2020)</td><td>LSTM (1997)</td><td>TCN (2019)</td><td>LSSL (2022)</td></tr><tr><td rowspan="3">Yearly</td><td>SMAPE</td><td>13.387</td><td>13.418</td><td>13.436</td><td>18.009</td><td>14.247</td><td>16.965</td><td>13.728</td><td>13.717</td><td>13.974</td><td>15.530</td><td>14.727</td><td>17.107</td><td>16.169</td><td>176.040</td><td>14.920</td><td>61.675</td></tr><tr><td>MASE</td><td>2.996</td><td>3.045</td><td>3.043</td><td>4.487</td><td>3.109</td><td>4.283</td><td>3.048</td><td>3.078</td><td>3.134</td><td>3.711</td><td>3.418</td><td>4.177</td><td>3.800</td><td>31.033</td><td>3.364</td><td>19.953</td></tr><tr><td>OWA</td><td>0.786</td><td>0.793</td><td>0.794</td><td>1.115</td><td>0.827</td><td>1.058</td><td>0.803</td><td>0.807</td><td>0.822</td><td>0.942</td><td>0.881</td><td>1.049</td><td>0.973</td><td>9.290</td><td>0.880</td><td>4.397</td></tr><tr><td rowspan="3">Quarterly</td><td>SMAPE</td><td>10.100</td><td>10.202</td><td>10.124</td><td>13.376</td><td>11.364</td><td>12.145</td><td>10.792</td><td>10.958</td><td>11.338</td><td>15.449</td><td>11.360</td><td>13.207</td><td>13.313</td><td>172.808</td><td>11.122</td><td>65.999</td></tr><tr><td>MASE</td><td>1.182</td><td>1.194</td><td>1.169</td><td>1.906</td><td>1.328</td><td>1.520</td><td>1.283</td><td>1.325</td><td>1.365</td><td>2.350</td><td>1.401</td><td>1.827</td><td>1.775</td><td>19.753</td><td>1.360</td><td>17.662</td></tr><tr><td>OWA</td><td>0.890</td><td>0.899</td><td>0.886</td><td>1.302</td><td>1.000</td><td>1.106</td><td>0.958</td><td>0.981</td><td>1.012</td><td>1.558</td><td>1.027</td><td>1.266</td><td>1.252</td><td>15.049</td><td>1.001</td><td>9.436</td></tr><tr><td rowspan="3">Monthly</td><td>SMAPE</td><td>12.670</td><td>12.791</td><td>12.677</td><td>14.588</td><td>14.014</td><td>13.514</td><td>14.260</td><td>13.917</td><td>13.958</td><td>17.642</td><td>14.062</td><td>16.149</td><td>20.128</td><td>143.237</td><td>15.626</td><td>64.664</td></tr><tr><td>MASE</td><td>0.933</td><td>0.969</td><td>0.937</td><td>1.368</td><td>1.053</td><td>1.037</td><td>1.102</td><td>1.097</td><td>1.103</td><td>1.913</td><td>1.141</td><td>1.660</td><td>2.614</td><td>16.551</td><td>1.274</td><td>16.245</td></tr><tr><td>OWA</td><td>0.878</td><td>0.899</td><td>0.880</td><td>1.149</td><td>0.981</td><td>0.956</td><td>1.012</td><td>0.998</td><td>1.002</td><td>1.511</td><td>1.024</td><td>1.340</td><td>1.927</td><td>12.747</td><td>1.141</td><td>9.879</td></tr><tr><td rowspan="3">Others</td><td>SMAPE</td><td>4.891</td><td>5.061</td><td>4.925</td><td>7.267</td><td>15.880</td><td>6.709</td><td>4.954</td><td>6.302</td><td>5.485</td><td>24.786</td><td>24.460</td><td>23.236</td><td>32.491</td><td>186.282</td><td>7.186</td><td>121.844</td></tr><tr><td>MASE</td><td>3.302</td><td>3.216</td><td>3.391</td><td>5.240</td><td>11.434</td><td>4.953</td><td>3.264</td><td>4.064</td><td>3.865</td><td>18.581</td><td>20.960</td><td>16.288</td><td>33.355</td><td>119.294</td><td>4.677</td><td>91.650</td></tr><tr><td>OWA</td><td>1.035</td><td>1.040</td><td>1.053</td><td>1.591</td><td>3.474</td><td>1.487</td><td>1.036</td><td>1.304</td><td>1.187</td><td>5.538</td><td>5.879</td><td>5.013</td><td>8.679</td><td>38.411</td><td>1.494</td><td>27.273</td></tr><tr><td rowspan="3">Weighted Average</td><td>SMAPE</td><td>11.829</td><td>11.927</td><td>11.851</td><td>14.718</td><td>13.525</td><td>13.639</td><td>12.840</td><td>12.780</td><td>12.909</td><td>16.987</td><td>14.086</td><td>16.018</td><td>18.200</td><td>160.031</td><td>13.961</td><td>67.156</td></tr><tr><td>MASE</td><td>1.585</td><td>1.613</td><td>1.599</td><td>2.408</td><td>2.111</td><td>2.095</td><td>1.701</td><td>1.756</td><td>1.771</td><td>3.265</td><td>2.718</td><td>3.010</td><td>4.223</td><td>25.788</td><td>1.945</td><td>21.208</td></tr><tr><td>OWA</td><td>0.851</td><td>0.861</td><td>0.855</td><td>1.172</td><td>1.051</td><td>1.051</td><td>0.918</td><td>0.930</td><td>0.939</td><td>1.480</td><td>1.230</td><td>1.378</td><td>1.775</td><td>12.642</td><td>1.023</td><td>8.021</td></tr></table>


∗ The original paper of N-BEATS (Oreshkin et al., 2019) adopts a special ensemble method to promote the performance. For fair comparisons, we remove the ensemble and only compare the pure forecasting models.



Table 15: Full results for the anomaly detection task. The P, R and F1 represent the precision, recall and F1-score (%) respectively. F1-score is the harmonic mean of precision and recall. A higher value of P, R and F1 indicates a better performance.


<table><tr><td colspan="2">Datasets</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td><td rowspan="2">Avg F1 (%)</td></tr><tr><td colspan="2">Metrics</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>LSTM</td><td>(1997)</td><td>78.52</td><td>65.47</td><td>71.41</td><td>78.04</td><td>86.22</td><td>81.93</td><td>91.06</td><td>57.49</td><td>70.48</td><td>78.06</td><td>91.72</td><td>84.34</td><td>69.24</td><td>99.53</td><td>81.67</td><td>77.97</td></tr><tr><td>Transformer</td><td>(2017)</td><td>83.58</td><td>76.13</td><td>79.56</td><td>71.57</td><td>87.37</td><td>78.68</td><td>89.37</td><td>57.12</td><td>69.70</td><td>68.84</td><td>96.53</td><td>80.37</td><td>62.75</td><td>96.56</td><td>76.07</td><td>76.88</td></tr><tr><td>LogTrans</td><td>(2019)</td><td>83.46</td><td>70.13</td><td>76.21</td><td>73.05</td><td>87.37</td><td>79.57</td><td>89.15</td><td>57.59</td><td>69.97</td><td>68.67</td><td>97.32</td><td>80.52</td><td>63.06</td><td>98.00</td><td>76.74</td><td>76.60</td></tr><tr><td>TCN</td><td>(2019)</td><td>84.06</td><td>79.07</td><td>81.49</td><td>75.11</td><td>82.44</td><td>78.60</td><td>86.90</td><td>59.23</td><td>70.45</td><td>76.59</td><td>95.71</td><td>85.09</td><td>54.59</td><td>99.77</td><td>70.57</td><td>77.24</td></tr><tr><td>Reformer</td><td>(2020)</td><td>82.58</td><td>69.24</td><td>75.32</td><td>85.51</td><td>83.31</td><td>84.40</td><td>90.91</td><td>57.44</td><td>70.40</td><td>72.50</td><td>96.53</td><td>82.80</td><td>59.93</td><td>95.38</td><td>73.61</td><td>77.31</td></tr><tr><td>Informer</td><td>(2021)</td><td>86.60</td><td>77.23</td><td>81.65</td><td>81.77</td><td>86.48</td><td>84.06</td><td>90.11</td><td>57.13</td><td>69.92</td><td>70.29</td><td>96.75</td><td>81.43</td><td>64.27</td><td>96.33</td><td>77.10</td><td>78.83</td></tr><tr><td>Anomaly*</td><td>(2021)</td><td>88.91</td><td>82.23</td><td>85.49</td><td>79.61</td><td>87.37</td><td>83.31</td><td>91.85</td><td>58.11</td><td>71.18</td><td>72.51</td><td>97.32</td><td>83.10</td><td>68.35</td><td>94.72</td><td>79.40</td><td>80.50</td></tr><tr><td>Pyraformer</td><td>(2021a)</td><td>85.61</td><td>80.61</td><td>83.04</td><td>83.81</td><td>85.93</td><td>84.86</td><td>92.54</td><td>57.71</td><td>71.09</td><td>87.92</td><td>96.00</td><td>91.78</td><td>71.67</td><td>96.02</td><td>82.08</td><td>82.57</td></tr><tr><td>Autoformer</td><td>(2021)</td><td>88.06</td><td>82.35</td><td>85.11</td><td>77.27</td><td>80.92</td><td>79.05</td><td>90.40</td><td>58.62</td><td>71.12</td><td>89.85</td><td>95.81</td><td>92.74</td><td>99.08</td><td>88.15</td><td>93.29</td><td>84.26</td></tr><tr><td>LSSL</td><td>(2022)</td><td>78.51</td><td>65.32</td><td>71.31</td><td>77.55</td><td>88.18</td><td>82.53</td><td>89.43</td><td>53.43</td><td>66.90</td><td>79.05</td><td>93.72</td><td>85.76</td><td>66.02</td><td>92.93</td><td>77.20</td><td>76.74</td></tr><tr><td>Stationary</td><td>(2022a)</td><td>88.33</td><td>81.21</td><td>84.62</td><td>68.55</td><td>89.14</td><td>77.50</td><td>89.37</td><td>59.02</td><td>71.09</td><td>68.03</td><td>96.75</td><td>79.88</td><td>97.82</td><td>96.76</td><td>97.29</td><td>82.08</td></tr><tr><td>DLinear</td><td>(2023)</td><td>83.62</td><td>71.52</td><td>77.10</td><td>84.34</td><td>85.42</td><td>84.88</td><td>92.32</td><td>55.41</td><td>69.26</td><td>80.91</td><td>95.30</td><td>87.52</td><td>98.28</td><td>89.26</td><td>93.55</td><td>82.46</td></tr><tr><td>ETSformer</td><td>(2022)</td><td>87.44</td><td>79.23</td><td>83.13</td><td>85.13</td><td>84.93</td><td>85.03</td><td>92.25</td><td>55.75</td><td>69.50</td><td>90.02</td><td>80.36</td><td>84.91</td><td>99.31</td><td>85.28</td><td>91.76</td><td>82.87</td></tr><tr><td>LightTS</td><td>(2022)</td><td>87.10</td><td>78.42</td><td>82.53</td><td>82.40</td><td>75.78</td><td>78.95</td><td>92.58</td><td>55.27</td><td>69.21</td><td>91.98</td><td>94.72</td><td>93.33</td><td>98.37</td><td>95.97</td><td>97.15</td><td>84.23</td></tr><tr><td>FEDformer</td><td>(2022)</td><td>87.95</td><td>82.39</td><td>85.08</td><td>77.14</td><td>80.07</td><td>78.57</td><td>90.47</td><td>58.10</td><td>70.76</td><td>90.17</td><td>96.42</td><td>93.19</td><td>97.31</td><td>97.16</td><td>97.23</td><td>84.97</td></tr><tr><td>TimesNet</td><td>(Inception)</td><td>87.76</td><td>82.63</td><td>85.12</td><td>82.97</td><td>85.42</td><td>84.18</td><td>91.50</td><td>57.80</td><td>70.85</td><td>88.31</td><td>96.24</td><td>92.10</td><td>98.22</td><td>92.21</td><td>95.21</td><td>85.49</td></tr><tr><td>TimesNet</td><td>(ResNeXt)</td><td>88.66</td><td>83.14</td><td>85.81</td><td>83.92</td><td>86.42</td><td>85.15</td><td>92.52</td><td>58.29</td><td>71.52</td><td>86.76</td><td>97.32</td><td>91.74</td><td>98.19</td><td>96.76</td><td>97.47</td><td>86.34</td></tr></table>


∗ The original paper of Anomaly Transformer (Xu et al., 2021) adopts the temporal association and reconstruction error as a joint anomaly criterion. For fair comparisons, we only use reconstruction error here.



Table 16: Full results for the imputation task. We randomly mask 12.5%, 25%, 37.5% and 50% time points to compare the model performance under different missing degrees. . in the Transformers indicates the name of former.


<table><tr><td colspan="2">Models</td><td colspan="2">TimesNet(Ours)</td><td colspan="2">ETS.(2022)</td><td colspan="2">LightTS*(2022)</td><td colspan="2">DLinear*(2023)</td><td colspan="2">FED.(2022)</td><td colspan="2">Stationary(2022a)</td><td colspan="2">Auto.(2021)</td><td colspan="2">Pyra.(2021a)</td><td colspan="2">In.(2021)</td><td colspan="2">LogTrans(2019)</td><td colspan="2">Re.(2020)</td><td colspan="2">LSTM(1997)</td><td colspan="2">TCN(2019)</td><td colspan="2">LSSL(2022)</td></tr><tr><td colspan="2">Mask Ratio</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTm1</td><td>12.5%</td><td>0.019</td><td>0.092</td><td>0.067</td><td>0.188</td><td>0.075</td><td>0.180</td><td>0.058</td><td>0.162</td><td>0.035</td><td>0.135</td><td>0.026</td><td>0.107</td><td>0.034</td><td>0.124</td><td>0.670</td><td>0.541</td><td>0.047</td><td>0.155</td><td>0.041</td><td>0.141</td><td>0.032</td><td>0.126</td><td>0.974</td><td>0.780</td><td>0.510</td><td>0.493</td><td>0.101</td><td>0.231</td></tr><tr><td>25%</td><td>0.023</td><td>0.101</td><td>0.096</td><td>0.229</td><td>0.093</td><td>0.206</td><td>0.080</td><td>0.193</td><td>0.052</td><td>0.166</td><td>0.032</td><td>0.119</td><td>0.046</td><td>0.144</td><td>0.689</td><td>0.553</td><td>0.063</td><td>0.180</td><td>0.044</td><td>0.144</td><td>0.042</td><td>0.146</td><td>1.032</td><td>0.807</td><td>0.518</td><td>0.500</td><td>0.106</td><td>0.235</td></tr><tr><td>37.5%</td><td>0.029</td><td>0.111</td><td>0.133</td><td>0.271</td><td>0.113</td><td>0.231</td><td>0.103</td><td>0.219</td><td>0.069</td><td>0.191</td><td>0.039</td><td>0.131</td><td>0.057</td><td>0.161</td><td>0.737</td><td>0.581</td><td>0.079</td><td>0.200</td><td>0.052</td><td>0.158</td><td>0.063</td><td>0.182</td><td>0.999</td><td>0.792</td><td>0.516</td><td>0.499</td><td>0.116</td><td>0.246</td></tr><tr><td>50%</td><td>0.036</td><td>0.124</td><td>0.186</td><td>0.323</td><td>0.134</td><td>0.255</td><td>0.132</td><td>0.248</td><td>0.089</td><td>0.218</td><td>0.047</td><td>0.145</td><td>0.067</td><td>0.174</td><td>0.770</td><td>0.605</td><td>0.093</td><td>0.218</td><td>0.063</td><td>0.173</td><td>0.082</td><td>0.208</td><td>0.952</td><td>0.763</td><td>0.519</td><td>0.496</td><td>0.129</td><td>0.260</td></tr><tr><td></td><td>Avg</td><td>0.027</td><td>0.107</td><td>0.120</td><td>0.253</td><td>0.104</td><td>0.218</td><td>0.093</td><td>0.206</td><td>0.062</td><td>0.177</td><td>0.036</td><td>0.126</td><td>0.051</td><td>0.150</td><td>0.717</td><td>0.570</td><td>0.071</td><td>0.188</td><td>0.050</td><td>0.154</td><td>0.055</td><td>0.166</td><td>0.989</td><td>0.786</td><td>0.516</td><td>0.497</td><td>0.113</td><td>0.254</td></tr><tr><td rowspan="4">ETTm2</td><td>12.5%</td><td>0.018</td><td>0.080</td><td>0.108</td><td>0.239</td><td>0.034</td><td>0.127</td><td>0.062</td><td>0.166</td><td>0.056</td><td>0.159</td><td>0.021</td><td>0.088</td><td>0.023</td><td>0.092</td><td>0.394</td><td>0.470</td><td>0.133</td><td>0.270</td><td>0.103</td><td>0.229</td><td>0.108</td><td>0.228</td><td>1.013</td><td>0.805</td><td>0.307</td><td>0.441</td><td>0.150</td><td>0.298</td></tr><tr><td>25%</td><td>0.020</td><td>0.085</td><td>0.164</td><td>0.294</td><td>0.042</td><td>0.143</td><td>0.085</td><td>0.196</td><td>0.080</td><td>0.195</td><td>0.024</td><td>0.096</td><td>0.026</td><td>0.101</td><td>0.421</td><td>0.482</td><td>0.135</td><td>0.272</td><td>0.120</td><td>0.248</td><td>0.136</td><td>0.262</td><td>1.039</td><td>0.814</td><td>0.263</td><td>0.402</td><td>0.159</td><td>0.306</td></tr><tr><td>37.5%</td><td>0.023</td><td>0.091</td><td>0.237</td><td>0.356</td><td>0.051</td><td>0.159</td><td>0.106</td><td>0.222</td><td>0.110</td><td>0.231</td><td>0.027</td><td>0.103</td><td>0.030</td><td>0.108</td><td>0.478</td><td>0.521</td><td>0.155</td><td>0.293</td><td>0.138</td><td>0.260</td><td>0.175</td><td>0.300</td><td>0.917</td><td>0.744</td><td>0.250</td><td>0.396</td><td>0.180</td><td>0.321</td></tr><tr><td>50%</td><td>0.026</td><td>0.098</td><td>0.323</td><td>0.421</td><td>0.059</td><td>0.174</td><td>0.131</td><td>0.247</td><td>0.156</td><td>0.276</td><td>0.030</td><td>0.108</td><td>0.035</td><td>0.119</td><td>0.568</td><td>0.560</td><td>0.200</td><td>0.333</td><td>0.117</td><td>0.247</td><td>0.211</td><td>0.329</td><td>1.140</td><td>0.835</td><td>0.246</td><td>0.389</td><td>0.210</td><td>0.353</td></tr><tr><td></td><td>Avg</td><td>0.022</td><td>0.088</td><td>0.208</td><td>0.327</td><td>0.046</td><td>0.151</td><td>0.096</td><td>0.208</td><td>0.101</td><td>0.215</td><td>0.026</td><td>0.099</td><td>0.029</td><td>0.105</td><td>0.465</td><td>0.508</td><td>0.156</td><td>0.292</td><td>0.119</td><td>0.246</td><td>0.157</td><td>0.280</td><td>1.027</td><td>0.800</td><td>0.266</td><td>0.407</td><td>0.175</td><td>0.324</td></tr><tr><td rowspan="4">ETTh1</td><td>12.5%</td><td>0.057</td><td>0.159</td><td>0.126</td><td>0.263</td><td>0.240</td><td>0.345</td><td>0.151</td><td>0.267</td><td>0.070</td><td>0.190</td><td>0.060</td><td>0.165</td><td>0.074</td><td>0.182</td><td>0.857</td><td>0.609</td><td>0.114</td><td>0.234</td><td>0.229</td><td>0.330</td><td>0.074</td><td>0.194</td><td>1.265</td><td>0.896</td><td>0.599</td><td>0.554</td><td>0.422</td><td>0.461</td></tr><tr><td>25%</td><td>0.069</td><td>0.178</td><td>0.169</td><td>0.304</td><td>0.265</td><td>0.364</td><td>0.180</td><td>0.292</td><td>0.106</td><td>0.236</td><td>0.080</td><td>0.189</td><td>0.090</td><td>0.203</td><td>0.829</td><td>0.672</td><td>0.140</td><td>0.262</td><td>0.207</td><td>0.323</td><td>0.102</td><td>0.227</td><td>1.262</td><td>0.883</td><td>0.610</td><td>0.567</td><td>0.412</td><td>0.456</td></tr><tr><td>37.5%</td><td>0.084</td><td>0.196</td><td>0.220</td><td>0.347</td><td>0.296</td><td>0.382</td><td>0.215</td><td>0.318</td><td>0.124</td><td>0.258</td><td>0.102</td><td>0.212</td><td>0.109</td><td>0.222</td><td>0.830</td><td>0.675</td><td>0.174</td><td>0.293</td><td>0.210</td><td>0.328</td><td>0.135</td><td>0.261</td><td>1.200</td><td>0.867</td><td>0.628</td><td>0.577</td><td>0.421</td><td>0.461</td></tr><tr><td>50%</td><td>0.102</td><td>0.215</td><td>0.293</td><td>0.402</td><td>0.334</td><td>0.404</td><td>0.257</td><td>0.347</td><td>0.165</td><td>0.299</td><td>0.133</td><td>0.240</td><td>0.137</td><td>0.248</td><td>0.854</td><td>0.691</td><td>0.215</td><td>0.325</td><td>0.230</td><td>0.348</td><td>0.179</td><td>0.298</td><td>1.174</td><td>0.849</td><td>0.648</td><td>0.587</td><td>0.443</td><td>0.473</td></tr><tr><td></td><td>Avg</td><td>0.078</td><td>0.187</td><td>0.202</td><td>0.329</td><td>0.284</td><td>0.373</td><td>0.201</td><td>0.306</td><td>0.117</td><td>0.246</td><td>0.094</td><td>0.201</td><td>0.103</td><td>0.214</td><td>0.842</td><td>0.682</td><td>0.161</td><td>0.279</td><td>0.219</td><td>0.332</td><td>0.122</td><td>0.245</td><td>1.225</td><td>0.873</td><td>0.621</td><td>0.571</td><td>0.424</td><td>0.481</td></tr><tr><td rowspan="4">ETTh2</td><td>12.5%</td><td>0.040</td><td>0.130</td><td>0.187</td><td>0.319</td><td>0.101</td><td>0.231</td><td>0.100</td><td>0.216</td><td>0.095</td><td>0.212</td><td>0.042</td><td>0.133</td><td>0.044</td><td>0.138</td><td>0.976</td><td>0.754</td><td>0.305</td><td>0.431</td><td>0.173</td><td>0.308</td><td>0.163</td><td>0.289</td><td>2.060</td><td>1.120</td><td>0.410</td><td>0.494</td><td>0.521</td><td>0.555</td></tr><tr><td>25%</td><td>0.046</td><td>0.141</td><td>0.279</td><td>0.390</td><td>0.115</td><td>0.246</td><td>0.127</td><td>0.247</td><td>0.137</td><td>0.258</td><td>0.049</td><td>0.147</td><td>0.050</td><td>0.149</td><td>1.037</td><td>0.774</td><td>0.322</td><td>0.444</td><td>0.175</td><td>0.310</td><td>0.206</td><td>0.331</td><td>2.007</td><td>1.105</td><td>0.419</td><td>0.490</td><td>0.487</td><td>0.535</td></tr><tr><td>37.5%</td><td>0.052</td><td>0.151</td><td>0.400</td><td>0.465</td><td>0.126</td><td>0.257</td><td>0.158</td><td>0.276</td><td>0.187</td><td>0.304</td><td>0.056</td><td>0.158</td><td>0.060</td><td>0.163</td><td>1.107</td><td>0.800</td><td>0.353</td><td>0.462</td><td>0.185</td><td>0.315</td><td>0.252</td><td>0.370</td><td>2.033</td><td>1.111</td><td>0.429</td><td>0.498</td><td>0.487</td><td>0.529</td></tr><tr><td>50%</td><td>0.060</td><td>0.162</td><td>0.602</td><td>0.572</td><td>0.136</td><td>0.268</td><td>0.183</td><td>0.299</td><td>0.232</td><td>0.341</td><td>0.065</td><td>0.170</td><td>0.068</td><td>0.173</td><td>1.193</td><td>0.838</td><td>0.369</td><td>0.472</td><td>0.212</td><td>0.339</td><td>0.316</td><td>0.419</td><td>2.054</td><td>1.119</td><td>0.467</td><td>0.529</td><td>0.484</td><td>0.523</td></tr><tr><td></td><td>Avg</td><td>0.049</td><td>0.146</td><td>0.367</td><td>0.436</td><td>0.119</td><td>0.250</td><td>0.142</td><td>0.259</td><td>0.163</td><td>0.279</td><td>0.053</td><td>0.152</td><td>0.055</td><td>0.156</td><td>1.079</td><td>0.792</td><td>0.337</td><td>0.452</td><td>0.186</td><td>0.318</td><td>0.234</td><td>0.352</td><td>2.039</td><td>1.114</td><td>0.431</td><td>0.503</td><td>0.495</td><td>0.475</td></tr><tr><td rowspan="4">Electricity</td><td>12.5%</td><td>0.085</td><td>0.202</td><td>0.196</td><td>0.321</td><td>0.102</td><td>0.229</td><td>0.092</td><td>0.214</td><td>0.107</td><td>0.237</td><td>0.093</td><td>0.210</td><td>0.089</td><td>0.210</td><td>0.297</td><td>0.383</td><td>0.218</td><td>0.326</td><td>0.164</td><td>0.296</td><td>0.190</td><td>0.308</td><td>0.277</td><td>0.366</td><td>0.621</td><td>0.620</td><td>0.217</td><td>0.341</td></tr><tr><td>25%</td><td>0.089</td><td>0.206</td><td>0.207</td><td>0.332</td><td>0.121</td><td>0.252</td><td>0.118</td><td>0.247</td><td>0.120</td><td>0.251</td><td>0.097</td><td>0.214</td><td>0.096</td><td>0.220</td><td>0.294</td><td>0.380</td><td>0.219</td><td>0.326</td><td>0.169</td><td>0.299</td><td>0.197</td><td>0.312</td><td>0.281</td><td>0.369</td><td>0.559</td><td>0.585</td><td>0.219</td><td>0.341</td></tr><tr><td>37.5%</td><td>0.094</td><td>0.213</td><td>0.219</td><td>0.344</td><td>0.141</td><td>0.273</td><td>0.144</td><td>0.276</td><td>0.136</td><td>0.266</td><td>0.102</td><td>0.220</td><td>0.104</td><td>0.229</td><td>0.296</td><td>0.381</td><td>0.222</td><td>0.328</td><td>0.178</td><td>0.305</td><td>0.203</td><td>0.315</td><td>0.275</td><td>0.364</td><td>0.567</td><td>0.588</td><td>0.223</td><td>0.343</td></tr><tr><td>50%</td><td>0.100</td><td>0.221</td><td>0.235</td><td>0.357</td><td>0.160</td><td>0.293</td><td>0.175</td><td>0.305</td><td>0.158</td><td>0.284</td><td>0.108</td><td>0.228</td><td>0.113</td><td>0.239</td><td>0.299</td><td>0.383</td><td>0.228</td><td>0.331</td><td>0.187</td><td>0.312</td><td>0.210</td><td>0.319</td><td>0.273</td><td>0.361</td><td>0.581</td><td>0.597</td><td>0.229</td><td>0.347</td></tr><tr><td></td><td>Avg</td><td>0.092</td><td>0.210</td><td>0.214</td><td>0.339</td><td>0.131</td><td>0.262</td><td>0.132</td><td>0.260</td><td>0.130</td><td>0.259</td><td>0.100</td><td>0.218</td><td>0.101</td><td>0.225</td><td>0.297</td><td>0.382</td><td>0.222</td><td>0.328</td><td>0.175</td><td>0.303</td><td>0.200</td><td>0.313</td><td>0.277</td><td>0.365</td><td>0.582</td><td>0.597</td><td>0.222</td><td>0.293</td></tr><tr><td rowspan="4">Weather</td><td>12.5%</td><td>0.025</td><td>0.045</td><td>0.057</td><td>0.141</td><td>0.047</td><td>0.101</td><td>0.039</td><td>0.084</td><td>0.041</td><td>0.107</td><td>0.027</td><td>0.051</td><td>0.026</td><td>0.047</td><td>0.140</td><td>0.220</td><td>0.037</td><td>0.093</td><td>0.037</td><td>0.072</td><td>0.031</td><td>0.076</td><td>0.296</td><td>0.379</td><td>0.176</td><td>0.287</td><td>0.036</td><td>0.095</td></tr><tr><td>25%</td><td>0.029</td><td>0.052</td><td>0.065</td><td>0.155</td><td>0.052</td><td>0.111</td><td>0.048</td><td>0.103</td><td>0.064</td><td>0.163</td><td>0.029</td><td>0.056</td><td>0.030</td><td>0.054</td><td>0.147</td><td>0.229</td><td>0.042</td><td>0.100</td><td>0.038</td><td>0.074</td><td>0.035</td><td>0.082</td><td>0.327</td><td>0.409</td><td>0.187</td><td>0.293</td><td>0.042</td><td>0.104</td></tr><tr><td>37.5%</td><td>0.031</td><td>0.057</td><td>0.081</td><td>0.180</td><td>0.058</td><td>0.121</td><td>0.057</td><td>0.117</td><td>0.107</td><td>0.229</td><td>0.033</td><td>0.062</td><td>0.032</td><td>0.060</td><td>0.156</td><td>0.240</td><td>0.049</td><td>0.111</td><td>0.039</td><td>0.078</td><td>0.040</td><td>0.091</td><td>0.406</td><td>0.463</td><td>0.172</td><td>0.281</td><td>0.047</td><td>0.112</td></tr><tr><td>50%</td><td>0.034</td><td>0.062</td><td>0.102</td><td>0.207</td><td>0.065</td><td>0.133</td><td>0.066</td><td>0.134</td><td>0.183</td><td>0.312</td><td>0.037</td><td>0.068</td><td>0.037</td><td>0.067</td><td>0.164</td><td>0.249</td><td>0.053</td><td>0.114</td><td>0.042</td><td>0.082</td><td>0.046</td><td>0.099</td><td>0.431</td><td>0.483</td><td>0.195</td><td>0.303</td><td>0.054</td><td>0.123</td></tr><tr><td></td><td>Avg</td><td>0.030</td><td>0.054</td><td>0.076</td><td>0.171</td><td>0.055</td><td>0.117</td><td>0.052</td><td>0.110</td><td>0.099</td><td>0.203</td><td>0.032</td><td>0.059</td><td>0.031</td><td>0.057</td><td>0.152</td><td>0.235</td><td>0.045</td><td>0.104</td><td>0.039</td><td>0.076</td><td>0.038</td><td>0.087</td><td>0.365</td><td>0.434</td><td>0.183</td><td>0.291</td><td>0.045</td><td>0.108</td></tr></table>


Table 17: Full results for the classification task. . in the Transformers indicates the name of former. We report the classification accuracy (%) as the result. The standard deviation is within 0.1%.


<table><tr><td rowspan="2">Datasets / Models</td><td colspan="3">Classical methods</td><td colspan="3">RNN</td><td colspan="3">TCN</td><td colspan="7">Transformers</td><td colspan="2">MLP</td><td rowspan="2">TimesNet(Ours)</td></tr><tr><td>DTWXGBoost(1994)</td><td>Rocket(2016)</td><td>LSTM(2020)</td><td>LSTNet(1997)</td><td>LSSL(2018)</td><td>LSSL(2022)</td><td>TCN(2019)</td><td>Trans.(2017)</td><td>Re.(2020)</td><td>In.(2021)</td><td>Pyra.(2021a)</td><td>Auto.(2021)</td><td>Station.(2022a)</td><td>FED.(2022)</td><td>ETS.(2022)</td><td>Flow.(2022)</td><td>DLinear(2023)</td><td>LightTS.(2022)</td></tr><tr><td>EthanolConcentration</td><td>32.3</td><td>43.7</td><td>45.2</td><td>32.3</td><td>39.9</td><td>31.1</td><td>28.9</td><td>32.7</td><td>31.9</td><td>31.6</td><td>30.8</td><td>31.6</td><td>32.7</td><td>31.2</td><td>28.1</td><td>33.8</td><td>32.6</td><td>29.7</td><td>35.7</td></tr><tr><td>FaceDetection</td><td>52.9</td><td>63.3</td><td>64.7</td><td>57.7</td><td>65.7</td><td>66.7</td><td>52.8</td><td>67.3</td><td>68.6</td><td>67.0</td><td>65.7</td><td>68.4</td><td>68.0</td><td>66.0</td><td>66.3</td><td>67.6</td><td>68.0</td><td>67.5</td><td>68.6</td></tr><tr><td>Handwriting</td><td>28.6</td><td>15.8</td><td>58.8</td><td>15.2</td><td>25.8</td><td>24.6</td><td>53.3</td><td>32.0</td><td>27.4</td><td>32.8</td><td>29.4</td><td>36.7</td><td>31.6</td><td>28.0</td><td>32.5</td><td>33.8</td><td>27.0</td><td>26.1</td><td>32.1</td></tr><tr><td>Heartbeat</td><td>71.7</td><td>73.2</td><td>75.6</td><td>72.2</td><td>77.1</td><td>72.7</td><td>75.6</td><td>76.1</td><td>77.1</td><td>80.5</td><td>75.6</td><td>74.6</td><td>73.7</td><td>73.7</td><td>71.2</td><td>77.6</td><td>75.1</td><td>75.1</td><td>78.0</td></tr><tr><td>JapaneseVowels</td><td>94.9</td><td>86.5</td><td>96.2</td><td>79.7</td><td>98.1</td><td>98.4</td><td>98.9</td><td>98.7</td><td>97.8</td><td>98.9</td><td>98.4</td><td>96.2</td><td>99.2</td><td>98.4</td><td>95.9</td><td>98.9</td><td>96.2</td><td>96.2</td><td>98.4</td></tr><tr><td>PEMS-SF</td><td>71.1</td><td>98.3</td><td>75.1</td><td>39.9</td><td>86.7</td><td>86.1</td><td>68.8</td><td>82.1</td><td>82.7</td><td>81.5</td><td>83.2</td><td>82.7</td><td>87.3</td><td>80.9</td><td>86.0</td><td>83.8</td><td>75.1</td><td>88.4</td><td>89.6</td></tr><tr><td>SelfRegulationSCP1</td><td>77.7</td><td>84.6</td><td>90.8</td><td>68.9</td><td>84.0</td><td>90.8</td><td>84.6</td><td>92.2</td><td>90.4</td><td>90.1</td><td>88.1</td><td>84.0</td><td>89.4</td><td>88.7</td><td>89.6</td><td>92.5</td><td>87.3</td><td>89.8</td><td>91.8</td></tr><tr><td>SelfRegulationSCP2</td><td>53.9</td><td>48.9</td><td>53.3</td><td>46.6</td><td>52.8</td><td>52.2</td><td>55.6</td><td>53.9</td><td>56.7</td><td>53.3</td><td>53.3</td><td>50.6</td><td>57.2</td><td>54.4</td><td>55.0</td><td>56.1</td><td>50.5</td><td>51.1</td><td>57.2</td></tr><tr><td>SpokenArabicDigits</td><td>96.3</td><td>69.6</td><td>71.2</td><td>31.9</td><td>100.0</td><td>100.0</td><td>95.6</td><td>98.4</td><td>97.0</td><td>100.0</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>98.8</td><td>81.4</td><td>100.0</td><td>99.0</td></tr><tr><td>UWaveGestureLibrary</td><td>90.3</td><td>75.9</td><td>94.4</td><td>41.2</td><td>87.8</td><td>85.9</td><td>88.4</td><td>85.6</td><td>85.6</td><td>85.6</td><td>83.4</td><td>85.9</td><td>87.5</td><td>85.3</td><td>85.0</td><td>86.6</td><td>82.1</td><td>80.3</td><td>85.3</td></tr><tr><td>Average Accuracy</td><td>67.0</td><td>66.0</td><td>72.5</td><td>48.6</td><td>71.8</td><td>70.9</td><td>70.3</td><td>71.9</td><td>71.5</td><td>72.1</td><td>70.8</td><td>71.1</td><td>72.7</td><td>70.7</td><td>71.0</td><td>73.0</td><td>67.5</td><td>70.4</td><td>73.6</td></tr></table>