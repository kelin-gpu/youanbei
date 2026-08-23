# TIME-MOE: BILLION-SCALE TIME SERIES FOUN-DATION MODELS WITH MIXTURE OF EXPERTS

Xiaoming Shi<sup>1∗♠</sup>, Shiyu Wang<sup>∗♠</sup>, Yuqi Nie<sup>2∗</sup>, Dianqi Li, Zhou Ye, Qingsong Wen<sup>3†</sup>, Ming Jin<sup>4†♠</sup> 

<sup>1</sup>Xiaohongshu Inc <sup>2</sup>Princeton University <sup>3</sup>Squirrel Ai Learning <sup>4</sup>Griffith University 

sxm728@hotmail.com, kwuking@gmail.com, ynie@princeton.edu 

{dianqili77, yezhou199032, qingsongedu, mingjinedu}@gmail.com 

## ABSTRACT

Deep learning for time series forecasting has seen significant advancements over the past decades. However, despite the success of large-scale pre-training in language and vision domains, pre-trained time series models remain limited in scale and operate at a high cost, hindering the development of larger capable forecasting models in real-world applications. In response, we introduce TIME-MOE, a scalable and unified architecture designed to pre-train larger, more capable forecasting foundation models while reducing inference costs. By leveraging a sparse mixture-of-experts (MoE) design, TIME-MOE enhances computational efficiency by activating only a subset of networks for each prediction, reducing computational load while maintaining high model capacity. This allows TIME-MOE to scale effectively without a corresponding increase in inference costs. TIME-MOE comprises a family of decoder-only transformer models that operate in an autoregressive manner and support flexible forecasting horizons with varying input context lengths. We pre-trained these models on our newly introduced large-scale data Time-300B, which spans over 9 domains and encompassing over 300 billion time points. For the first time, we scaled a time series foundation model up to 2.4 billion parameters, achieving significantly improved forecasting precision. Our results validate the applicability of scaling laws for training tokens and model size in the context of time series forecasting. Compared to dense models with the same number of activated parameters or equivalent computation budgets, our models consistently outperform them by large margin. These advancements position TIME-MOE as a state-of-the-art solution for tackling real-world time series forecasting challenges with superior capability, efficiency, and flexibility. 

Resources: https://github.com/Time-MoE/Time-MoE 

## 1 INTRODUCTION

Time series data is a major modality in real-world dynamic systems and applications across various domains (Box et al., 2015; Zhang et al., 2024; Liang et al., 2024). Analyzing time series data is challenging due to its inherent complexity and distribution shifts, yet it is crucial for unlocking insights that enhance predictive analytics and decision-making. As a key task in high demand, time series forecasting has long been studied and is vital for driving various use cases in fields such as energy, climate, education, quantitative finance, cloud service, and urban computing, (Jin et al., 2023; Nie et al., 2024; Wang et al., 2023c; Mao et al., 2024). Traditionally, forecasting has been performed in a task-specific, end-to-end manner using either statistical or deep learning models. Despite their competitive performance, the field has not converged on building unified, general-purpose forecasting models until recently, with the emergence of a few foundation models (FMs) for universal forecasting (Das et al., 2024; Woo et al., 2024; Ansari et al., 2024). Although promising, they are generally small in scale and have limited task-solving capabilities compared to domain-specific models, limiting their real-world impact when balancing forecasting precision against computational budget. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/3b058d8f318d86578f4f247d72f8e00205b8e6ddfca19c75b0cb5754310241cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/afc9cb225bc4761220d0287ede8038ee56284b9651d9037a15f99c6b9200a22b.jpg)



Figure 1: Performance overview. (Left) Comparison between TIME-MOE models and state-of-theart time series foundation models, reporting the average zero-shot performance across six benchmark datasets. (Right) Comparison of few- and zero-shot performance between TIME-MOE and dense variants, with similar effective FLOPs per time series token, across the same six benchmarks.


Increasing model size and training tokens typically leads to performance improvements, as known as scaling laws, which have been extensively explored in the language and vision domains (Kaplan et al., 2020; Alabdulmohsin et al., 2022). However, such properties have not been thoroughly investigated in the time series domain (Yao et al., 2025). Assuming that scaling forecasting models with high-quality training data follows similar principles, several challenges remain: Dense versus sparse training. Most time series forecasting models compose of dense layers, which means each input time series tokens requires computations with all model parameters. While effective, this is computationally intensive. In contrast, sparse training with mixture-of-experts (MoE) is more flopefficient per parameter and allows for scaling up model size with a fixed inference budget while giving better performance, as showcased on the right of Figure 1. However, optimizing a sparse, large-scale time series model faces another challenge of stability and convergency. Time series are highly heterogeneous (Woo et al., 2024; Dong et al., 2024), and selecting the appropriate model design and routing algorithm often involves a trade-off between performance and computational efficiency. Sparse solutions for time series foundation models have yet to be explored, leaving a significant gap in addressing these two challenges. While time series pre-training datasets are no longer a major bottleneck, most existing works (Das et al., 2024; Woo et al., 2024; Ansari et al., 2024) have not extensively discussed their in-model data processing pipelines or mixing strategies. Answering this is particularly important, given that existing data archives are often noisy and largely imbalanced across domains. 

On the other hand, most time series FMs face limitations inflexibility and generalizability. Generalpurpose forecasting is a fundamental capability, requiring a model to handle any forecasting problems, regardless of context lengths, forecasting horizons, input variables, and other properties such as frequencies and distributions. Meanwhile, achieving strong generalizability pushes the boundaries further that existing works often fail to meet simultaneously. For instance, Timer (Liu et al., 2024d) has limited native support for arbitrary output lengths, which may lead to truncated outputs, while Moment (Goswami et al., 2024) operates with a fixed input context length. Although Moirai (Woo et al., 2024) achieves universal forecasting, it depends on hardcoded heuristics in both the input and output layers. 

The recognition of the above challenges naturally raises a pivotal question: 

How to scale time seriesfoundation models to achieve universalforecasting while balancing model capability and computational overhead, mirroring the success offoundation models in other domains? 

Answering this question drives the design of TIME-MOE, a scalable and unified architecture for pre-training larger, more capable forecasting FMs while reducing computational costs. TIME-MOE consists of a family of decoder-only transformer models with a mixture-of-experts architecture, operating in an auto-regressive manner to support any forecasting horizon and accommodate context lengths of up to 4096. With its sparsely activated design, TIME-MOE enhances computational efficiency by activating only a subset of networks for each prediction, reducing computational load while maintaining high model capacity. This allows TIME-MOE to scale effectively without significantly increasing inference costs. Our proposal is built on a minimalist design, where the input time series is point-wise tokenized and encoded before being processed by a sparse transformer decoder, activating only a small subset of parameters. Pre-trained on large-scale time series data across 9 domains and over 300 billion time points, TIME-MOE is optimized through multi-task learning to forecast at multiple resolutions. During inference, different forecasting heads are utilized to enable forecasts across diverse scales, enabling flexible forecast horizons. For the first time, we scale a time series FM up to 2.4 billion parameters, achieving substantial improvements in forecasting precision compared to existing models, as shown on the left of Figure 1. Compared to dense models with the same number of activated parameters or equivalent computational budgets, our models consistently outperform them by a large margin. Our contributions lie in three aspects: 

1. We present TIME-MOE, a universal decoder-only time series forecasting foundation model architecture with mixture-of-experts. To the best of our knowledge, this is the first work to scale time series foundation models up to 2.4 billion parameters. TIME-MOE achieves substantial improvements in forecasting accuracy and consistently outperforms dense models with comparable computational resources, while maintaining high efficiency. 

2. We introduce Time-300B, the largest open-access time series data collection, comprising over 300 billion time points spanning more than nine domains, accompanied by a well-designed datacleaning pipeline. Our TIME-MOE models and Time-300B data collection are open-sourced. 

3. Trained on Time-300B, TIME-MOE models outperform other time series foundation models with a similar number of activated parameters across six real-world benchmarks, achieving reductions in forecasting errors by an average of 20% and 24% in zero-shot and in-distribution scenarios, respectively. 

## 2 RELATED WORK

Time Series Forecasting. Deep learning models have become powerful tools for time series forecasting over the past decade, which can be broadly categorized into two types: (1) univariate models, such as DeepState (Rangapuram et al., 2018), DeepAR (Salinas et al., 2020), and N-BEATS (Oreshkin et al., 2020), which focus on modeling individual time series, and (2) multivariate models, which include both transformer-based approaches (Wen et al., 2023; Zhou et al., 2021; Nie et al., 2023; Liu et al., 2024b; Wang et al., 2024c; Chen et al., 2024; Wang et al., 2022) and non-transformer models (Sen et al., 2019; Jin et al., 2022; Wang et al., 2024b; Hu et al., 2024; Qi et al., 2024; Wang, 2024), designed to handle multiple time series simultaneously. While these models achieve competitive in-domain performance (Wang et al., 2025), many are task-specific and fall short in generalizability when applied to cross-domain data in few-shot or zero-shot scenarios. 

Large Time Series Models. Self-supervised learning has been extensively developed for time series (Zhang et al., 2024), employing masked reconstruction (Zerveas et al., 2021; Nie et al., 2023) or contrastive learning (Zhang et al., 2022; Wang et al., 2023b; Yue et al., 2022). However, these methods are limited in both data and model scale, with many focused on in-domain learning and transfer. Recently, general pre-training of time series models on large-scale data has emerged (Liang et al., 2024), though still in its early stages with insufficient exploration into sparse solutions. See Appendix A for more information. Unlike these dense models, TIME-MOE introduces a scalable, unified architecture for pre-training larger, more capable forecasting foundation models while maintaining the same scale of activated parameters and computational budget. 

Sparse Deep Learning for Time Series. Deep learning models are often dense and overparameterized (Hoefler et al., 2021), leading to increased memory and computational demands during both training and inference. However, sparse networks, such as mixture-of-experts models (Jacobs et al., 1991), which dynamically route inputs to specialized expert networks, have shown comparable or even superior generalization to dense models while being more efficient (Fedus et al., 2022; Riquelme et al., 2021). In time series research, model sparsification has received relatively less attention, as time series models have traditionally been small in scale, with simple models like DLinear (Zeng et al., 2023) and SparseTSF (Lin et al., 2024) excelling in specific tasks prior to the advent of large-scale, general pre-training. The most relevant works on this topic include Pathformer (Chen et al., 2024), MoLE (Ni et al., 2024), and IME (Ismail et al., 2023). However, none of them delve into the scalability of foundation models with sparse structures. Besides, MoLE and IME are not sparse models, as input data is passed to all heads and then combined to make predictions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/5d0b5ef91bf2f549185bfcdda5c78a2b7ea018c91947291800ed3b03d6d8a977.jpg)



Figure 2: The architecture of TIME-MOE, which is a decoder-only model. Given an input time series of arbitrary length, 1 we first tokenize it into a sequence of data points, 2 which are then encoded. These tokens are processed through N-stacked backbone layers, primarily consisting of causal multi-head self-attention and 3 sparse temporal mixture-of-expert layers. During training, 4 we optimize forecasting heads at multiple resolutions. For model inference, TIME-MOE provides forecasts of flexible length by 5 dynamically scheduling these heads. Details about the causal multihead self-attention are in Appendix B and illustrated in Figure 5.


## 3 METHODOLOGY

Our proposed TIME-MOE, illustrated in Figure 2, adopts a mixture-of-experts-based, decoder-only transformer architecture, comprising three key components: (1) input token embedding, (2) MoE transformer block, and (3) multi-resolution forecasting. For the first time, we scale a sparselyactivated time series model to 2.4 billion parameters, achieving significantly better zero-shot performance with the same computation. This marks a major step forward in developing large time series models for universal forecasting. 

Problem Statement. We address the problem of predicting future values in a time series: given a sequence of historical observations $\dot { \bf X } _ { 1 : T } = ( x _ { 1 } , \dot { x } _ { 2 } , \ldots , \dot { x } _ { T } ) \in \mathbb { R } ^ { T }$ spanning T time steps, our objective is to forecast the next H time steps, i.e., $\hat { \mathbf { X } } _ { T + 1 : T + H } = f _ { \theta } \left( \mathbf { X } _ { 1 : T } \right) \in \mathbb { R } ^ { H }$ . Here, $f _ { \theta }$ represents a time series model, where T is the context length and H is the forecasting horizon. Notably, both T and H can be flexible during TIME-MOE inference, distinguishing it from taskspecific models with fixed horizons. Additionally, channel independence (Nie et al., 2023) is adopted to transform a multivariate input into univariate series, allowing TIME-MOE to handle any-variate forecasting problems in real-world applications. 

## 3.1 TIME-MOE OVERVIEW

Input Token Embedding. We utilize point-wise tokenization for time series embedding to ensure the completeness of temporal information. This enhances our model’s flexibility and broad applicability in handling variable-length sequences. Then, we employ SwiGLU (Shazeer, 2020) to embed each time series point: 

$$
\mathbf {h} _ {t} ^ {0} = \operatorname{SwiGLU} (x _ {t}) = \operatorname{Swish} \left(W x _ {t}\right) \otimes \left(V x _ {t}\right),\tag{1}
$$

where $W \in R ^ { D \times 1 }$ and $V \in R ^ { D \times 1 }$ are learnable parameters, and D denotes the hidden dimension. 

MoE Transformer Block. Our approach builds upon a decoder-only transformer (Vaswani, 2017) and integrates recent advancements from large language models (Bai et al., 2023; Touvron et al., 2023). We employ RMSNorm (Zhang & Sennrich, 2019) to normalize the input of each transformer sub-layer, thereby enhancing training stability. Instead of using absolute positional encoding, we adopt rotary positional embeddings (Su et al., 2024), which provide greater flexibility in sequence length and improved extrapolation capabilities. In line with (Chowdhery et al., 2023), we remove biases from most layers but retain them in the QKV layer of self-attention to improve extrapolation. To introduce sparsity, we replace a feed-forward network (FFN) with a mixture-of-experts layer, incorporating a shared pool of experts that are sparsely activated. 

$$
\mathbf {u} _ {t} ^ {l} = \operatorname{SA} \left(\operatorname{RMSNorm} \left(\mathbf {h} _ {t} ^ {l - 1}\right)\right) + \mathbf {h} _ {t} ^ {l - 1},\tag{2}
$$

$$
\bar {\mathbf {u}} _ {t} ^ {l} = \mathrm{RMSNorm} \left(\mathbf {u} _ {t} ^ {l}\right),\tag{3}
$$

$$
\mathbf {h} _ {t} ^ {l} = \mathrm{Mixture} \left(\bar {\mathbf {u}} _ {t} ^ {l}\right) + \mathbf {u} _ {t} ^ {l}.\tag{4}
$$

Here, SA denotes self-attention with a causal mask, and Mixture refers to the mixture-of-experts layer. In practice, Mixture comprises several expert networks, each mirroring the architecture of a standard FFN. An individual time series point can be routed to either a single expert (Fedus et al., 2022) or multiple experts (Lepikhin et al., 2020). One expert is designated as a shared expert to capture and consolidate common knowledge across different contexts. 

$$
\text { Mixture } \left(\bar {\mathbf {u}} _ {t} ^ {l}\right) = g _ {N + 1, t} \operatorname{FFN} _ {N + 1} \left(\bar {\mathbf {u}} _ {t} ^ {l}\right) + \sum_ {i = 1} ^ {N} \left(g _ {i, t} \operatorname{FFN} _ {i} \left(\bar {\mathbf {u}} _ {t} ^ {l}\right)\right),\tag{5}
$$

$$
g _ {i, t} = \left\{ \begin{array}{l l} s _ {i, t}, & s _ {i, t} \in \operatorname{Topk} (\{s _ {j, t} | 1 \leq j \leq N \}, K), \\ 0, & \text { otherwise }, \end{array} \right.\tag{6}
$$

$$
g _ {N + 1, t} = \text { Sigmoid } \left(\mathbf {W} _ {N + 1} ^ {l} \bar {\mathbf {u}} _ {t} ^ {l}\right),\tag{7}
$$

$$
s _ {i, t} = \mathrm{Softmax} _ {i} \left(\mathbf {W} _ {i} ^ {l} \bar {\mathbf {u}} _ {t} ^ {l}\right),\tag{8}
$$

where $\mathbf { W } _ { i } ^ { l } \in \mathbb { R } ^ { 1 \times D }$ denotes the trainable parameters, and N and K respectively denote the numbers of non-shared experts and activated non-shared experts per mixture-of-experts layer. 

Multi-resolution Forecasting. We introduce a novel multi-resolution forecasting head, which allows for forecasting at multiple scales simultaneously, in contrast to existing foundation models that are limited to a single fixed scale. This capability enhances TIME-MOE ’s flexibility by enabling forecasting across various horizons. The model employs multiple output projections from single-layer FFNs, each designed for different prediction horizons. During training, TIME-MOE aggregates forecasting errors from different horizons to compute a composite loss (Section 3.2.2), thereby improving the model generalization. By incorporating a simple greedy scheduling algorithm (see Appendix B), TIME-MOE efficiently handles predictions across arbitrary horizons. This design also boosts prediction robustness through multi-resolution ensemble learning during inference. 

## 3.2 MODEL TRAINING

## 3.2.1 TIME-300B DATASET

Training time series foundation models require extensive, high-quality data. Recent advancements have facilitated the collection of numerous time series datasets from various sources (Godahewa et al., 2021; Ansari et al., 2024; Woo et al., 2024; Liu et al., 2024d;a). Nonetheless, data quality still remains a challenge, with prevalent issues such as missing values and invalid observations (Wang et al., 2024a) that can significantly impair model performance and destabilize training. To mitigate these issues, we developed a streamlined data-cleaning pipeline (Appendix C) to filter and refine raw data, and constructed the largest open-access, high-quality time series data collection named Time-300B for foundation model pre-training. Time-300B comprises a diverse array of publicly available datasets from domains such as energy, retail, healthcare, weather, finance, transportation, and web, augmented with synthetic data to enhance both quantity and diversity. It spans sampling frequencies from seconds to yearly intervals and, after processing through our data-cleaning pipeline, includes over 300 billion time points, as summarized in Table 1. 


Table 1: Key statistics of the pre-training dataset Time-300B from various domains.


<table><tr><td></td><td>Energy</td><td>Finance</td><td>Healthcare</td><td>Nature</td><td>Sales</td><td>Synthetic</td><td>Transport</td><td>Web</td><td>Other</td><td>Total</td></tr><tr><td># Seqs.</td><td>2,875,335</td><td>1,715</td><td>1,752</td><td>31,621,183</td><td>110,210</td><td>11,968,625</td><td>622,414</td><td>972,158</td><td>40,265</td><td>48,220,929</td></tr><tr><td># Obs.</td><td>15.981 B</td><td>413.696 K</td><td>471.040 K</td><td>279.724 B</td><td>26.382 M</td><td>9.222 B</td><td>2.130 B</td><td>1.804 B</td><td>20.32 M</td><td>309.09 B</td></tr><tr><td>Percent%</td><td>5.17 %</td><td>0.0001%</td><td>0.0001%</td><td>90.50 %</td><td>0.008 %</td><td>2.98%</td><td>0.69 %</td><td>0.58 %</td><td>0.006 %</td><td>100%</td></tr></table>

## 3.2.2 LOSS FUNCTION

Pre-training time series foundation models in large scale presents significant challenges in training stability due to the massive datasets and the vast number of parameters involved. To address this, we use the Huber loss (Huber, 1992; Wen et al., 2019), which provides greater robustness to outliers and improves training stability: 

$$
\mathcal {L} _ {\mathrm{ar}} \left(x _ {t}, \hat {x} _ {t}\right) = \left\{ \begin{array}{l l} \frac {1}{2} \left(x _ {t} - \hat {x} _ {t}\right) ^ {2}, & \text { if } | x _ {t} - \hat {x} _ {t} | \leq \delta , \\ \delta \times \left(| x _ {t} - \hat {x} _ {t} | - \frac {1}{2} \times \delta\right), & \text { otherwise }, \end{array} \right.\tag{9}
$$

where $\delta$ is a hyperparameter that balances the L1 and L2 loss components. 

When training the model with a MoE architecture, focusing solely on optimizing prediction error often leads to load imbalance issues among the experts. A common problem is routing collapse (Shazeer et al., 2017), where the model predominantly selects only a few experts, limiting training opportunities for others. To mitigate this, following the approaches of (Dai et al., 2024; Fedus et al., 2022), we achieve expert-level balancing with an auxiliary loss to reduce routing collapse: 

$$
\mathcal {L} _ {\text { aux }} = N \sum_ {i = 1} ^ {N} f _ {i} r _ {i}, \quad f _ {i} = \frac {1}{K T} \sum_ {t = 1} ^ {T} \mathbb {I} (\text { Time   point } t \text { selects   Expert } i), \quad r _ {i} = \frac {1}{T} \sum_ {t = 1} ^ {T} s _ {i, t},\tag{10}
$$

where $f _ { i }$ represents the fraction of tokens assigned to expert $i ,$ and $r _ { i }$ denotes the proportion of router probability allocated to expert i. I is the indicator function. Finally, we combine the auto-regressive losses across all multi-resolution projections with the auxiliary balance loss to form the final loss: 

$$
\mathcal {L} = \frac {1}{P} \sum_ {j = 1} ^ {P} \mathcal {L} _ {\mathrm{ar}} \left(\mathbf {X} _ {t + 1: t + p _ {j}}, \hat {\mathbf {X}} _ {t + 1: t + p _ {j}}\right) + \alpha \mathcal {L} _ {\mathrm{aux}},\tag{11}
$$

where $P$ is the number of multi-resolution projections and $p _ { j }$ is the horizon of the $j \cdot$ -th projection. 

## 3.2.3 MODEL CONFIGURATIONS AND TRAINING DETAILS

Informed by the scaling laws demonstrated in (Dubey et al., 2024; Touvron et al., 2023), which show that a 7- or 8-billion parameter model continues to improve performance even after training on over one trillion tokens, we chose to scale TIME-MOE up to 2.4 billion parameters with around 1 billion of them activated. This model, $\mathrm { { T I M E - M O E } _ { u l t r a } , }$ supports inference on consumer-grade GPUs with less than 8GB of VRAM. We have also developed two smaller models: TIME- $\mathbf { \partial } . \mathbf { M o E _ { b a s e } }$ , with 50 million activated parameters, and $\mathrm { T I M E - M O E _ { l a r g e } }$ , with 200 million activated parameters, both specifically designed for fast inference on CPU architectures. The detailed model configurations are in Table 2. Each model undergoes training for 100, 000 steps with a batch size of 1024, where the maximum sequence length is capped at 4096. This setup results in the consumption of 4 million time points per iteration. We choose {1, 8, 32, 64} as different forecast horizons in the output projection and set the factor of the auxiliary loss α to 0.02. Refer to Appendix B for optimization details. 


Table 2: A high-level summary of TIME-MOE model configurations.


<table><tr><td></td><td>Layers</td><td>Heads</td><td>Experts</td><td>K</td><td><eq>d_{model}</eq></td><td><eq>d_{ff}</eq></td><td><eq>d_{expert}</eq></td><td>Activated Params</td><td>Total Params</td></tr><tr><td>TIME-MOEbase</td><td>12</td><td>12</td><td>8</td><td>2</td><td>384</td><td>1536</td><td>192</td><td>50 M</td><td>113 M</td></tr><tr><td>TIME-MOElarge</td><td>12</td><td>12</td><td>8</td><td>2</td><td>768</td><td>3072</td><td>384</td><td>200 M</td><td>453 M</td></tr><tr><td>TIME-MOEultra</td><td>36</td><td>16</td><td>8</td><td>2</td><td>1024</td><td>4096</td><td>512</td><td>1.1 B</td><td>2.4 B</td></tr></table>

## 4 MAIN RESULTS

TIME-MOE consistently outperforms state-of-the-art models by large margins across 6 wellestablished benchmarks and settings (Appendix B). To ensure a fair comparison, we adhered to the configurations from (Woo et al., 2024) for out-of-distribution forecasting and (Wu et al., 2023a) for in-distribution forecasting with a unified evaluation pipeline we developed. Specifically, we evaluate TIME-MOE against 16 different baselines, representing state-of-the-art forecasting foundation models. They are categorized into two groups: (1) zero-shot forecasting group, includes pre-trained models such as Moirai (2024), TimesFM (2024), Moment (2024), and Chronos (2024); (2) in-distribution (full-shot) forecasting group, consists of up-to-date models such as iTransformer (2024b), TimeMixer (2024b), TimesNet (2023a), PatchTST (2023), Crossformer (2023), TiDE (2023), DLinear (2023),FEDformer (2022b). We also include addition comparisons with Timer (2024d), TFT (2021), and N-BEATS (2020) in Appendix D.3. 

## 4.1 ZERO-SHOT FORECASTING


Table 3: Full results of zero-shot forecasting experiments. A lower MSE or MAE indicates a better prediction. TimesFM, due to its use of Weather datasets in pretraining, is not evaluated on this dataset and is denoted by a dash (−). Red: the best, Blue: the 2nd best.


<table><tr><td rowspan="3">Models Metrics</td><td colspan="21">TIME-MoE(Ours)</td></tr><tr><td colspan="2">TIME-MoEbase</td><td colspan="2">TIME-MoElarge</td><td colspan="2">TIME-MOeultra</td><td colspan="2">Moirai small</td><td colspan="2">Moirai base</td><td colspan="2">Moirai large</td><td colspan="2">TimesFM</td><td colspan="2">Moment</td><td colspan="2">Chronossmall</td><td colspan="2">Chronosbase</td><td>Chronoslarge</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.357</td><td>0.381</td><td>0.350</td><td>0.382</td><td>0.349</td><td>0.379</td><td>0.401</td><td>0.402</td><td>0.376</td><td>0.392</td><td>0.381</td><td>0.388</td><td>0.414</td><td>0.404</td><td>0.688</td><td>0.557</td><td>0.466</td><td>0.409</td><td>0.440</td><td>0.393</td></tr><tr><td>192</td><td>0.384</td><td>0.404</td><td>0.388</td><td>0.412</td><td>0.395</td><td>0.413</td><td>0.435</td><td>0.421</td><td>0.412</td><td>0.413</td><td>0.434</td><td>0.415</td><td>0.465</td><td>0.434</td><td>0.688</td><td>0.560</td><td>0.530</td><td>0.450</td><td>0.492</td><td>0.426</td></tr><tr><td>336</td><td>0.411</td><td>0.434</td><td>0.411</td><td>0.430</td><td>0.447</td><td>0.453</td><td>0.438</td><td>0.434</td><td>0.433</td><td>0.428</td><td>0.495</td><td>0.445</td><td>0.503</td><td>0.456</td><td>0.675</td><td>0.563</td><td>0.570</td><td>0.486</td><td>0.550</td><td>0.462</td></tr><tr><td>720</td><td>0.449</td><td>0.477</td><td>0.427</td><td>0.455</td><td>0.457</td><td>0.462</td><td>0.439</td><td>0.454</td><td>0.447</td><td>0.444</td><td>0.611</td><td>0.510</td><td>0.511</td><td>0.481</td><td>0.683</td><td>0.585</td><td>0.615</td><td>0.543</td><td>0.882</td><td>0.591</td></tr><tr><td>Avg.</td><td>0.400</td><td>0.424</td><td>0.394</td><td>0.419</td><td>0.412</td><td>0.426</td><td>0.428</td><td>0.427</td><td>0.417</td><td>0.419</td><td>0.480</td><td>0.439</td><td>0.473</td><td>0.443</td><td>0.683</td><td>0.566</td><td>0.545</td><td>0.472</td><td>0.591</td><td>0.468</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.305</td><td>0.359</td><td>0.302</td><td>0.354</td><td>0.292</td><td>0.352</td><td>0.297</td><td>0.336</td><td>0.294</td><td>0.330</td><td>0.296</td><td>0.330</td><td>0.315</td><td>0.349</td><td>0.342</td><td>0.396</td><td>0.307</td><td>0.356</td><td>0.308</td><td>0.343</td></tr><tr><td>192</td><td>0.351</td><td>0.386</td><td>0.364</td><td>0.385</td><td>0.347</td><td>0.379</td><td>0.368</td><td>0.381</td><td>0.365</td><td>0.375</td><td>0.361</td><td>0.371</td><td>0.388</td><td>0.395</td><td>0.354</td><td>0.402</td><td>0.376</td><td>0.401</td><td>0.384</td><td>0.392</td></tr><tr><td>336</td><td>0.391</td><td>0.418</td><td>0.417</td><td>0.425</td><td>0.406</td><td>0.419</td><td>0.370</td><td>0.393</td><td>0.376</td><td>0.390</td><td>0.390</td><td>0.390</td><td>0.422</td><td>0.427</td><td>0.356</td><td>0.407</td><td>0.408</td><td>0.431</td><td>0.429</td><td>0.430</td></tr><tr><td>720</td><td>0.419</td><td>0.454</td><td>0.537</td><td>0.496</td><td>0.439</td><td>0.447</td><td>0.411</td><td>0.426</td><td>0.416</td><td>0.433</td><td>0.423</td><td>0.418</td><td>0.443</td><td>0.454</td><td>0.395</td><td>0.434</td><td>0.604</td><td>0.533</td><td>0.501</td><td>0.477</td></tr><tr><td>Avg.</td><td>0.366</td><td>0.404</td><td>0.405</td><td>0.415</td><td>0.371</td><td>0.399</td><td>0.361</td><td>0.384</td><td>0.362</td><td>0.382</td><td>0.367</td><td>0.377</td><td>0.392</td><td>0.406</td><td>0.361</td><td>0.409</td><td>0.424</td><td>0.430</td><td>0.405</td><td>0.410</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.338</td><td>0.368</td><td>0.309</td><td>0.357</td><td>0.281</td><td>0.341</td><td>0.418</td><td>0.392</td><td>0.363</td><td>0.356</td><td>0.380</td><td>0.361</td><td>0.361</td><td>0.370</td><td>0.654</td><td>0.527</td><td>0.511</td><td>0.423</td><td>0.454</td><td>0.408</td></tr><tr><td>192</td><td>0.353</td><td>0.388</td><td>0.346</td><td>0.381</td><td>0.305</td><td>0.358</td><td>0.431</td><td>0.405</td><td>0.388</td><td>0.375</td><td>0.412</td><td>0.383</td><td>0.414</td><td>0.405</td><td>0.662</td><td>0.532</td><td>0.618</td><td>0.485</td><td>0.567</td><td>0.477</td></tr><tr><td>336</td><td>0.381</td><td>0.413</td><td>0.373</td><td>0.408</td><td>0.369</td><td>0.395</td><td>0.433</td><td>0.412</td><td>0.416</td><td>0.392</td><td>0.436</td><td>0.400</td><td>0.445</td><td>0.429</td><td>0.672</td><td>0.537</td><td>0.683</td><td>0.524</td><td>0.662</td><td>0.525</td></tr><tr><td>720</td><td>0.504</td><td>0.493</td><td>0.475</td><td>0.477</td><td>0.469</td><td>0.472</td><td>0.462</td><td>0.432</td><td>0.460</td><td>0.418</td><td>0.462</td><td>0.420</td><td>0.512</td><td>0.471</td><td>0.692</td><td>0.551</td><td>0.748</td><td>0.566</td><td>0.900</td><td>0.591</td></tr><tr><td>Avg.</td><td>0.394</td><td>0.415</td><td>0.376</td><td>0.405</td><td>0.356</td><td>0.391</td><td>0.436</td><td>0.410</td><td>0.406</td><td>0.385</td><td>0.422</td><td>0.391</td><td>0.433</td><td>0.418</td><td>0.670</td><td>0.536</td><td>0.640</td><td>0.499</td><td>0.645</td><td>0.500</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.201</td><td>0.291</td><td>0.197</td><td>0.286</td><td>0.198</td><td>0.288</td><td>0.214</td><td>0.288</td><td>0.205</td><td>0.273</td><td>0.211</td><td>0.274</td><td>0.202</td><td>0.270</td><td>0.260</td><td>0.335</td><td>0.209</td><td>0.291</td><td>0.199</td><td>0.274</td></tr><tr><td>192</td><td>0.258</td><td>0.334</td><td>0.250</td><td>0.322</td><td>0.235</td><td>0.312</td><td>0.284</td><td>0.332</td><td>0.275</td><td>0.316</td><td>0.281</td><td>0.318</td><td>0.289</td><td>0.321</td><td>0.289</td><td>0.350</td><td>0.280</td><td>0.341</td><td>0.261</td><td>0.322</td></tr><tr><td>336</td><td>0.324</td><td>0.373</td><td>0.337</td><td>0.375</td><td>0.293</td><td>0.348</td><td>0.331</td><td>0.362</td><td>0.329</td><td>0.350</td><td>0.341</td><td>0.355</td><td>0.360</td><td>0.366</td><td>0.324</td><td>0.369</td><td>0.354</td><td>0.390</td><td>0.326</td><td>0.366</td></tr><tr><td>720</td><td>0.488</td><td>0.464</td><td>0.480</td><td>0.461</td><td>0.427</td><td>0.428</td><td>0.402</td><td>0.408</td><td>0.437</td><td>0.411</td><td>0.485</td><td>0.428</td><td>0.462</td><td>0.430</td><td>0.394</td><td>0.409</td><td>0.553</td><td>0.499</td><td>0.455</td><td>0.439</td></tr><tr><td>Avg.</td><td>0.317</td><td>0.365</td><td>0.316</td><td>0.361</td><td>0.288</td><td>0.344</td><td>0.307</td><td>0.347</td><td>0.311</td><td>0.337</td><td>0.329</td><td>0.343</td><td>0.328</td><td>0.346</td><td>0.316</td><td>0.365</td><td>0.349</td><td>0.380</td><td>0.310</td><td>0.350</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.160</td><td>0.214</td><td>0.159</td><td>0.213</td><td>0.157</td><td>0.211</td><td>0.198</td><td>0.222</td><td>0.220</td><td>0.217</td><td>0.199</td><td>0.211</td><td>-</td><td>-</td><td>0.243</td><td>0.255</td><td>0.211</td><td>0.243</td><td>0.203</td><td>0.238</td></tr><tr><td>192</td><td>0.210</td><td>0.260</td><td>0.215</td><td>0.266</td><td>0.208</td><td>0.256</td><td>0.247</td><td>0.265</td><td>0.271</td><td>0.259</td><td>0.246</td><td>0.251</td><td>-</td><td>-</td><td>0.278</td><td>0.329</td><td>0.263</td><td>0.294</td><td>0.256</td><td>0.290</td></tr><tr><td>336</td><td>0.274</td><td>0.309</td><td>0.291</td><td>0.322</td><td>0.255</td><td>0.290</td><td>0.283</td><td>0.303</td><td>0.286</td><td>0.297</td><td>0.274</td><td>0.291</td><td>-</td><td>-</td><td>0.306</td><td>0.346</td><td>0.321</td><td>0.339</td><td>0.314</td><td>0.336</td></tr><tr><td>720</td><td>0.418</td><td>0.405</td><td>0.415</td><td>0.400</td><td>0.405</td><td>0.397</td><td>0.373</td><td>0.354</td><td>0.373</td><td>0.354</td><td>0.337</td><td>0.340</td><td>-</td><td>-</td><td>0.350</td><td>0.374</td><td>0.404</td><td>0.397</td><td>0.397</td><td>0.396</td></tr><tr><td>Avg.</td><td>0.265</td><td>0.297</td><td>0.270</td><td>0.300</td><td>0.256</td><td>0.288</td><td>0.275</td><td>0.286</td><td>0.287</td><td>0.281</td><td>0.264</td><td>0.273</td><td>-</td><td>-</td><td>0.294</td><td>0.326</td><td>0.300</td><td>0.318</td><td>0.292</td><td>0.315</td></tr><tr><td rowspan="5">Global Temp</td><td>96</td><td>0.211</td><td>0.343</td><td>0.210</td><td>0.342</td><td>0.214</td><td>0.345</td><td>0.227</td><td>0.354</td><td>0.224</td><td>0.351</td><td>0.224</td><td>0.351</td><td>0.255</td><td>0.375</td><td>0.363</td><td>0.472</td><td>0.234</td><td>0.361</td><td>0.230</td><td>0.355</td></tr><tr><td>192</td><td>0.257</td><td>0.386</td><td>0.254</td><td>0.385</td><td>0.246</td><td>0.379</td><td>0.269</td><td>0.396</td><td>0.266</td><td>0.394</td><td>0.267</td><td>0.395</td><td>0.313</td><td>0.423</td><td>0.387</td><td>0.489</td><td>0.276</td><td>0.400</td><td>0.273</td><td>0.395</td></tr><tr><td>336</td><td>0.281</td><td>0.405</td><td>0.267</td><td>0.395</td><td>0.266</td><td>0.398</td><td>0.292</td><td>0.419</td><td>0.296</td><td>0.420</td><td>0.291</td><td>0.417</td><td>0.362</td><td>0.460</td><td>0.430</td><td>0.517</td><td>0.314</td><td>0.431</td><td>0.324</td><td>0.434</td></tr><tr><td>720</td><td>0.354</td><td>0.465</td><td>0.289</td><td>0.420</td><td>0.288</td><td>0.421</td><td>0.351</td><td>0.437</td><td>0.403</td><td>0.498</td><td>0.387</td><td>0.488</td><td>0.486</td><td>0.545</td><td>0.582</td><td>0.617</td><td>0.418</td><td>0.504</td><td>0.505</td><td>0.542</td></tr><tr><td>Avg.</td><td>0.275</td><td>0.400</td><td>0.255</td><td>0.385</td><td>0.253</td><td>0.385</td><td>0.285</td><td>0.409</td><td>0.297</td><td>0.416</td><td>0.292</td><td>0.413</td><td>0.354</td><td>0.451</td><td>0.440</td><td>0.524</td><td>0.311</td><td>0.424</td><td>0.333</td><td>0.431</td></tr><tr><td>Average</td><td></td><td>0.336</td><td>0.384</td><td>0.336</td><td>0.380</td><td>0.322</td><td>0.372</td><td>0.349</td><td>0.377</td><td>0.347</td><td>0.370</td><td>0.359</td><td>0.373</td><td>0.396</td><td>0.413</td><td>0.461</td><td>0.454</td><td>0.428</td><td>0.420</td><td>0.429</td><td>0.412</td></tr></table>

Setup. Time series foundation models have recently demonstrated impressive zero-shot learning capabilities (Liang et al., 2024; Liu et al., 2024c). In this section, we conducted experiments on the six well-known long-term forecasting benchmarks for which datasets were not included in the pre-training corpora. We use four different prediction horizons, which are {96, 192, 336, 720}, with the corresponding input time series lengths {512, 1024, 2048, 3072}. The evaluation metrics adopt mean square error (MSE) and mean absolute error (MAE). 

Results. Detailed results of zero-shot forecasting are in Table 3. TIME-MOE achieves consis tent state-of-the-art performances, improving a large margin as MSE reduction in average exceeding 20% over the other most competitive baselines. Importantly, as the model size scales $( \mathrm { e . g . , \bar { T } I M E { \mathrm { - } } M O E _ { b a s e }  T I M E { \mathrm { - } } M O E _ { u l t r a } ) }$ , it continuously exhibits enhanced performance across all datasets, affirming the efficacy of scaling laws within our time series foundation models. Furthermore, in comparisons with robust baselines that have a similar number of activated parameters, TIME-MOE demonstrates significantly superior performance. The largest models among the stateof-the-art baselines are $\mathrm { C h r o n o s } _ { \mathrm { l a r g e } } ,$ Moment and $\mathrm { M o i r a i _ { l a r g e } }$ Compared to those models, TIME-MOE achieves average MSE reductions of 23%, 30% and 11% respectively. 


Table 4: Full results of in-domain forecasting experiments. A lower MSE or MAE indicates a better prediction. Full-shot results besides Global Temp are obtained from (Liu et al., 2024b). Red: the best, Blue: the 2nd best.


<table><tr><td rowspan="3">Models Metrics</td><td colspan="6">TIME-MoE(Ours)</td><td colspan="15">Full-shot Time Series Models</td><td></td><td></td></tr><tr><td colspan="2">TIME-MoEbase</td><td colspan="2">TIME-MoElarge</td><td colspan="2">TIME-MoEultra</td><td colspan="2">iTransformer</td><td colspan="2">TimeMixer</td><td colspan="2">TimesNet</td><td colspan="2">PatchTST</td><td colspan="2">Crossformer</td><td colspan="2">TiDE</td><td colspan="2">DLinear</td><td>FEDformer</td><td></td><td></td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td></td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.345</td><td>0.373</td><td>0.335</td><td>0.371</td><td>0.323</td><td>0.365</td><td>0.386</td><td>0.405</td><td>0.375</td><td>0.400</td><td>0.384</td><td>0.402</td><td>0.414</td><td>0.419</td><td>0.423</td><td>0.448</td><td>0.479</td><td>0.464</td><td>0.386</td><td>0.400</td><td>0.376</td><td>0.419</td></tr><tr><td>192</td><td>0.372</td><td>0.396</td><td>0.374</td><td>0.400</td><td>0.359</td><td>0.391</td><td>0.441</td><td>0.436</td><td>0.436</td><td>0.429</td><td>0.421</td><td>0.429</td><td>0.460</td><td>0.445</td><td>0.471</td><td>0.474</td><td>0.525</td><td>0.492</td><td>0.437</td><td>0.432</td><td>0.420</td><td>0.448</td></tr><tr><td>336</td><td>0.389</td><td>0.412</td><td>0.390</td><td>0.412</td><td>0.388</td><td>0.418</td><td>0.487</td><td>0.458</td><td>0.484</td><td>0.458</td><td>0.491</td><td>0.469</td><td>0.501</td><td>0.466</td><td>0.570</td><td>0.546</td><td>0.565</td><td>0.515</td><td>0.481</td><td>0.459</td><td>0.459</td><td>0.465</td></tr><tr><td>720</td><td>0.410</td><td>0.443</td><td>0.402</td><td>0.433</td><td>0.425</td><td>0.450</td><td>0.503</td><td>0.491</td><td>0.498</td><td>0.482</td><td>0.521</td><td>0.500</td><td>0.500</td><td>0.488</td><td>0.653</td><td>0.621</td><td>0.594</td><td>0.558</td><td>0.519</td><td>0.516</td><td>0.506</td><td>0.507</td></tr><tr><td>Avg.</td><td>0.379</td><td>0.406</td><td>0.375</td><td>0.404</td><td>0.373</td><td>0.406</td><td>0.454</td><td>0.447</td><td>0.448</td><td>0.442</td><td>0.454</td><td>0.450</td><td>0.468</td><td>0.454</td><td>0.529</td><td>0.522</td><td>0.540</td><td>0.507</td><td>0.455</td><td>0.451</td><td>0.440</td><td>0.459</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.276</td><td>0.340</td><td>0.278</td><td>0.335</td><td>0.274</td><td>0.338</td><td>0.297</td><td>0.349</td><td>0.289</td><td>0.341</td><td>0.340</td><td>0.374</td><td>0.302</td><td>0.348</td><td>0.745</td><td>0.584</td><td>0.400</td><td>0.440</td><td>0.333</td><td>0.387</td><td>0.358</td><td>0.397</td></tr><tr><td>192</td><td>0.331</td><td>0.371</td><td>0.345</td><td>0.373</td><td>0.330</td><td>0.370</td><td>0.380</td><td>0.400</td><td>0.372</td><td>0.392</td><td>0.402</td><td>0.414</td><td>0.388</td><td>0.400</td><td>0.877</td><td>0.656</td><td>0.528</td><td>0.509</td><td>0.477</td><td>0.476</td><td>0.429</td><td>0.439</td></tr><tr><td>336</td><td>0.373</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.362</td><td>0.396</td><td>0.428</td><td>0.432</td><td>0.386</td><td>0.414</td><td>0.452</td><td>0.541</td><td>0.426</td><td>0.433</td><td>1.043</td><td>0.731</td><td>0.643</td><td>0.571</td><td>0.594</td><td>0.541</td><td>0.496</td><td>0.487</td></tr><tr><td>720</td><td>0.404</td><td>0.431</td><td>0.437</td><td>0.437</td><td>0.370</td><td>0.417</td><td>0.427</td><td>0.445</td><td>0.412</td><td>0.434</td><td>0.462</td><td>0.657</td><td>0.431</td><td>0.446</td><td>1.104</td><td>0.763</td><td>0.874</td><td>0.679</td><td>0.831</td><td>0.657</td><td>0.463</td><td>0.474</td></tr><tr><td>Avg.</td><td>0.346</td><td>0.386</td><td>0.361</td><td>0.386</td><td>0.334</td><td>0.380</td><td>0.383</td><td>0.406</td><td>0.364</td><td>0.395</td><td>0.414</td><td>0.496</td><td>0.386</td><td>0.406</td><td>0.942</td><td>0.683</td><td>0.611</td><td>0.549</td><td>0.558</td><td>0.515</td><td>0.436</td><td>0.449</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.286</td><td>0.334</td><td>0.264</td><td>0.325</td><td>0.256</td><td>0.323</td><td>0.334</td><td>0.368</td><td>0.320</td><td>0.357</td><td>0.338</td><td>0.375</td><td>0.329</td><td>0.367</td><td>0.404</td><td>0.426</td><td>0.364</td><td>0.387</td><td>0.345</td><td>0.372</td><td>0.379</td><td>0.419</td></tr><tr><td>192</td><td>0.307</td><td>0.358</td><td>0.295</td><td>0.350</td><td>0.281</td><td>0.343</td><td>0.377</td><td>0.391</td><td>0.361</td><td>0.381</td><td>0.374</td><td>0.387</td><td>0.367</td><td>0.385</td><td>0.450</td><td>0.451</td><td>0.398</td><td>0.404</td><td>0.380</td><td>0.389</td><td>0.426</td><td>0.441</td></tr><tr><td>336</td><td>0.354</td><td>0.390</td><td>0.323</td><td>0.376</td><td>0.326</td><td>0.374</td><td>0.426</td><td>0.420</td><td>0.390</td><td>0.404</td><td>0.410</td><td>0.411</td><td>0.399</td><td>0.410</td><td>0.532</td><td>0.515</td><td>0.428</td><td>0.425</td><td>0.413</td><td>0.413</td><td>0.445</td><td>0.459</td></tr><tr><td>720</td><td>0.433</td><td>0.445</td><td>0.409</td><td>0.435</td><td>0.454</td><td>0.452</td><td>0.491</td><td>0.459</td><td>0.454</td><td>0.441</td><td>0.478</td><td>0.450</td><td>0.454</td><td>0.439</td><td>0.666</td><td>0.589</td><td>0.487</td><td>0.461</td><td>0.474</td><td>0.453</td><td>0.543</td><td>0.490</td></tr><tr><td>Avg.</td><td>0.345</td><td>0.381</td><td>0.322</td><td>0.371</td><td>0.329</td><td>0.373</td><td>0.407</td><td>0.409</td><td>0.381</td><td>0.395</td><td>0.400</td><td>0.405</td><td>0.387</td><td>0.400</td><td>0.513</td><td>0.495</td><td>0.419</td><td>0.419</td><td>0.403</td><td>0.406</td><td>0.448</td><td>0.452</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.172</td><td>0.265</td><td>0.169</td><td>0.259</td><td>0.183</td><td>0.273</td><td>0.180</td><td>0.264</td><td>0.175</td><td>0.258</td><td>0.187</td><td>0.267</td><td>0.175</td><td>0.259</td><td>0.287</td><td>0.366</td><td>0.207</td><td>0.305</td><td>0.193</td><td>0.292</td><td>0.203</td><td>0.287</td></tr><tr><td>192</td><td>0.228</td><td>0.306</td><td>0.223</td><td>0.295</td><td>0.223</td><td>0.301</td><td>0.250</td><td>0.309</td><td>0.237</td><td>0.299</td><td>0.249</td><td>0.309</td><td>0.241</td><td>0.302</td><td>0.414</td><td>0.492</td><td>0.290</td><td>0.364</td><td>0.284</td><td>0.362</td><td>0.269</td><td>0.328</td></tr><tr><td>336</td><td>0.281</td><td>0.345</td><td>0.293</td><td>0.341</td><td>0.278</td><td>0.339</td><td>0.311</td><td>0.348</td><td>0.298</td><td>0.340</td><td>0.321</td><td>0.351</td><td>0.305</td><td>0.343</td><td>0.597</td><td>0.542</td><td>0.377</td><td>0.422</td><td>0.369</td><td>0.427</td><td>0.325</td><td>0.366</td></tr><tr><td>720</td><td>0.403</td><td>0.424</td><td>0.451</td><td>0.433</td><td>0.425</td><td>0.424</td><td>0.412</td><td>0.407</td><td>0.391</td><td>0.396</td><td>0.408</td><td>0.403</td><td>0.402</td><td>0.400</td><td>1.730</td><td>1.042</td><td>0.558</td><td>0.524</td><td>0.554</td><td>0.522</td><td>0.421</td><td>0.415</td></tr><tr><td>Avg.</td><td>0.271</td><td>0.335</td><td>0.284</td><td>0.332</td><td>0.277</td><td>0.334</td><td>0.288</td><td>0.332</td><td>0.275</td><td>0.323</td><td>0.291</td><td>0.332</td><td>0.280</td><td>0.326</td><td>0.757</td><td>0.610</td><td>0.358</td><td>0.403</td><td>0.350</td><td>0.400</td><td>0.304</td><td>0.349</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.151</td><td>0.203</td><td>0.149</td><td>0.201</td><td>0.154</td><td>0.208</td><td>0.174</td><td>0.214</td><td>0.163</td><td>0.209</td><td>0.172</td><td>0.220</td><td>0.177</td><td>0.218</td><td>0.158</td><td>0.230</td><td>0.202</td><td>0.261</td><td>0.196</td><td>0.255</td><td>0.217</td><td>0.296</td></tr><tr><td>192</td><td>0.195</td><td>0.246</td><td>0.192</td><td>0.244</td><td>0.202</td><td>0.251</td><td>0.221</td><td>0.254</td><td>0.208</td><td>0.250</td><td>0.219</td><td>0.261</td><td>0.225</td><td>0.259</td><td>0.206</td><td>0.277</td><td>0.242</td><td>0.298</td><td>0.237</td><td>0.296</td><td>0.276</td><td>0.336</td></tr><tr><td>336</td><td>0.247</td><td>0.288</td><td>0.245</td><td>0.285</td><td>0.252</td><td>0.287</td><td>0.278</td><td>0.296</td><td>0.251</td><td>0.287</td><td>0.280</td><td>0.306</td><td>0.278</td><td>0.297</td><td>0.272</td><td>0.335</td><td>0.287</td><td>0.335</td><td>0.283</td><td>0.335</td><td>0.339</td><td>0.380</td></tr><tr><td>720</td><td>0.352</td><td>0.366</td><td>0.352</td><td>0.365</td><td>0.392</td><td>0.376</td><td>0.358</td><td>0.349</td><td>0.339</td><td>0.341</td><td>0.365</td><td>0.359</td><td>0.354</td><td>0.348</td><td>0.398</td><td>0.418</td><td>0.351</td><td>0.386</td><td>0.345</td><td>0.381</td><td>0.403</td><td>0.428</td></tr><tr><td>Avg.</td><td>0.236</td><td>0.275</td><td>0.234</td><td>0.273</td><td>0.250</td><td>0.280</td><td>0.257</td><td>0.278</td><td>0.240</td><td>0.271</td><td>0.259</td><td>0.286</td><td>0.258</td><td>0.280</td><td>0.258</td><td>0.315</td><td>0.270</td><td>0.320</td><td>0.265</td><td>0.316</td><td>0.308</td><td>0.360</td></tr><tr><td rowspan="5">Global Temp</td><td>96</td><td>0.192</td><td>0.328</td><td>0.192</td><td>0.329</td><td>0.189</td><td>0.322</td><td>0.223</td><td>0.351</td><td>0.215</td><td>0.346</td><td>0.250</td><td>0.381</td><td>0.219</td><td>0.349</td><td>0.272</td><td>0.406</td><td>0.223</td><td>0.352</td><td>0.221</td><td>0.354</td><td>0.261</td><td>0.392</td></tr><tr><td>192</td><td>0.238</td><td>0.375</td><td>0.236</td><td>0.375</td><td>0.234</td><td>0.376</td><td>0.282</td><td>0.404</td><td>0.266</td><td>0.393</td><td>0.298</td><td>0.418</td><td>0.269</td><td>0.395</td><td>0.305</td><td>0.435</td><td>0.278</td><td>0.401</td><td>0.257</td><td>0.388</td><td>0.299</td><td>0.423</td></tr><tr><td>336</td><td>0.259</td><td>0.397</td><td>0.256</td><td>0.397</td><td>0.253</td><td>0.399</td><td>0.313</td><td>0.431</td><td>0.313</td><td>0.430</td><td>0.315</td><td>0.434</td><td>0.319</td><td>0.435</td><td>0.352</td><td>0.468</td><td>0.330</td><td>0.440</td><td>0.294</td><td>0.418</td><td>0.341</td><td>0.454</td></tr><tr><td>720</td><td>0.345</td><td>0.465</td><td>0.322</td><td>0.451</td><td>0.292</td><td>0.426</td><td>0.393</td><td>0.488</td><td>0.468</td><td>0.536</td><td>0.407</td><td>0.497</td><td>0.452</td><td>0.526</td><td>0.508</td><td>0.562</td><td>0.485</td><td>0.544</td><td>0.380</td><td>0.479</td><td>0.359</td><td>0.469</td></tr><tr><td>Avg.</td><td>0.258</td><td>0.391</td><td>0.251</td><td>0.388</td><td>0.242</td><td>0.380</td><td>0.303</td><td>0.419</td><td>0.316</td><td>0.426</td><td>0.318</td><td>0.433</td><td>0.315</td><td>0.426</td><td>0.359</td><td>0.468</td><td>0.329</td><td>0.434</td><td>0.288</td><td>0.410</td><td>0.315</td><td>0.435</td></tr><tr><td colspan="2">Average</td><td>0.306</td><td>0.362</td><td>0.304</td><td>0.359</td><td>0.301</td><td>0.358</td><td>0.349</td><td>0.382</td><td>0.337</td><td>0.375</td><td>0.356</td><td>0.400</td><td>0.349</td><td>0.382</td><td>0.560</td><td>0.516</td><td>0.421</td><td>0.439</td><td>0.387</td><td>0.416</td><td>0.375</td><td>0.417</td></tr></table>

## 4.2 IN-DISTRIBUTION FORECASTING

Setup. We fine-tune the pre-trained TIME-MOE models on the train split of the above-mentioned six benchmarks and set the number of finetuning epochs to only one. 

Results. The full results are in Table 4. TIME-MOE exhibits remarkable capabilities, comprehensively surpassing advanced deep time series models from recent years, achieving a MSE reduction of24% in average. Fine-tuning on downstream data with only one epoch significantly improves predictive performance, showcasing the remarkable potential of large time series models built on the MoE architecture. Similar to zero-shot forecasting, as the model size increases, the scaling law continues to be effective, leading to continuous improvements in the performance of the TIME-MOE. 

## 4.3 ABLATION STUDY


Table 5: Ablation studies. (Left) Average MSE for horizon-96 forecasting across six benchmarks, evaluated with different model components. (Right) Analysis of various multi-resolution forecasting configurations. More details are in Appendix D.1.


<table><tr><td></td><td>Average MSE</td></tr><tr><td>TIME-MOEbase</td><td>0.262</td></tr><tr><td>w/o Huber loss</td><td>0.267</td></tr><tr><td>w/o multi-resolution layer</td><td>0.269</td></tr><tr><td>w/o mixture-of-experts</td><td>0.272</td></tr><tr><td>w/o auxiliary loss</td><td>0.275</td></tr></table>

<table><tr><td></td><td>Average MSE</td><td>Inference Speed</td></tr><tr><td>TIME-MOEbase w/ {1,8,32,64}</td><td>0.262</td><td>0.095 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1,8,32}</td><td>0.273</td><td>0.130 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1,8}</td><td>0.320</td><td>0.411 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1}</td><td>1.382</td><td>2.834 s/iter</td></tr></table>

To validate our designs in TIME-MOE, we conducted detailed ablation studies on key architectural components and loss functions across all experimental benchmarks, as shown in Table 5. 

Model Architecture. Replacing the MoE layers with standard FFNs (w/o mixture-of-experts) led to an average performance drop from 0.262 to 0.272, highlighting the performance boost provided by the sparse architecture. A detailed comparison of dense and sparse models is presented in Section 4.4. We retained only the horizon-32 output layer by eliminating the other multi-resolution output layers from the TIME- $\mathbf { M o E _ { b a s e } } ,$ , excluding the multi-task optimization (w/o multi-resolution layer). Consequently, we observed that the performance of this modified model was slightly inferior compared to that of the $\mathrm { T I M E - M O E } _ { \mathrm { b a s e } }$ . Additionally, as shown in the right side of Table 5, our default selection of four multi-resolution output projections with receptive horizons of {1, 8, 32, 64} results in optimal predictive performance and inference speed. As we reduce the number of multiresolution output projections, performance consistently declines, and inference speed significantly increases. This demonstrates the rationality of our multi-resolution output projection design. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/4ea2e498505ae58728d967defe580d9d6043c48e9bd3f57c7fbdc964e809e6ae.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/888fe6e7904276bed9eace1eacf3b599fb4959fb5f927a2946faa1964cb070af.jpg)



Figure 3: Scalability analysis. (Left) Comparison of dense and sparse models in terms of training and inference costs. (Right) Average MSE for 96-horizon forecasting across six benchmarks, comparing TIME-MOE and dense models, both trained from scratch with varying data sizes.


Training Loss. Models trained with Huber loss outperformed those using MSE loss (w/o Huber loss), due to Huber loss’s superior robustness in handling outlier time points. We also removed the auxiliary loss from the objective function, retaining only the auto-regressive loss (w/o auxiliary loss) while still using the MoE architecture. This adjustment caused the expert layers to collapse into a smaller FFN during training, as the activation score of the most effective expert became disproportionately stronger without the load balance loss. Consequently, the model’s performance was significantly worse than the TIME- $\mathbf { \partial } . \mathbf { M o E _ { b a s e } }$ 

## 4.4 SCALABILITY ANALYSIS

Dense versus Sparse Models. To assess the performance and efficiency benefits of sparse architectures in time series forecasting, we replaced the MoE layer with a dense layer containing an equivalent number of parameters as the activated parameters in the MoE layer. Using identical training setup and data, we trained three dense models corresponding to the sizes of the three TIME-MOE models. A zero-shot performance comparison between the dense and sparse models is shown in Figure 3. Our approach reduced training costs by an average of 78% and inference costs by 39% compared to dense variants. This clearly demonstrates the advantages of TIME-MOE, particularly in maintaining exceptional performance while significantly reducing costs. 

Model and Data Scaling. We save model checkpoints at intervals of every 20 billion time points during training, allowing to plot performance traces for models of different sizes trained on various data scales. The right side of Figure 3 shows that models trained on larger datasets consistently outperform those trained on smaller datasets, regardless of model size. Our empirical results confirm that as both data volume and model parameters scale, sparse models demonstrate continuous and substantial improvements in performance, as well as achieve better forecasting accuracy compared to the dense counterparts under the same scales. 

Training Precision. We trained a new model, $\mathrm { T I M E - M O E } _ { \mathrm { b a s e } }$ (FP32), using identical configurations but with float32 precision instead of bfloat16. As shown in Table 6, the forecasting performance of both models is comparable. However, the bfloat16 model achieves a 12% improvement in training speed and reduces memory consumption by 20% compared to the float32 model. Moreover, the bfloat16 model can seamlessly integrate with flash-attention (Dao, 2024), further boosting training and inference speed by 23% and 19% respectively. 


Table 6: Comparison of BF16 and FP32 in terms of training and inference efficiency. FA denotes flash-attention. More details are in Table 13 of Appendix D.2.


<table><tr><td></td><td>Average MSE</td><td>Training Speed</td><td>Inference Speed</td><td>Training Memory</td><td>Inference Memory</td></tr><tr><td>TIME-MOEbase</td><td>0.262</td><td>0.84 s/iter</td><td>0.095 s/iter</td><td>1.77 GB</td><td>226.70 MB</td></tr><tr><td>TIME-MOEbase w/o FA</td><td>0.262</td><td>1.09 s/iter</td><td>0.118 s/iter</td><td>1.77 GB</td><td>226.70 MB</td></tr><tr><td>TIME-MOEbase w/ FP32</td><td>0.261</td><td>1.24 s/iter</td><td>0.133 s/iter</td><td>2.21 GB</td><td>453.41 MB</td></tr></table>

## 4.5 SPARSIFICATION ANALYSIS

Activation Visualization. As shown in Figure 4, TIME-MOE dynamically activates different experts across various datasets, with each expert specializing in learning distinct knowledge. This leads to diverse activation patterns across datasets from different domains, showcasing TIME-MOE’s strong generalization capabilities. The heterogeneous activations indicate that the model adapts its learned representations to the specific characteristics of each dataset, contributing to its great transferability and generalization as a large-scale time series foundation model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/cef22b2f552fc981c98449480eb58d30003768ead37c6057a957044ad015c621.jpg)



Figure 4: Gating scores for experts across different layers in the six benchmarks.


Number of Experts. We performed a sensitivity analysis on the number of experts, represented as top<sub>k</sub>, within the TIME-MOE architecture, as shown in Table 7. As k increases, performance shows only marginal changes, with minimal improvements in average MSE. However, inference time increases noticeably as more experts are utilized. This indicates that increasing sparsity within the MoE architecture does not compromise performance but significantly enhances computational efficiency. This balance is critical for scaling time series foundation models, where optimizing performance and computational cost is essential. Sparse MoE architectures inherentl y offer advantages in these areas. 

Table 7: Performance and inference speed across different top setups. Average MSE for horizon-96 forecasting evaluated across six benchmarks. Lower values of inference speed (s/iter) indicate better performance. 

<table><tr><td>TIME-MOEbase</td><td>Average MSE</td><td>Inference Speed</td></tr><tr><td>w/ {Top1}</td><td>0.264</td><td>0.082 s/iter</td></tr><tr><td>w/ {Top2}</td><td>0.262</td><td>0.095 s/iter</td></tr><tr><td>w/ {Top4}</td><td>0.262</td><td>0.109 s/iter</td></tr><tr><td>w/ {Top6}</td><td>0.265</td><td>0.120 s/iter</td></tr><tr><td>w/ {Top8}</td><td>0.269</td><td>0.129 s/iter</td></tr></table>

## 5 CONCLUSION

In this paper, we introduced TIME-MOE, a scalable and unified architecture for time series foundation models that leverages a sparse design with mixture-of-experts to enhance computational efficiency without compromising model capacity. Pre-trained on our newly introduced large-scale time series dataset, Time-300B, TIME-MOE was scaled to 2.4 billion parameters, with 1.1 billion activated, demonstrating significant improvements in forecasting capabilities. Our results validate the scaling properties in time series forecasting, showing that TIME-MOE consistently outperforms dense models with equivalent computational budgets across multiple widely accepted benchmarks. With its ability to perform universal forecasting and superior performance in both zero-shot and finetuned scenarios, TIME-MOE establishes itself as a state-of-the-art solution for real-world forecasting challenges. This work lays the groundwork for future advancements in scaling and enhancing the efficiency of time series foundation models, paving the way toward time series general intelligence. 

## ACKNOWLEDGEMENT

Y. Nie acknowledges financial support from Princeton Language and Intelligence at Princeton University. M. Jin was supported in part by the NVIDIA Academic Grant Program and CSIRO – National Science Foundation (US) AI Research Collaboration Program. 

## REFERENCES



Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in language and vision. Advances in Neural Information Processing Systems, 35:22300–22312, 2022. 





Alexander Alexandrov, Konstantinos Benidis, Michael Bohlke-Schneider, Valentin Flunkert, Jan Gasthaus, Tim Januschowski, Danielle C. Maddix, Syama Rangapuram, David Salinas, Jasper Schulz, Lorenzo Stella, Ali Caner TA¼rkmen, and Yuyang Wang. Gluonts: Probabilistic and<sup>˜</sup> neural time series modeling in python. Journal of Machine Learning Research, 21(116):1–6, 2020. 





Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024. 





Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 





Christoph Bergmeir, Quang Bui, Frits de Nijs, and Peter Stuckey. Residential power and battery data, August 2023. URL https://doi.org/10.5281/zenodo.8219786. 





George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015. 





CDC. Flu portal dashboard, 2017. URL https://gis.cdc.gov/grasp/fluview/ fluportaldashboard.html. 





Peng Chen, Yingying Zhang, Yunyao Cheng, Yang Shu, Yihang Wang, Qingsong Wen, Bin Yang, and Chenjuan Guo. Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting. In International Conference on Learning Representations, 2024. 





Song Chen. Beijing Multi-Site Air-Quality Data. UCI Machine Learning Repository, 2019. DOI: https://doi.org/10.24432/C5RK5G. 





Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240): 1–113, 2023. 





Together Computer. Redpajama: an open dataset for training large language models, 2023. URL https://github.com/togethercomputer/RedPajama-Data. 





Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. arXiv preprint arXiv:2401.06066, 2024. 





Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024. 





Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K Mathur, Rajat Sen, and Rose Yu. Longterm forecasting with tide: Time-series dense encoder. Transactions on Machine Learning Research, 2023. 





Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning, 2024. 





Zheng Dong, Renhe Jiang, Haotian Gao, Hangchen Liu, Jinliang Deng, Qingsong Wen, and Xuan Song. Heterogeneity-informed meta-parameter learning for spatiotemporal time series forecasting. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 631–641, 2024. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 





Patrick Emami, Abhijeet Sahu, and Peter Graf. Buildingsbench: A large-scale dataset of 900k buildings and benchmark for short-term load forecasting. Advances in Neural Information Processing Systems, 36:19823–19857, 2023. 





William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022. 





Azul Garza, Cristian Challu, and Max Mergenthaler-Canseco. Timegpt-1. arXiv preprint arXiv:2310.03589, 2023. 





Rakshitha Wathsadini Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob Hyndman, and Pablo Montero-Manso. Monash time series forecasting archive. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https: //openreview.net/forum?id=wEc1mgAjU-. 





Georg Goerg. Forecastable component analysis. ICML, 2013. 





Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. Moment: A family of open time-series foundation models. In Forty-first International Conference on Machine Learning, 2024. 





Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. Journal of Machine Learning Research, 22(241):1–124, 2021. 





Jiaxi Hu, Yuehong Hu, Wei Chen, Ming Jin, Shirui Pan, Qingsong Wen, and Yuxuan Liang. Attractor memory for long-term time series forecasting: A chaos perspective. arXiv preprint arXiv:2402.11463, 2024. 





Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pp. 492–518. Springer, 1992. 





Aya Abdelsalam Ismail, Sercan O Arik, Jinsung Yoon, Ankur Taly, Soheil Feizi, and Tomas Pfister. Interpretable mixture of experts. Transactions on Machine Learning Research, 2023. ISSN 2835- 8856. 





Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991. 





Ming Jin, Yu Zheng, Yuan-Fang Li, Siheng Chen, Bin Yang, and Shirui Pan. Multivariate time series forecasting with dynamic graph neural odes. IEEE Transactions on Knowledge and Data Engineering, 35(9):9168–9180, 2022. 





Ming Jin, Qingsong Wen, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li, et al. Large models for time series and spatio-temporal data: A survey and outlook. arXiv preprint arXiv:2310.10196, 2023. 





Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang, Bin Yang, Jindong Wang, Shirui Pan, and Qingsong Wen. Position: What can large language models tell us about time series analysis. In Forty-first International Conference on Machine Learning, 2024. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 





Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. 





Yuxuan Liang, Haomin Wen, Yuqi Nie, Yushan Jiang, Ming Jin, Dongjin Song, Shirui Pan, and Qingsong Wen. Foundation models for time series analysis: A tutorial and survey. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 6555– 6565, 2024. 





Bryan Lim, Sercan O Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for<sup>¨</sup> interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4): 1748–1764, 2021. 





Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, and Junjie Yang. Sparsetsf: Modeling long-term time series forecasting with 1k parameters. In Forty-first International Conference on Machine Learning, 2024. 





Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. Largest: A benchmark dataset for large-scale traffic forecasting. arXiv preprint arXiv:2306.08259, 2023. 





Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Moirai-moe: Empowering time series foundation models with sparse mixture of experts. arXiv preprint arXiv:2410.10469, 2024a. 





Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024b. 





Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Autotimes: Autoregressive time series forecasters via large language models. arXiv preprint arXiv:2402.02370, 2024c. 





Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer: Generative pre-trained transformers are large time series models. In Forty-first International Conference on Machine Learning, 2024d. 





Paolo Mancuso, Veronica Piccialli, and Antonio M Sudoso. A machine learning approach for forecasting hierarchical time series. Expert Systems with Applications, 182:115102, 2021. 





Shengzhong Mao, Chaoli Zhang, Yichi Song, Jindong Wang, Xiao-Jun Zeng, Zenglin Xu, and Qingsong Wen. Time series analysis for education: Methods, applications, and future directions. arXiv preprint arXiv:2408.13960, 2024. 





Soukayna Mouatadid, Paulo Orenstein, Genevieve Elaine Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Edward Knight, Maria Geogdzhayeva, Samuel James Levang, Ernest Fraenkel, and Lester Mackey. SubseasonalclimateUSA: A dataset for subseasonal forecasting and benchmarking. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 





Tung Nguyen, Jason Kyle Jewik, Hritik Bansal, Prakhar Sharma, and Aditya Grover. Climatelearn: Benchmarking machine learning for weather and climate modeling. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 





Ronghao Ni, Zinan Lin, Shuaiqi Wang, and Giulia Fanti. Mixture-of-linear-experts for long-term time series forecasting. In International Conference on Artificial Intelligence and Statistics, pp. 4672–4680. PMLR, 2024. 





Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. 





Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M Mulvey, H Vincent Poor, Qingsong Wen, and Stefan Zohren. A survey of large language models for financial applications: Progress, prospects and challenges. arXiv preprint arXiv:2406.11903, 2024. 





Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations, 2020. 





ourownstory. Neuralprophet datasets, 2023. URL https://github.com/ourownstory/ neuralprophet-data. 





Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023. 





Shiyi Qi, Zenglin Xu, Yiduo Li, Liangjian Wen, Qingsong Wen, Qifan Wang, and Yuan Qi. Pdetime: Rethinking long-term multivariate time series forecasting from the perspective of partial differential equations. arXiv preprint arXiv:2402.16913, 2024. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020. 





Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. Advances in neural information processing systems, 31, 2018. 





Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn, Soukayna Mouatadid, and Nils Thuerey. Weatherbench: a benchmark data set for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12(11):e2020MS002203, 2020. 





Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Bilos, Hena Ghonia, Nadhir Vincent Hassen, Anderson Schneider,ˇ Sahil Garg, Alexandre Drouin, Nicolas Chapados, Yuriy Nevmyvaka, and Irina Rish. Lag-llama: Towards foundation models for time series forecasting, 2023. 





Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre´ Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021. 





David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International journal offorecasting, 36(3):1181– 1191, 2020. 





Rajat Sen, Hsiang-Fu Yu, and Inderjit S Dhillon. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. Advances in neural information processing systems, 32, 2019. 





N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017. 





Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 





Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Willem G van Panhuis, Anne Cross, and Donald S Burke. Project tycho 2.0: a repository to improve the integration and reuse of data for global population health. Journal of the American Medical Informatics Association, 25:1608–1617, 2018. 





Ashish Vaswani. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017. 





Jingyuan Wang, Jiawei Jiang, Wenjun Jiang, Chengkai Han, and Wayne Xin Zhao. Towards efficient and comprehensive urban spatial-temporal prediction: A unified library and performance benchmark. arXiv preprint arXiv:2304.14343, 2023a. 





Jun Wang, Wenjie Du, Wei Cao, Keli Zhang, Wenjia Wang, Yuxuan Liang, and Qingsong Wen. Deep learning for multivariate time series imputation: A survey. arXiv preprint arXiv:2402.04059, 2024a. 





Shiyu Wang. Neuralreconciler for hierarchical time series forecasting. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pp. 731–739, 2024. 





Shiyu Wang, Fan Zhou, Yinbo Sun, Lintao Ma, James Zhang, and Yangfei Zheng. End-to-end modeling of hierarchical time series using autoregressive transformer and conditional normalizing flow-based reconciliation. In 2022 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 1087–1094. IEEE, 2022. 





Shiyu Wang, Yinbo Sun, Xiaoming Shi, Zhu Shiyi, Lin-Tao Ma, James Zhang, YangFei Zheng, and Liu Jian. Full scaling automation for sustainable development of green data centers. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pp. 6264–6271, 2023b. 





Shiyu Wang, Yinbo Sun, Yan Wang, Fan Zhou, Lin-Tao Ma, James Zhang, and YangFei Zheng. Flow-based end-to-end model for hierarchical time series forecasting via trainable attentivereconciliation. In International Conference on Database Systems for Advanced Applications, pp. 167–176. Springer, 2023c. 





Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang, and Jun Zhou. Timemixer: Decomposable multiscale mixing for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024b. 





Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, and Ming Jin. Timemixer++: A general time series pattern machine for universal predictive analysis. In The Thirteenth International Conference on Learning Representations (ICLR), 2025. 





Xue Wang, Tian Zhou, Qingsong Wen, Jinyang Gao, Bolin Ding, and Rong Jin. Card: Channel aligned robust blend transformer for time series forecasting. In The Twelfth International Conference on Learning Representations (ICLR), 2024c. 





Zhixian Wang, Qingsong Wen, Chaoli Zhang, Liang Sun, Leandro Von Krannichfeldt, and Yi Wang. Benchmarks and custom package for electrical load forecasting. arXiv preprint arXiv:2307.07191, 2023d. 





Qingsong Wen, Jingkun Gao, Xiaomin Song, Liang Sun, and Jian Tan. RobustTrend: a huber loss with a combined first and second order difference regularization for time series trend filtering. In Proceedings ofthe 28th International Joint Conference on Artificial Intelligence, pp. 3856–3862, 2019. 





Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: a survey. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI), pp. 6778–6786, 2023. 





Gerald Woo, Chenghao Liu, Akshat Kumar, and Doyen Sahoo. Pushing the limits of pre-training for time series forecasting in the cloudops domain. arXiv preprint arXiv:2310.05063, 2023. 





Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In Forty-first International Conference on Machine Learning, 2024. 





Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021. 





Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023a. 





Haixu Wu, Hang Zhou, Mingsheng Long, and Jianmin Wang. Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 2023b. 





Qingren Yao, Chao-Han Huck Yang, Renhe Jiang, Yuxuan Liang, Ming Jin, and Shirui Pan. Towards neural scaling laws for time series foundation models. In The Thirteenth International Conference on Learning Representations (ICLR), 2025. 





Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pp. 8980–8987, 2022. 





Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023. 





George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty, and Carsten Eickhoff. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2114–2124, 2021. 





Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 





Junbo Zhang, Yu Zheng, and Dekang Qi. Deep spatio-temporal residual networks for citywide crowd flows prediction. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017. 





Kexin Zhang, Qingsong Wen, Chaoli Zhang, Rongyao Cai, Ming Jin, Yong Liu, James Y Zhang, Yuxuan Liang, Guansong Pang, Dongjin Song, et al. Self-supervised learning for time series analysis: Taxonomy, progress, and prospects. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 





Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik. Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in Neural Information Processing Systems, 35:3988–4003, 2022. 





Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In International Conference on Learning Representations, 2023. 





Yu Zheng, Xiuwen Yi, Ming Li, Ruiyuan Li, Zhangqing Shan, Eric Chang, and Tianrui Li. Forecasting fine-grained air quality based on big data. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 2267–2276, 2015. 





Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, pp. 11106–11115, 2021. 





Jingbo Zhou, Xinjiang Lu, Yixiong Xiao, Jiantao Su, Junfu Lyu, Yanjun Ma, and Dejing Dou. Sdwpf: A dataset for spatial dynamic wind power forecasting challenge at kdd cup 2022. arXiv preprint arXiv:2208.04360, 2022a. 





Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International Conference on Machine Learning (ICML 2022), 2022b. 



## A FURTHER RELATED WORK

In this section, we delve deeper into the related work on large time series models. Current research efforts in universal forecasting with time series foundation models can be broadly classified into three categories, as summarized in Table 8: (1) encoder-only models, such as Moirai (Woo et al., 2024) and Moment (Goswami et al., 2024), which employ masked reconstruction and have been pre-trained on datasets containing 27B and 1B time points, respectively, with model sizes reaching up to 385M parameters; (2) encoder-decoder models, exemplified by Chronos (Ansari et al., 2024), which offers pre-trained models at four scales, with up to 710M parameters; and (3) decoder-only models, including TimesFM (Das et al., 2024), Lag-Llama (Rasul et al., 2023), and Timer (Liu et al., 2024d), with the largest models containing up to 200M parameters. The concurrent work, Moirai-MoE (Liu et al., 2024a), includes up to 935M parameters but with a different expert and routing design. In contrast to these models, TIME-MOE introduces a scalable, unified architecture with a sparse mixture-of-experts design, optimized for larger time series forecasting models while reducing inference costs. Trained on our Time-300B dataset, comprising over 300B time points, TIME-MOE is scaled to 2.4B parameters for the first time. It outperforms existing models with the same number of activated parameters, significantly enhancing both model efficiency and forecasting precision, while avoiding limitations such as fixed context lengths or hardcoded heuristics. 


Table 8: Comparison between large time series models.


<table><tr><td>Method</td><td>Time-MoE</td><td>Moirai</td><td>TimesFM</td><td>Moment</td><td>Chronos</td><td>Timer</td><td>Lag-Llama</td><td>TimeGPT</td></tr><tr><td>Architecture</td><td>Decoder-Only</td><td>Encoder-Only</td><td>Decoder-Only</td><td>Encoder-Only</td><td>Encoder-Decoder</td><td>Decoder-Only</td><td>Decoder-Only</td><td>Encoder-Decoder</td></tr><tr><td>(Max) Model Size</td><td>2.4B</td><td>311M</td><td>200M</td><td>385M</td><td>710M</td><td>67M</td><td>200M</td><td>Unknown</td></tr><tr><td>Input Token</td><td>Point</td><td>Patch</td><td>Patch</td><td>Patch</td><td>Point</td><td>Patch</td><td>Point</td><td>Patch</td></tr><tr><td>Dataset Scale</td><td>309B</td><td><eq>27B/231B^*</eq></td><td>100B</td><td>1.13B</td><td>84B</td><td>28B</td><td>0.36B</td><td>100B</td></tr><tr><td>Max Length</td><td><eq>4096^\dagger</eq></td><td>5000</td><td>512</td><td>512</td><td>512</td><td>1440</td><td>1024</td><td>Unknown</td></tr><tr><td>FFN</td><td>Sparse</td><td>Dense</td><td>Dense</td><td>Dense</td><td>Dense</td><td>Dense</td><td>Dense</td><td>Dense</td></tr><tr><td>Open-source Data</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>Source</td><td>Ours</td><td>Woo et al.</td><td>Das et al.</td><td>Goswami et al.</td><td>Ansari et al.</td><td>Liu et al.</td><td>Rasul et al.</td><td>Garza et al.</td></tr></table>


Depend on the way of calculation according to the original paper. <sup>†</sup> indicates the total of the context and prediction lengths. 


## B IMPLEMENTATION DETAILS

Training Configuration. Each model is trained for 100,000 steps with a batch size of 1,024, and a maximum sequence length capped at 4,096. This setup processes 4 million time points per iteration. We use forecast horizons of {1, 8, 32, 64} in the output projection and set the auxiliary loss factor α to 0.02. For optimization, we apply the AdamW optimizer with the following hyperparameters: lr = 1e-3, weight decay = 1e-1, $\beta _ { 1 } = 0 . 9$ , and $\beta _ { 2 } = 0 . 9 5$ . A learning rate scheduler with a linear warmup for the first 10,000 steps, followed by cosine annealing, is used. Training is performed on 128 × NVIDIA A100-80G GPUs with BF16 precision. To improve batch processing efficiency and handle varying sequence lengths, we employ sequence packing (Raffel et al., 2020), which reduces padding requirements. 

Benchmark Details. We evaluate the performance of various models for long-term forecasting across eight well-established datasets, including the Weather (Wu et al., 2021), Global Temp (Wu et al., 2023b), and ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2) (Zhou et al., 2021). A detailed description of each dataset is provided in Table 9. 


Table 9: Detailed dataset descriptions. Dataset sizes are listed as (Train, Validation, Test).


<table><tr><td>Tasks</td><td>Dataset</td><td>Dim</td><td>Series Length</td><td>Dataset Size</td><td>Frequency</td><td>Forecastability*</td><td>Information</td></tr><tr><td rowspan="6">Long-term Forecasting</td><td>ETTm1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.46</td><td>Temperature</td></tr><tr><td>ETTm2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(34465, 11521, 11521)</td><td>15min</td><td>0.55</td><td>Temperature</td></tr><tr><td>ETTh1</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Hourly</td><td>0.38</td><td>Temperature</td></tr><tr><td>ETTh2</td><td>7</td><td>{96, 192, 336, 720}</td><td>(8545, 2881, 2881)</td><td>Hourly</td><td>0.45</td><td>Temperature</td></tr><tr><td>Weather</td><td>21</td><td>{96, 192, 336, 720}</td><td>(36792, 5271, 10540)</td><td>10 min</td><td>0.75</td><td>Weather</td></tr><tr><td>Global Temp</td><td>1000</td><td>{96, 192, 336, 720}</td><td>(12280, 1755, 3509)</td><td>Hourly</td><td>0.78</td><td>Temperature</td></tr></table>


∗ The forecastability is calculated by one minus the entropy of Fourier decomposition of time series (Goerg, 2013). A larger value indicates better predictability. 


Metrics. We use mean square error (MSE) and mean absolute error (MAE) as evaluation metrics for time-series forecasting. These metrics are calculated as follows: 

$$
\mathrm{MSE} = \frac {1}{H} \sum_ {i = 1} ^ {H} (x _ {i} - \widehat {x} _ {i}) ^ {2},
$$

$$
\mathrm{MAE} = \frac {1}{H} \sum_ {i = 1} ^ {H} | x _ {i} - \widehat {x} _ {i} |,
$$

where $x _ { i } , { \widehat { x } } _ { i } \in \mathbb { R }$ are the ground truth and predictions of the i-th future time point. 

Technical Details. Our mixture-of-experts layer consists of one shared expert and several isolated experts, each represented by a feedforward network that is smaller than the standard FFN employed in dense models. In the formulation from Equations 5 to 8, $\mathrm { F F N } _ { N + 1 }$ denotes the shared expert, while FFN to FFN correspond to the isolated experts. The weight $g _ { N + 1 , t }$ associated with the shared expert for token t is normalized using the Sigmoid function. In contrast, the weight $g _ { i , t }$ for the i-th isolated expert of token t is normalized using the Softmax function. Furthermore, we retain only the top-k largest scores among the isolated experts and set the remaining scores to zero. 

To prevent routing collapse among experts, we adopt the strategy proposed by (Fedus et al., 2022), incorporating an auxiliary loss to ensure balanced expert load. The key aspect of this method is to penalize experts with high gating scores. This helps prevent a scenario where stronger experts, being exposed to more tokens, become even stronger while weaker experts continue to fall behind. The mathematical formulation is presented in Equation 10, where $f _ { i }$ represents the fraction of tokens assigned to expert $i ,$ and $r _ { i }$ denotes the proportion of router probability allocated to expert i. If one expert is assigned too many tokens and achieves a higher routing score, it will incur a correspondingly higher loss. 

Multi-resolution Forecasting. To construct the multiresolution forecasting head, we define $P$ output projections, each corresponding to a distinct forecasting horizon, denoted as $( p _ { 1 } , p _ { 2 } , \ldots , p _ { P } )$ . The output projection for horizon $p _ { j }$ is used to forecast the subsequent $p _ { j }$ time steps, as follows: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/5f1898a37554ad7cf4ee90b42b103334ca7145a744ec6a97c18d478a8df49ee2.jpg)



Figure 5: Causal attention layer.


$$
\hat {\mathbf {X}} _ {t + 1: t + p _ {j}} = \mathbf {W} _ {p _ {j}} \mathbf {h} _ {t} ^ {L},\tag{12}
$$

where $\mathbf { W } _ { p _ { i } } \in \mathbb { R } ^ { p _ { j } \times D }$ is the learnable parameter matrix for that horizon, and $\mathbf { h } _ { t } ^ { L }$ represents the output hidden state from the last MoE Transformer block. All output projections are optimized simultaneously during model training. 

During inference, we apply a greedy scheduling algorithm for arbitrary target output lengths H, as outlined in Algorithm 1. For each forecast operation in the auto-regressive process, we select a projection $p _ { j }$ with the closest forecasting horizon that does not exceed the remaining forecast duration. This approach allows TIME-MOE to extend predictions beyond the next immediate time step or fixed horizon, significantly improving both the model’s utility and overall forecasting accuracy. 

Algorithm 1 Scheduling for the Multi-resolution Forecasting
Require: Target output length H, forecast horizon of each output projection $\{p_{1}, p_{2}, \ldots, p_{P}\}$ in ascending order
Ensure: Combined output length $\hat{H} = H, p_{1} = 1$ 1: $\hat{H} \leftarrow 0$ 2: $J \leftarrow \{\}$ 3: while $\hat{H} < H$ do
4: for j = P down to 1 do
5: if $\hat{H} + p_{j} \leq H$ then
6: $\hat{H} \leftarrow \hat{H} + p_{j}$ 7: add $p_{j}$ to J
8: break
9: end if
10: end for
11: end while
12: return J 

## C PROCESSED DATA ARCHIVE

Going beyond the previous work (Ansari et al., 2024; Woo et al., 2024; Liu et al., 2024d), we organized a comprehensive large-scale time series dataset from a vast collection of complex raw data. We utilize the missing value ratio and the invalid observation ratio as metrics to assess the quality of the dataset. These two metrics can effectively identify data issues caused by the instability of data collection and artificially imputed values. The missing value ratio is defined as the proportion of ‘nan’ and ‘inf’ values present in the time series. Meanwhile, the invalid observation ratio refers to the maximum proportion of zeros in the first- or second-order differences of the time series. To address these issues and drawing inspiration from the data processing techniques of large language models (Penedo et al., 2023; Computer, 2023; Jin et al., 2024), we developed a fine-grained datacleaning pipeline specifically designed for time series data: 

Missing Value Processing. In time series data, missing values often appear as ‘nan’ (not a number) or ‘inf’ (infinity). While previous studies commonly address this by replacing missing values with the mean, this may distort the original time series pattern. Instead, we employ a method that splits the original sequence into multiple sub-sequences at points where missing values occur, effectively removing those segments while preserving the integrity of the original time series pattern. 

Invalid Observation Processing. In some data collection systems, missing values are often filled with 0 or another constant, leading to sequences with constant values that do not represent valid patterns for the model. To address this, we developed a filtering method that uses a fixed-length window to scan the entire sequence. For each window, we calculate the ratio of first-order and second-order differences, discarding the window if this ratio exceeds a pre-specified threshold (set to 0.2 in our case). The remaining valid continuous window sequences are then concatenated into a single sequence. This process transforms the original sequence into multiple sub-sequences, effectively removing segments with invalid patterns. 

Following the processing steps described above, we compiled a high-quality time series dataset named Time-300B, which spans a range of sampling frequencies from seconds to yearly intervals, encompassing a total of 309.09 billion time points. To optimize memory efficiency and loading speed, each dataset is split into multiple binary files, with a metafile providing details such as the start and end positions of each sequence. This setup allows us to load the data using a fixed amount of memory during training, preventing memory shortages. Datasets like Weatherbench, CMIP6, and ERA5 are particularly large, often leading to data imbalance and homogenization. To mitigate these issues, we apply down-sampling to these datasets. During training, we utilized approximately 117 billion time points in Time-300B, sampling each batch according to fixed proportions of domains and distributions of observation values. 

Below, we outline the key properties of the datasets after processing, including their domain, sampling frequency, number of time series, total number of observations, and data source. Also, we present the key component’s source code of the data-cleaning pipeline in Algorithm 2. 


Table 10: Datasets and key properties from Time-300B. For frequency: S = second, T = minute, H = hour, D = day, B = business day, W = week, M = month, Q = quarter, Y = year.


<table><tr><td>Dataset</td><td>Domain</td><td>Freq.</td><td># Time Series</td><td># Obs.</td><td>Source</td></tr><tr><td>Electricity (15 min)</td><td>Energy</td><td>15T</td><td>347</td><td>39,708,170</td><td>Godahewa et al. (2021)</td></tr><tr><td>Electricity (Weekly)</td><td>Energy</td><td>W</td><td>318</td><td>49,608</td><td>Godahewa et al. (2021)</td></tr><tr><td>ERCOT Load</td><td>Energy</td><td>H</td><td>152</td><td>1,238,832</td><td>ourownstory (2023)</td></tr><tr><td>Australian Electricity</td><td>Energy</td><td>30T</td><td>5</td><td>1,153,584</td><td>Godahewa et al. (2021)</td></tr><tr><td>Solar Power</td><td>Energy</td><td>4S</td><td>26</td><td>5,248</td><td>Godahewa et al. (2021)</td></tr><tr><td>Wind Farms</td><td>Energy</td><td>T</td><td>43,246</td><td>39,705,317</td><td>Godahewa et al. (2021)</td></tr><tr><td>BDG-2 Bear</td><td>Energy</td><td>H</td><td>215</td><td>1,422,320</td><td>Emami et al. (2023)</td></tr><tr><td>BDG-2 Fox</td><td>Energy</td><td>H</td><td>179</td><td>2,285,288</td><td>Emami et al. (2023)</td></tr><tr><td>BDG-2 Panther</td><td>Energy</td><td>H</td><td>136</td><td>893,840</td><td>Emami et al. (2023)</td></tr><tr><td>BDG-2 Rat</td><td>Energy</td><td>H</td><td>455</td><td>4,596,080</td><td>Emami et al. (2023)</td></tr><tr><td>Borealis</td><td>Energy</td><td>H</td><td>17</td><td>82,757</td><td>Emami et al. (2023)</td></tr><tr><td>Buildings900K</td><td>Energy</td><td>H</td><td>2,464,188</td><td>15,124,358,211</td><td>Emami et al. (2023)</td></tr><tr><td>BDG-2 Bull</td><td>Energy</td><td>H</td><td>464</td><td>501,832</td><td>Wang et al. (2023d)</td></tr><tr><td>BDG-2 Cockatoo</td><td>Energy</td><td>H</td><td>4</td><td>17032</td><td>Wang et al. (2023d)</td></tr><tr><td>Covid19 Energy</td><td>Energy</td><td>H</td><td>1</td><td>31,912</td><td>Wang et al. (2023d)</td></tr><tr><td>Elec demand</td><td>Energy</td><td>30T</td><td>1</td><td>17,520</td><td>Godahewa et al. (2021)</td></tr><tr><td>GEF12</td><td>Energy</td><td>H</td><td>20</td><td>788,280</td><td>Wang et al. (2023d)</td></tr><tr><td>GEF17</td><td>Energy</td><td>H</td><td>8</td><td>140,352</td><td>Wang et al. (2023d)</td></tr><tr><td>BDG-2 Hog</td><td>Energy</td><td>H</td><td>152</td><td>365,304</td><td>Wang et al. (2023d)</td></tr><tr><td>IDEAL</td><td>Energy</td><td>H</td><td>225</td><td>1,253,088</td><td>Emami et al. (2023)</td></tr><tr><td>KDD Cup 2018</td><td>Energy</td><td>H</td><td>3,054</td><td>922,746</td><td>Godahewa et al. (2021)</td></tr><tr><td>KDD Cup 2022</td><td>Energy</td><td>10T</td><td>8,554</td><td>2,332,874</td><td>Zhou et al. (2022a)</td></tr><tr><td>London Smart Meters</td><td>Energy</td><td>30T</td><td>24,132</td><td>160,041,727</td><td>Godahewa et al. (2021)</td></tr><tr><td>PDB</td><td>Energy</td><td>H</td><td>1</td><td>17,520</td><td>Wang et al. (2023d)</td></tr><tr><td>Residential Load Power</td><td>Energy</td><td>T</td><td>79,508</td><td>404,832,695</td><td>Bergmeir et al. (2023)</td></tr><tr><td>Residential PV Power</td><td>Energy</td><td>T</td><td>248,888</td><td>184,238,228</td><td>Bergmeir et al. (2023)</td></tr><tr><td>Sceaux</td><td>Energy</td><td>H</td><td>1</td><td>34,223</td><td>Emami et al. (2023)</td></tr><tr><td>SMART</td><td>Energy</td><td>H</td><td>5</td><td>95,709</td><td>Emami et al. (2023)</td></tr><tr><td>Spanish</td><td>Energy</td><td>H</td><td>1</td><td>35,064</td><td>Wang et al. (2023d)</td></tr><tr><td>Exchange Rate</td><td>Finance</td><td>B</td><td>13</td><td>56,096</td><td>Ansari et al. (2024)</td></tr><tr><td>CIF 2016</td><td>Finance</td><td>M</td><td>72</td><td>7,108</td><td>Godahewa et al. (2021)</td></tr><tr><td>Bitcoin</td><td>Finance</td><td>D</td><td>29</td><td>68927</td><td>Godahewa et al. (2021)</td></tr><tr><td>FRED MD</td><td>Finance</td><td>M</td><td>104</td><td>71,624</td><td>Godahewa et al. (2021)</td></tr><tr><td>NN5 Daily</td><td>Finance</td><td>D</td><td>220</td><td>35,303</td><td>Godahewa et al. (2021)</td></tr><tr><td>Tourism Monthly</td><td>Finance</td><td>M</td><td>359</td><td>98,867</td><td>Godahewa et al. (2021)</td></tr><tr><td>Tourism Quarterly</td><td>Finance</td><td>Q</td><td>427</td><td>39,128</td><td>Godahewa et al. (2021)</td></tr><tr><td>Tourism Yearly</td><td>Finance</td><td>Y</td><td>419</td><td>11,198</td><td>Godahewa et al. (2021)</td></tr><tr><td>COVID Deaths</td><td>Healthcare</td><td>D</td><td>2</td><td>364</td><td>Godahewa et al. (2021)</td></tr><tr><td>Hospital</td><td>Healthcare</td><td>M</td><td>727</td><td>55,224</td><td>Godahewa et al. (2021)</td></tr><tr><td>CDC Fluview ILINet</td><td>Healthcare</td><td>W</td><td>286</td><td>220,144</td><td>CDC (2017)</td></tr><tr><td>CDC Fluview WHO NREVSS</td><td>Healthcare</td><td>W</td><td>108</td><td>56,407</td><td>CDC (2017)</td></tr><tr><td>Project Tycho</td><td>Healthcare</td><td>W</td><td>588</td><td>120,183</td><td>van Panhuis et al. (2018)</td></tr><tr><td colspan="6">Table 10 continued from previous page</td></tr><tr><td>Dataset</td><td>Domain</td><td>Freq.</td><td># Time Series</td><td># Obs.</td><td>Source</td></tr><tr><td>US Births</td><td>Healthcare</td><td>D</td><td>1</td><td>7,275</td><td>Godahewa et al. (2021)</td></tr><tr><td>Weatherbench (Hourly)</td><td>Nature</td><td>H</td><td>3,984,029</td><td>74,630,250,518</td><td>Rasp et al. (2020)</td></tr><tr><td>Weatherbench (Daily)</td><td>Nature</td><td>D</td><td>301,229</td><td>3,223,513,345</td><td>Rasp et al. (2020)</td></tr><tr><td>Weatherbench (Weekly)</td><td>Nature</td><td>W</td><td>226,533</td><td>462,956,049</td><td>Rasp et al. (2020)</td></tr><tr><td>Beijing Air Quality</td><td>Nature</td><td>H</td><td>4,262</td><td>2,932,657</td><td>Chen (2019)</td></tr><tr><td>China Air Quality</td><td>Nature</td><td>H</td><td>17,686</td><td>4,217,605</td><td>Zheng et al. (2015)</td></tr><tr><td>CMIP6</td><td>Nature</td><td>6H</td><td>14,327,808</td><td>104,592,998,400</td><td>Nguyen et al. (2023)</td></tr><tr><td>ERA5</td><td>Nature</td><td>H</td><td>11,940,789</td><td>93,768,721,472</td><td>Nguyen et al. (2023)</td></tr><tr><td>Oikolab Weather</td><td>Nature</td><td>H</td><td>309</td><td>615,574</td><td>Godahewa et al. (2021)</td></tr><tr><td>Saugeen</td><td>Nature</td><td>D</td><td>38</td><td>17,311</td><td>Godahewa et al. (2021)</td></tr><tr><td>Subseasonal</td><td>Nature</td><td>D</td><td>17,604</td><td>51,968,498</td><td>Mouatadid et al. (2023)</td></tr><tr><td>Subseasonal Precipitation</td><td>Nature</td><td>D</td><td>13,467</td><td>4,830,284</td><td>Mouatadid et al. (2023)</td></tr><tr><td>Sunspot</td><td>Nature</td><td>D</td><td>19</td><td>45,312</td><td>Godahewa et al. (2021)</td></tr><tr><td>Temperature Rain</td><td>Nature</td><td>D</td><td>13,226</td><td>3,368,098</td><td>Godahewa et al. (2021)</td></tr><tr><td>Weather</td><td>Nature</td><td>D</td><td>9,525</td><td>26,036,234</td><td>Ansari et al. (2024)</td></tr><tr><td>Dominick</td><td>Sales</td><td>D</td><td>3,712</td><td>759,817</td><td>Godahewa et al. (2021)</td></tr><tr><td>Car Parts</td><td>Sales</td><td>M</td><td>16</td><td>816</td><td>Godahewa et al. (2021)</td></tr><tr><td>Favorita Sales</td><td>Sales</td><td>D</td><td>91,513</td><td>20,371,303</td><td>Woo et al. (2024)</td></tr><tr><td>Favorita Transactions</td><td>Sales</td><td>D</td><td>258</td><td>81,196</td><td>Woo et al. (2024)</td></tr><tr><td>Hierarchical Sales</td><td>Sales</td><td>D</td><td>215</td><td>114,372</td><td>Mancuso et al. (2021)</td></tr><tr><td>Restaurant</td><td>Sales</td><td>D</td><td>155</td><td>30,289</td><td>Woo et al. (2024)</td></tr><tr><td>M5</td><td>Sales</td><td>D</td><td>14,341</td><td>5,011,077</td><td>Alexandrov et al. (2020)</td></tr><tr><td>Mexico City Bikes</td><td>Transport</td><td>H</td><td>556</td><td>78,848</td><td>Ansari et al. (2024)</td></tr><tr><td>Traffic</td><td>Transport</td><td>H</td><td>1,371</td><td>14,993,544</td><td>Godahewa et al. (2021)</td></tr><tr><td>Taxi (Hourly)</td><td>Transport</td><td>H</td><td>2,433</td><td>1,762,024</td><td>Ansari et al. (2024)</td></tr><tr><td>Beijing Subway</td><td>Transport</td><td>30T</td><td>552</td><td>19,872</td><td>Wang et al. (2023a)</td></tr><tr><td>Covid Mobility</td><td>Transport</td><td>D</td><td>426</td><td>120,950</td><td>Godahewa et al. (2021)</td></tr><tr><td>HZMetro</td><td>Transport</td><td>15T</td><td>160</td><td>11,680</td><td>Wang et al. (2023a)</td></tr><tr><td>LargeST</td><td>Transport</td><td>5T</td><td>1,208,997</td><td>4,175,062,621</td><td>Liu et al. (2023)</td></tr><tr><td>Loop Seattle</td><td>Transport</td><td>5T</td><td>1,809</td><td>33,700,832</td><td>Wang et al. (2023a)</td></tr><tr><td>Los-Loop</td><td>Transport</td><td>5T</td><td>3,381</td><td>6,231,168</td><td>Wang et al. (2023a)</td></tr><tr><td>Pedestrian Counts</td><td>Transport</td><td>H</td><td>80</td><td>3,125,914</td><td>Godahewa et al. (2021)</td></tr><tr><td>PEMS Bay</td><td>Transport</td><td>5T</td><td>3,980</td><td>15,975,920</td><td>Wang et al. (2023a)</td></tr><tr><td>PEMS03</td><td>Transport</td><td>5T</td><td>1,651</td><td>9,210,432</td><td>Wang et al. (2023a)</td></tr><tr><td>PEMS04</td><td>Transport</td><td>5T</td><td>6,634</td><td>14,638,784</td><td>Wang et al. (2023a)</td></tr><tr><td>PEMS07</td><td>Transport</td><td>5T</td><td>3,828</td><td>23,789,760</td><td>Wang et al. (2023a)</td></tr><tr><td>PEMS08</td><td>Transport</td><td>5T</td><td>2,612</td><td>8,684,480</td><td>Wang et al. (2023a)</td></tr><tr><td>Q-Traffic</td><td>Transport</td><td>15T</td><td>46,990</td><td>257,200,384</td><td>Wang et al. (2023a)</td></tr><tr><td>SHMetro</td><td>Transport</td><td>15T</td><td>574</td><td>41,902</td><td>Wang et al. (2023a)</td></tr><tr><td>SZ-Taxi</td><td>Transport</td><td>15T</td><td>156</td><td>464,256</td><td>Wang et al. (2023a)</td></tr><tr><td>Rideshare</td><td>Transport</td><td>H</td><td>1,352</td><td>192,949</td><td>Godahewa et al. (2021)</td></tr><tr><td>Taxi</td><td>Transport</td><td>30T</td><td>96,758</td><td>40,584,636</td><td>Alexandrov et al. (2020)</td></tr><tr><td>Traffic Hourly</td><td>Transport</td><td>H</td><td>1,363</td><td>14,858,016</td><td>Godahewa et al. (2021)</td></tr><tr><td>Traffic Weekly</td><td>Transport</td><td>W</td><td>821</td><td>78,816</td><td>Godahewa et al. (2021)</td></tr><tr><td>Uber TLC Daily</td><td>Transport</td><td>D</td><td>235</td><td>42,533</td><td>Alexandrov et al. (2020)</td></tr><tr><td>Uber TLC Hourly</td><td>Transport</td><td>H</td><td>344</td><td>510,284</td><td>Alexandrov et al. (2020)</td></tr><tr><td>Vehicle Trips</td><td>Transport</td><td>D</td><td>10</td><td>1,626</td><td>Godahewa et al. (2021)</td></tr><tr><td>Wiki Daily (100k)</td><td>Web</td><td>D</td><td>100,001</td><td>274,099,872</td><td>Ansari et al. (2024)</td></tr><tr><td>Alibaba Cluster Trace 2018</td><td>Web</td><td>5T</td><td>48,640</td><td>83,776,950</td><td>Woo et al. (2023)</td></tr><tr><td>Azure VM Traces 2017</td><td>Web</td><td>5T</td><td>263,928</td><td>880,648,165</td><td>Woo et al. (2023)</td></tr><tr><td>Borg Cluster Data 2011</td><td>Web</td><td>5T</td><td>216,636</td><td>176,650,715</td><td>Woo et al. (2023)</td></tr><tr><td>Kaggle Web Traffic Weekly</td><td>Web</td><td>W</td><td>133,388</td><td>15,206,232</td><td>Godahewa et al. (2021)</td></tr><tr><td>Extended Web Traffic</td><td>Web</td><td>D</td><td>161,890</td><td>332,586,145</td><td>Godahewa et al. (2021)</td></tr><tr><td>Wiki-Rolling</td><td>Web</td><td>D</td><td>47,675</td><td>40,619,100</td><td>Alexandrov et al. (2020)</td></tr><tr><td>TSMixup 10M</td><td>Synthetic</td><td>-</td><td>10,968,625</td><td>8,198,358,952</td><td>Ansari et al. (2024)</td></tr><tr><td>KernelSynth 1M</td><td>Synthetic</td><td>-</td><td>1,000,000</td><td>1,024,000,000</td><td>Ansari et al. (2024)</td></tr><tr><td>M1 Monthly</td><td>Other</td><td>M</td><td>8</td><td>1,047</td><td>Godahewa et al. (2021)</td></tr><tr><td>M1 Quarterly</td><td>Other</td><td>3M</td><td>195</td><td>9,628</td><td>Godahewa et al. (2021)</td></tr><tr><td>M1 Yearly</td><td>Other</td><td>Y</td><td>106</td><td>3136</td><td>Godahewa et al. (2021)</td></tr><tr><td>M3 Monthly</td><td>Other</td><td>M</td><td>799</td><td>109,538</td><td>Godahewa et al. (2021)</td></tr><tr><td>M3 Quarterly</td><td>Other</td><td>3M</td><td>755</td><td>36,960</td><td>Godahewa et al. (2021)</td></tr><tr><td>M3 Yearly</td><td>Other</td><td>Y</td><td>645</td><td>18,319</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Daily</td><td>Other</td><td>D</td><td>4,134</td><td>9,903,554</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Hourly</td><td>Other</td><td>H</td><td>415</td><td>352,988</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Monthly</td><td>Other</td><td>M</td><td>30,126</td><td>8,480,953</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Quarterly</td><td>Other</td><td>3M</td><td>2,623</td><td>491,632</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Weekly</td><td>Other</td><td>W</td><td>293</td><td>348,224</td><td>Godahewa et al. (2021)</td></tr><tr><td>M4 Yearly</td><td>Other</td><td>Y</td><td>106</td><td>3,136</td><td>Godahewa et al. (2021)</td></tr></table>

## Algorithm 2 Code Snippet of Data-cleaning Pipline

```python
# Missing Value Processing
def split_seq_by_nan_inf(seq, minimum_seq_length: int = 1):
    output = []
    sublist = []
    for num in seq:
    if num is None or np.isnan(num) or np.isinf(num):
    if len(sublist) >= minimum_seq_length:
    output.append(sublist)
    sublist = []
    else:
    sublist.append(num)
    if len(sublist) >= minimum_seq_length:
    output.append(sublist)
    return output

# Invalid Observation Processing
def split_seq_by_window_quality(seq, window_size: int = 128, zero_threshold,
    minimum_seq_length: int = 256):
    if len(seq) <= window_size:
    flag, info = check_sequence(seq, zero_threshold=zero_threshold)
    if flag:
    return [seq]
    else:
    return []

    i = window_size
    sub_seq = []
    out_list = []

    while True:
    if i + window_size > len(seq):
    window_seq = seq[i - window_size: len(seq)]
    i = len(seq)
    else:
    window_seq = seq[i - window_size: i]
    flag, info = check_sequence(window_seq, zero_threshold=zero_threshold)
    if flag:
    sub_seq.extend(window_seq)
    else:
    if len(sub_seq) >= minimum_seq_length:
    out_list.append(sub_seq)
    sub_seq = []
    if i >= len(seq):
    break
    i += window_size

    if len(sub_seq) >= minimum_seq_length:
    out_list.append(sub_seq)

    return out_list

def check_sequence(seq, zero_threshold: float):
    import numpy as np
    if not isinstance(seq, np.ndarray):
    seq = np.array(seq)

    if len(seq.shape) > 1:
    raise RuntimeError(f'Dimension_of_the_seq_is_not_equal_to_1:{seq.shape}')
    flag = True
    info = {}

    nan_count = np.sum(np.isnan(seq))
    info['nan_count'] = nan_count
    if nan_count > 0:
    flag = False
    return flag, info

    inf_count = np.sum(np.isinf(seq))
    info['inf_count'] = inf_count
    if inf_count > 0:
    flag = False
    return flag, info

    zero_ratio = np.sum(seq == 0) / len(seq)
    info['zero_ratio'] = zero_ratio
    if zero_ratio > zero_threshold:
    flag = False

    first_diff = seq[1:] - seq[:-1]
    first_diff_zero_ratio = np.sum(first_diff == 0) / len(first_diff)

    info['first_diff_zero_ratio'] = first_diff_zero_ratio
    if first_diff_zero_ratio > zero_threshold:
    flag = False

    second_diff = seq[2:] - seq[:-2]
    second_diff_zero_ratio = np.sum(second_diff == 0) / len(second_diff)

    info['second_diff_zero_ratio'] = second_diff_zero_ratio
    if second_diff_zero_ratio > zero_threshold:
    flag = False

return flag, info 
```

## D ADDITIONAL RESULTS

## D.1 ABLATION STUDY


Table 11: MSE and MAE for horizon-96 forecasting across six benchmarks, evaluated with different model components.


<table><tr><td rowspan="2"></td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Weather</td><td colspan="2">Global Temp</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td><eq>TIME-MOE_{base}</eq></td><td>0.357</td><td>0.381</td><td>0.305</td><td>0.359</td><td>0.338</td><td>0.368</td><td>0.201</td><td>0.291</td><td>0.160</td><td>0.214</td><td>0.211</td><td>0.343</td></tr><tr><td>w/o Huber loss</td><td>0.365</td><td>0.383</td><td>0.309</td><td>0.366</td><td>0.344</td><td>0.369</td><td>0.205</td><td>0.295</td><td>0.163</td><td>0.221</td><td>0.217</td><td>0.359</td></tr><tr><td>w/o multi-resolution layer</td><td>0.358</td><td>0.379</td><td>0.313</td><td>0.362</td><td>0.348</td><td>0.377</td><td>0.212</td><td>0.301</td><td>0.164</td><td>0.219</td><td>0.217</td><td>0.354</td></tr><tr><td>w/o mixture-of-experts</td><td>0.370</td><td>0.398</td><td>0.317</td><td>0.372</td><td>0.347</td><td>0.373</td><td>0.212</td><td>0.298</td><td>0.163</td><td>0.218</td><td>0.223</td><td>0.357</td></tr><tr><td>w/o auxiliary loss</td><td>0.368</td><td>0.394</td><td>0.325</td><td>0.387</td><td>0.350</td><td>0.377</td><td>0.219</td><td>0.304</td><td>0.164</td><td>0.220</td><td>0.226</td><td>0.363</td></tr></table>

As shown in Table 11, replacing the MoE layers with standard FFNs (denoted as “w/o mixture-ofexperts ”) led to a noticeable performance decline, with the average MSE worsening from 0.262 to 0.272. This highlights the significant contribution of the sparse architecture to the model’s overall performance, as its dynamic routing enables more specialized processing of diverse input patterns. 

We also conducted experiments by retaining only the horizon-32 forecasting head from the TIME-$\mathbf { M O E _ { b a s e } }$ (denoted as “w/o multi-resolution layer”), excluding the multi-task optimization. The performance of this modified model was slightly inferior to the complete $\mathrm { T I M E - M O E } _ { \mathrm { b a s e } }$ 


Table 12: Full ablation results for different multi-resolution forecasting configurations.


<table><tr><td></td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Weather</td><td>Global Temp</td><td>Average MSE</td><td>Inference Speed</td></tr><tr><td>TIME-MOEbase w/ {1,8,32,64}</td><td>0.357</td><td>0.305</td><td>0.338</td><td>0.201</td><td>0.160</td><td>0.211</td><td>0.262</td><td>0.095 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1,8,32}</td><td>0.353</td><td>0.316</td><td>0.370</td><td>0.225</td><td>0.161</td><td>0.213</td><td>0.273</td><td>0.130 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1,8}</td><td>0.389</td><td>0.391</td><td>0.441</td><td>0.304</td><td>0.174</td><td>0.222</td><td>0.320</td><td>0.411 s/iter</td></tr><tr><td>TIME-MOEbase w/ {1}</td><td>1.071</td><td>0.920</td><td>2.098</td><td>2.320</td><td>1.500</td><td>0.383</td><td>1.382</td><td>2.834 s/iter</td></tr></table>

As shown in Table 12, the default configuration of four multi-resolution forecasting heads with receptive horizons of 1, 8, 32, 64 delivers optimal predictive performance and inference speed. Reducing the number of heads consistently resulted in decreased performance and longer inference time. This inverse relationship highlights the effectiveness of our multi-resolution forecasting design, striking a balance between accuracy and computational efficiency in a decoder-only forecasting foundation model. 

These findings highlight the importance of key architectural components in TIME-MOE, such as the mixture-of-experts, multi-task optimization, and multi-resolution forecasting, in delivering state-ofthe-art performance in universal time series forecasting. 

## D.2 TRAINING PRECISION ANALYSIS

To optimize model performance and efficiency, we conducted a comparative study examining the impact of numerical precision during training. We trained two versions of our model under identical configurations, with the only difference being the precision: one using bfloat16 and the other using float32. The model trained with float32 precision is referred to as $\mathrm { T I M E - M O E } _ { \mathrm { b a s e } }$ w/ FP32. 


Table 13: Full results of the comparison between BF16 and FP32 in terms of training and inference efficiency. FA denotes flash-attention.


<table><tr><td></td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td><td>Weather</td><td>Global Temp</td><td>Average MSE</td><td>Training Speed</td><td>Inference Speed</td><td>Training Memory</td><td>Inference Memory</td></tr><tr><td>TIME-MOEbase</td><td>0.357</td><td>0.305</td><td>0.338</td><td>0.201</td><td>0.160</td><td>0.211</td><td>0.262</td><td>0.84 s/iter</td><td>0.095 s/iter</td><td>1.77 GB</td><td>226.70 MB</td></tr><tr><td>TIME-MOEbase w/o FA</td><td>0.357</td><td>0.305</td><td>0.338</td><td>0.201</td><td>0.160</td><td>0.211</td><td>0.262</td><td>1.09 s/iter</td><td>0.118 s/iter</td><td>1.77 GB</td><td>226.70 MB</td></tr><tr><td>TIME-MOEbase w/ FP32</td><td>0.358</td><td>0.303</td><td>0.342</td><td>0.198</td><td>0.158</td><td>0.208</td><td>0.261</td><td>1.24 s/iter</td><td>0.133 s/iter</td><td>2.21 GB</td><td>453.41 MB</td></tr></table>

As detailed in Table $^ { 6 , }$ our analysis reveals that the forecasting performances of these two models are remarkably comparable. This finding is significant as it demonstrates that the use of reduced precision (e.g., bfloat16) does not compromise the predictive capabilities of our model. 

However, the similarities in performance belie the substantial differences in computational efficiency and resource utilization: 

• Training Speed: Notably, the bfloat16 model demonstrates a 12% improvement in training speed compared to its float32 counterpart. This considerable acceleration in the training process can significantly reduce the time-to-deployment for large-scale models and facilitate more rapid experimentation and iteration. 

• Memory Consumption: In terms of memory usage, the bfloat16 model exhibits superior efficiency, consuming substantially less memory than the float32 model. Specifically, we observed a reduction of 20% in memory usage. This memory optimization is crucial for scaling models to larger sizes or deploying them on memory-constrained hardware. 

• Compatibility with Advanced Techniques: A key advantage of the bfloat16 model is its seamless integration with advanced optimization techniques. In particular, it can easily be combined with flash-attention (Dao, 2024), a state-of-the-art attention mechanism designed for better efficiency. This integration results in an additional 23% increase in training speed and a 19% boost in inference speed, further enhancing the already significant performance gains. 

## The implications of these findings are far-reaching:

• Resource Efficiency: The reduced memory footprint and increased training speed of the bfloat16 model translate to more efficient utilization of computational resources, potentially lowering infrastructure costs and energy consumption. 

• Scalability: The memory savings offered by bfloat16 precision enable the training of larger, more complex models on the same hardware, potentially leading to improved model capabilities without increasing computational requirements. 

• Faster Development Cycles: The substantial improvements in training speed can accelerate the research and development process, allowing for more rapid prototyping and experimentation. 

• Inference Optimization: The compatibility with flash-attention not only benefits training but also enhances inference speed, which is crucial for real-time applications and large-scale deployments. 

Our experiments show that adopting bfloat16 precision, combined with advanced techniques like flash-attention, provides a compelling balance between model performance, computational efficiency, and resource utilization. These optimizations enable the scalable and efficient deployment of large-scale time series forecasting models without sacrificing predictive accuracy. 

## D.3 ADDITIONAL EXPERIMENTAL RESULTS

## D.3.1 TAXIBJ DATASET

We include a benchmark dataset, TaxiBJ (Zhang et al., 2017) for short-term forecasting evaluation. This original dataset encompasses taxicab GPS data and meteorological information collected from Beijing over four distinct intervals: July 1, 2013 - October 30, 2013; March 1, 2014 - June 30, 2014; March 1, 2015 - June 30, 2015; and November 1, 2015 - April 10, 2016. We selected the in-flow data from the period November 1, 2015, to April 10, 2016 as our benchmark. This benchmark dataset consists of 1,024 time-series sequences derived from 32 × 32 grid cells. 

We conducted evaluations on all zero-shot models using this benchmark, and set the context length to 512 for all baselines. The results are summarized in Table 14. 


Table 14: Short-term zero-shot forecasting results in TaxiBJ: A lower MSE or MAE indicates a better prediction. Red: the best, Blue: the 2nd best.


<table><tr><td rowspan="2" colspan="2">Models</td><td colspan="9">TIME-MOE (Ours)</td><td colspan="13">Zero-shot Time Series Models</td></tr><tr><td colspan="2"><eq>\text{TIME-MOE}_{\text{base}}</eq></td><td colspan="2"><eq>\text{TIME-MOE}_{\text{large}}</eq></td><td colspan="2"><eq>\text{TIME-MOE}_{\text{ultra}}</eq></td><td colspan="2"><eq>\text{Moirai}_{\text{small}}</eq></td><td colspan="2"><eq>\text{Moirai}_{\text{base}}</eq></td><td colspan="2"><eq>\text{Moirai}_{\text{large}}</eq></td><td colspan="2"><eq>\text{TimesFM}</eq></td><td colspan="2"><eq>\text{Moment}</eq></td><td colspan="2"><eq>\text{Chronos}_{\text{small}}</eq></td><td colspan="2"><eq>\text{Chronos}_{\text{base}}</eq></td><td colspan="2"><eq>\text{Chronos}_{\text{large}}</eq></td></tr><tr><td colspan="2">Metrics</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">TaxiBJ</td><td>1</td><td>0.214</td><td>0.294</td><td>0.214</td><td>0.292</td><td>0.214</td><td>0.294</td><td>0.334</td><td>0.373</td><td>0.282</td><td>0.334</td><td>0.267</td><td>0.323</td><td>0.247</td><td>0.316</td><td>0.866</td><td>0.751</td><td>0.250</td><td>0.315</td><td>0.255</td><td>0.316</td><td>0.250</td><td>0.303</td></tr><tr><td>8</td><td>0.302</td><td>0.363</td><td>0.297</td><td>0.356</td><td>0.302</td><td>0.362</td><td>0.487</td><td>0.470</td><td>0.427</td><td>0.422</td><td>0.431</td><td>0.425</td><td>0.393</td><td>0.430</td><td>0.883</td><td>0.759</td><td>0.341</td><td>0.380</td><td>0.311</td><td>0.352</td><td>0.310</td><td>0.351</td></tr><tr><td>24</td><td>0.385</td><td>0.419</td><td>0.376</td><td>0.410</td><td>0.385</td><td>0.417</td><td>0.610</td><td>0.529</td><td>0.530</td><td>0.477</td><td>0.548</td><td>0.488</td><td>0.494</td><td>0.495</td><td>0.894</td><td>0.764</td><td>0.438</td><td>0.440</td><td>0.427</td><td>0.420</td><td>0.431</td><td>0.418</td></tr><tr><td>48</td><td>0.423</td><td>0.448</td><td>0.414</td><td>0.440</td><td>0.422</td><td>0.444</td><td>0.626</td><td>0.542</td><td>0.559</td><td>0.497</td><td>0.563</td><td>0.500</td><td>0.524</td><td>0.515</td><td>0.892</td><td>0.765</td><td>0.502</td><td>0.478</td><td>0.475</td><td>0.450</td><td>0.494</td><td>0.460</td></tr></table>

The results indicate that our models consistently outperform other baselines in short-term forecasting on the TaxiBJ dataset. 

## D.3.2 COMPARISON TO TIMER, TFT, AND N-BEATS

In this section, we incorporate additional baseline models for a more comprehensive evaluation. Specifically, Timer (2024d) is included for zero-shot forecasting (Table 15), while TFT (2021) and N-BEATS (2020) are included for in-domain forecasting (Table 16). The results indicate that our models consistently demonstrate improved performance relative to these established approaches. 


Table 15: Additional zero-shot forecasting results of Timer, with 1B, 16B and 28B representing the scale of pretraining datasets as presented in the original paper Liu et al. (2024d). A lower MSE or MAE indicates a better prediction. Red: the best, Blue: the 2nd best.


<table><tr><td rowspan="3" colspan="2">Models Metrics</td><td colspan="6">TIME-MoE(Ours)</td><td colspan="6">Timer</td></tr><tr><td colspan="2">TIME-MoEbase</td><td colspan="2">TIME-MoElarge</td><td colspan="2">TIME-MoEultra</td><td colspan="2">1B</td><td colspan="2">16B</td><td colspan="2">28B</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.357</td><td>0.381</td><td>0.350</td><td>0.382</td><td>0.349</td><td>0.379</td><td>0.438</td><td>0.425</td><td>0.364</td><td>0.388</td><td>0.393</td><td>0.421</td></tr><tr><td>192</td><td>0.384</td><td>0.404</td><td>0.388</td><td>0.412</td><td>0.395</td><td>0.413</td><td>0.509</td><td>0.459</td><td>0.401</td><td>0.410</td><td>0.434</td><td>0.447</td></tr><tr><td>336</td><td>0.411</td><td>0.434</td><td>0.411</td><td>0.430</td><td>0.447</td><td>0.453</td><td>0.554</td><td>0.482</td><td>0.423</td><td>0.422</td><td>0.460</td><td>0.464</td></tr><tr><td>720</td><td>0.449</td><td>0.477</td><td>0.427</td><td>0.455</td><td>0.457</td><td>0.462</td><td>0.706</td><td>0.544</td><td>0.436</td><td>0.444</td><td>0.487</td><td>0.494</td></tr><tr><td>Avg.</td><td>0.400</td><td>0.424</td><td>0.394</td><td>0.419</td><td>0.412</td><td>0.426</td><td>0.552</td><td>0.478</td><td>0.406</td><td>0.416</td><td>0.444</td><td>0.456</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.305</td><td>0.359</td><td>0.302</td><td>0.354</td><td>0.292</td><td>0.352</td><td>0.315</td><td>0.351</td><td>0.294</td><td>0.350</td><td>0.308</td><td>0.369</td></tr><tr><td>192</td><td>0.351</td><td>0.386</td><td>0.364</td><td>0.385</td><td>0.347</td><td>0.379</td><td>0.393</td><td>0.402</td><td>0.353</td><td>0.385</td><td>0.348</td><td>0.398</td></tr><tr><td>336</td><td>0.391</td><td>0.418</td><td>0.417</td><td>0.425</td><td>0.406</td><td>0.419</td><td>0.412</td><td>0.422</td><td>0.376</td><td>0.400</td><td>0.366</td><td>0.414</td></tr><tr><td>720</td><td>0.419</td><td>0.454</td><td>0.537</td><td>0.496</td><td>0.439</td><td>0.447</td><td>0.425</td><td>0.440</td><td>0.393</td><td>0.420</td><td>0.409</td><td>0.446</td></tr><tr><td>Avg.</td><td>0.366</td><td>0.404</td><td>0.405</td><td>0.415</td><td>0.371</td><td>0.399</td><td>0.386</td><td>0.404</td><td>0.354</td><td>0.389</td><td>0.358</td><td>0.407</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.338</td><td>0.368</td><td>0.309</td><td>0.357</td><td>0.281</td><td>0.341</td><td>0.690</td><td>0.526</td><td>0.766</td><td>0.549</td><td>0.420</td><td>0.418</td></tr><tr><td>192</td><td>0.353</td><td>0.388</td><td>0.346</td><td>0.381</td><td>0.305</td><td>0.358</td><td>0.757</td><td>0.560</td><td>0.755</td><td>0.553</td><td>0.467</td><td>0.445</td></tr><tr><td>336</td><td>0.381</td><td>0.413</td><td>0.373</td><td>0.408</td><td>0.369</td><td>0.395</td><td>0.832</td><td>0.594</td><td>0.765</td><td>0.561</td><td>0.502</td><td>0.467</td></tr><tr><td>720</td><td>0.504</td><td>0.493</td><td>0.475</td><td>0.477</td><td>0.469</td><td>0.472</td><td>0.883</td><td>0.627</td><td>0.752</td><td>0.565</td><td>0.558</td><td>0.499</td></tr><tr><td>Avg.</td><td>0.394</td><td>0.415</td><td>0.376</td><td>0.405</td><td>0.356</td><td>0.391</td><td>0.791</td><td>0.577</td><td>0.760</td><td>0.557</td><td>0.487</td><td>0.457</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.201</td><td>0.291</td><td>0.197</td><td>0.286</td><td>0.198</td><td>0.288</td><td>0.213</td><td>0.295</td><td>0.234</td><td>0.312</td><td>0.247</td><td>0.324</td></tr><tr><td>192</td><td>0.258</td><td>0.334</td><td>0.250</td><td>0.322</td><td>0.235</td><td>0.312</td><td>0.283</td><td>0.339</td><td>0.287</td><td>0.343</td><td>0.294</td><td>0.358</td></tr><tr><td>336</td><td>0.324</td><td>0.373</td><td>0.337</td><td>0.375</td><td>0.293</td><td>0.348</td><td>0.346</td><td>0.377</td><td>0.340</td><td>0.373</td><td>0.335</td><td>0.385</td></tr><tr><td>720</td><td>0.488</td><td>0.464</td><td>0.480</td><td>0.461</td><td>0.427</td><td>0.428</td><td>0.424</td><td>0.424</td><td>0.437</td><td>0.426</td><td>0.386</td><td>0.418</td></tr><tr><td>Avg.</td><td>0.317</td><td>0.365</td><td>0.316</td><td>0.361</td><td>0.288</td><td>0.344</td><td>0.317</td><td>0.359</td><td>0.324</td><td>0.364</td><td>0.316</td><td>0.371</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.160</td><td>0.214</td><td>0.159</td><td>0.213</td><td>0.157</td><td>0.211</td><td>0.181</td><td>0.232</td><td>0.203</td><td>0.255</td><td>0.243</td><td>0.283</td></tr><tr><td>192</td><td>0.210</td><td>0.260</td><td>0.215</td><td>0.266</td><td>0.208</td><td>0.256</td><td>0.234</td><td>0.284</td><td>0.254</td><td>0.296</td><td>0.288</td><td>0.320</td></tr><tr><td>336</td><td>0.274</td><td>0.309</td><td>0.291</td><td>0.322</td><td>0.255</td><td>0.290</td><td>0.297</td><td>0.332</td><td>0.313</td><td>0.336</td><td>0.323</td><td>0.345</td></tr><tr><td>720</td><td>0.418</td><td>0.405</td><td>0.415</td><td>0.400</td><td>0.405</td><td>0.397</td><td>0.364</td><td>0.380</td><td>0.408</td><td>0.395</td><td>0.362</td><td>0.374</td></tr><tr><td>Avg.</td><td>0.265</td><td>0.297</td><td>0.270</td><td>0.300</td><td>0.256</td><td>0.288</td><td>0.269</td><td>0.307</td><td>0.294</td><td>0.321</td><td>0.304</td><td>0.331</td></tr><tr><td rowspan="5">Global Temp</td><td>96</td><td>0.211</td><td>0.343</td><td>0.210</td><td>0.342</td><td>0.214</td><td>0.345</td><td>0.250</td><td>0.373</td><td>0.245</td><td>0.372</td><td>0.308</td><td>0.425</td></tr><tr><td>192</td><td>0.257</td><td>0.386</td><td>0.254</td><td>0.385</td><td>0.246</td><td>0.379</td><td>0.299</td><td>0.415</td><td>0.300</td><td>0.418</td><td>0.359</td><td>0.465</td></tr><tr><td>336</td><td>0.281</td><td>0.405</td><td>0.267</td><td>0.395</td><td>0.266</td><td>0.398</td><td>0.347</td><td>0.451</td><td>0.365</td><td>0.466</td><td>0.415</td><td>0.507</td></tr><tr><td>720</td><td>0.354</td><td>0.465</td><td>0.289</td><td>0.420</td><td>0.288</td><td>0.421</td><td>0.452</td><td>0.521</td><td>0.542</td><td>0.585</td><td>0.579</td><td>0.617</td></tr><tr><td>Avg.</td><td>0.275</td><td>0.400</td><td>0.255</td><td>0.385</td><td>0.253</td><td>0.385</td><td>0.337</td><td>0.440</td><td>0.363</td><td>0.460</td><td>0.415</td><td>0.504</td></tr><tr><td colspan="2">Average</td><td>0.336</td><td>0.384</td><td>0.336</td><td>0.380</td><td>0.322</td><td>0.372</td><td>0.394</td><td>0.406</td><td>0.378</td><td>0.393</td><td>0.344</td><td>0.391</td></tr></table>


Table 16: Additional results of in-domain forecasting baselines. A lower MSE or MAE indicates a better prediction. Red: the best, Blue: the 2nd best.


<table><tr><td rowspan="3" colspan="2">Models Metrics</td><td colspan="6">TIME-MOE (Ours)</td><td colspan="4">Full-shot Time Series Models</td></tr><tr><td colspan="2">TIME-MOEbase</td><td colspan="2">TIME-MOElarge</td><td colspan="2">TIME-MOEultra</td><td colspan="2">TFT</td><td colspan="2">N-BEATS</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>96</td><td>0.345</td><td>0.373</td><td>0.335</td><td>0.371</td><td>0.323</td><td>0.365</td><td>0.478</td><td>0.476</td><td>0.383</td><td>0.405</td></tr><tr><td>192</td><td>0.372</td><td>0.396</td><td>0.374</td><td>0.400</td><td>0.359</td><td>0.391</td><td>0.510</td><td>0.486</td><td>0.453</td><td>0.447</td></tr><tr><td>336</td><td>0.389</td><td>0.412</td><td>0.390</td><td>0.412</td><td>0.388</td><td>0.418</td><td>0.548</td><td>0.505</td><td>0.517</td><td>0.493</td></tr><tr><td>720</td><td>0.410</td><td>0.443</td><td>0.402</td><td>0.433</td><td>0.425</td><td>0.450</td><td>0.549</td><td>0.515</td><td>0.594</td><td>0.546</td></tr><tr><td>Avg.</td><td>0.379</td><td>0.406</td><td>0.375</td><td>0.404</td><td>0.373</td><td>0.406</td><td>0.521</td><td>0.496</td><td>0.487</td><td>0.473</td></tr><tr><td rowspan="5">ETTh2</td><td>96</td><td>0.276</td><td>0.340</td><td>0.278</td><td>0.335</td><td>0.274</td><td>0.338</td><td>0.352</td><td>0.387</td><td>0.362</td><td>0.384</td></tr><tr><td>192</td><td>0.331</td><td>0.371</td><td>0.345</td><td>0.373</td><td>0.330</td><td>0.370</td><td>0.429</td><td>0.432</td><td>0.413</td><td>0.430</td></tr><tr><td>336</td><td>0.373</td><td>0.402</td><td>0.384</td><td>0.402</td><td>0.362</td><td>0.396</td><td>0.461</td><td>0.460</td><td>0.430</td><td>0.448</td></tr><tr><td>720</td><td>0.404</td><td>0.431</td><td>0.437</td><td>0.437</td><td>0.370</td><td>0.417</td><td>0.475</td><td>0.473</td><td>0.554</td><td>0.530</td></tr><tr><td>Avg.</td><td>0.346</td><td>0.386</td><td>0.361</td><td>0.386</td><td>0.334</td><td>0.380</td><td>0.429</td><td>0.438</td><td>0.440</td><td>0.448</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.286</td><td>0.334</td><td>0.264</td><td>0.325</td><td>0.256</td><td>0.323</td><td>0.468</td><td>0.444</td><td>0.334</td><td>0.372</td></tr><tr><td>192</td><td>0.307</td><td>0.358</td><td>0.295</td><td>0.350</td><td>0.281</td><td>0.343</td><td>0.557</td><td>0.488</td><td>0.379</td><td>0.401</td></tr><tr><td>336</td><td>0.354</td><td>0.390</td><td>0.323</td><td>0.376</td><td>0.326</td><td>0.374</td><td>0.682</td><td>0.528</td><td>0.421</td><td>0.425</td></tr><tr><td>720</td><td>0.433</td><td>0.445</td><td>0.409</td><td>0.435</td><td>0.454</td><td>0.452</td><td>0.722</td><td>0.565</td><td>0.476</td><td>0.471</td></tr><tr><td>Avg.</td><td>0.345</td><td>0.381</td><td>0.322</td><td>0.371</td><td>0.329</td><td>0.373</td><td>0.607</td><td>0.506</td><td>0.403</td><td>0.417</td></tr><tr><td rowspan="5">ETTm2</td><td>96</td><td>0.172</td><td>0.265</td><td>0.169</td><td>0.259</td><td>0.183</td><td>0.273</td><td>0.223</td><td>0.295</td><td>0.208</td><td>0.283</td></tr><tr><td>192</td><td>0.228</td><td>0.306</td><td>0.223</td><td>0.295</td><td>0.223</td><td>0.301</td><td>0.281</td><td>0.329</td><td>0.344</td><td>0.372</td></tr><tr><td>336</td><td>0.281</td><td>0.345</td><td>0.293</td><td>0.341</td><td>0.278</td><td>0.339</td><td>0.364</td><td>0.373</td><td>0.354</td><td>0.383</td></tr><tr><td>720</td><td>0.403</td><td>0.424</td><td>0.451</td><td>0.433</td><td>0.425</td><td>0.424</td><td>0.475</td><td>0.435</td><td>0.460</td><td>0.455</td></tr><tr><td>Avg.</td><td>0.271</td><td>0.335</td><td>0.284</td><td>0.332</td><td>0.277</td><td>0.334</td><td>0.336</td><td>0.358</td><td>0.342</td><td>0.373</td></tr><tr><td rowspan="5">Weather</td><td>96</td><td>0.151</td><td>0.203</td><td>0.149</td><td>0.201</td><td>0.154</td><td>0.208</td><td>0.186</td><td>0.231</td><td>0.165</td><td>0.224</td></tr><tr><td>192</td><td>0.195</td><td>0.246</td><td>0.192</td><td>0.244</td><td>0.202</td><td>0.251</td><td>0.240</td><td>0.275</td><td>0.209</td><td>0.269</td></tr><tr><td>336</td><td>0.247</td><td>0.288</td><td>0.245</td><td>0.285</td><td>0.252</td><td>0.287</td><td>0.302</td><td>0.317</td><td>0.261</td><td>0.310</td></tr><tr><td>720</td><td>0.352</td><td>0.366</td><td>0.352</td><td>0.365</td><td>0.392</td><td>0.376</td><td>0.388</td><td>0.369</td><td>0.336</td><td>0.362</td></tr><tr><td>Avg.</td><td>0.236</td><td>0.275</td><td>0.234</td><td>0.273</td><td>0.250</td><td>0.280</td><td>0.279</td><td>0.298</td><td>0.243</td><td>0.291</td></tr><tr><td rowspan="5">Global Temp</td><td>96</td><td>0.192</td><td>0.328</td><td>0.192</td><td>0.329</td><td>0.189</td><td>0.322</td><td>0.260</td><td>0.390</td><td>0.210</td><td>0.344</td></tr><tr><td>192</td><td>0.238</td><td>0.375</td><td>0.236</td><td>0.375</td><td>0.234</td><td>0.376</td><td>0.301</td><td>0.423</td><td>0.253</td><td>0.385</td></tr><tr><td>336</td><td>0.259</td><td>0.397</td><td>0.256</td><td>0.397</td><td>0.253</td><td>0.399</td><td>0.359</td><td>0.464</td><td>0.282</td><td>0.411</td></tr><tr><td>720</td><td>0.345</td><td>0.465</td><td>0.322</td><td>0.451</td><td>0.292</td><td>0.426</td><td>0.371</td><td>0.477</td><td>0.342</td><td>0.457</td></tr><tr><td>Avg.</td><td>0.258</td><td>0.391</td><td>0.251</td><td>0.388</td><td>0.242</td><td>0.380</td><td>0.323</td><td>0.439</td><td>0.272</td><td>0.399</td></tr><tr><td colspan="2">Average</td><td>0.306</td><td>0.362</td><td>0.304</td><td>0.359</td><td>0.301</td><td>0.358</td><td>0.416</td><td>0.422</td><td>0.364</td><td>0.400</td></tr></table>

## E FORECAST SHOWCASES

To visualize the performance differences among various time series foundation models, we present the forecasting results of our model, TIME-MOE, in comparison to the ground truth across six realworld benchmarks. These benchmarks include ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Global Temp datasets. Alongside TIME-MOE’s results, we also show the performance of other foundation models at different scales, providing a comprehensive view of their comparative capabilities (Figures 6 – 11). In all figures, the context length is set to 512, and the forecast horizon is 96. To enhance clarity and aesthetics, we display the full forecast output, complemented by a portion of the preceding historical input data, ensuring a more intuitive comparison. 

The results clearly demonstrate the superiority of TIME-MOE over the other foundational models. Its ability to consistently produce more accurate forecasts across a range of datasets underscores the effectiveness of its architecture and design. The performance gains are especially noticeable in long-term prediction scenarios, where TIME-MOE’s handling of temporal dependencies proves more robust than its counterparts. These visual comparisons highlight the practical advantages of TIME-MOE in large-scale time series forecasting, reinforcing its status as a state-of-the-art model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/1009b82e3b95d13b4ee674faddaa42a4d225400ae2f3b4c2e0a01fb8e0419121.jpg)



(a) Time−MoE<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/382d65140c7074fcf786430b1ac15dd95c75677eb9ddeb791184eafbfc795db7.jpg)



(b) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/aceac814d60057b8011a2952f59a89fa8d28ddc784934873b3ac4aeed6aaf6ed.jpg)



(c) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/0f1f888e5a78a13d95bb671e723f44525ac5c6bccb95d3f3ee9c13b4efe39b1d.jpg)



(d) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/191bf582ab8dc1e543f71309dae8cc8b97486835b48bb1c58c7db9f16217c41a.jpg)



(e) Moirai<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/d3057b1480108f2f31b2bde0893c42a2c5008f024e260c7f7c1b855548b3a3e0.jpg)



(f) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/207e40391a40f3f6e4b83239d1f6f6afb932f462400c3d45b99d9c9412c7ab43.jpg)



(g) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/0c62e25fc7afc2a33b8d20deec8f6a3c97d061901c935178c2514c8a8b5e3507.jpg)



(h) Chronos<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c9c4fc63fd163de354b762737b6884578fc9cc5953a70b9c371abaf765fd8378.jpg)



(i) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/283c08d55844e1a077f56a05a99d8986a77b1a85a25456749fd123f8a62a7ed4.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/85f3ab7581ffcdec77eaeda9dd7d73f16f0a22467ba8c983ac300d7384a6ee75.jpg)



(k) TimesFM



Figure 6: Zero-shot forecasting cases from ETTh1 by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/5f57bae9afef7b993819f1f86b02b2832592ada6011e809b51c33e9db0f880fa.jpg)



(a) Time−MoE<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/e1e99f133666c5617fd5baad4e7c55e270d7b38d8732ae2a53379ce7c47fd129.jpg)



(b) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/d991f98221e2e81b7fd78de0cedbfe2cf987da6cb4a7fc59462fe4b90ece358b.jpg)



(c) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/ba0b67d83e4366a5e29250a68c1372960f7bce406f846b7db1d28b846ba1cc45.jpg)



(d) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/ecbef08eb785a0ecea0379d086807a352b579e03dd4e8aee272309be38fdc4ba.jpg)



(e) Moirai<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/9b2ffc0eb818f7206ae992eb53ad5cb13cf9a1625decfad87939672cd109d712.jpg)



(f) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/054207315eeecf138dcf02aa37d271245ea1288bd1b1d3b5b7933f52772bf4e1.jpg)



(g) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/65ace76f5656be974dcc4c6dc97e0ad80f6cf1f105f9b2a2af7ffb15f9060eef.jpg)



(h) Chronos<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c9bf0da5850e5fed5d6520cbbab8053d421059081385295e573359d9b19d7c00.jpg)



(i) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/7997d65a2a4bacf84bfe7917c3bd9e2ffb82afdb52a4ee53f8240253ca92b13c.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/6f9ab5dfd77d81e036db150bff21fe475b07f261675e248e3ea8fab809bfd771.jpg)



(k) TimesFM



Figure 7: Zero-shot forecasting cases from ETTh2 by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/4cd71d081b4b99bae5c533a0d0afe4a35010ca0149b9ddcda12f5a610315a367.jpg)



(a) Time−MoE<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/f1176dfab7a11d3800e0a0ac7511a8fb9d3e7abb5f5871b54c40ad1fc1e0af90.jpg)



(b) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/eb3c38ee70c19a0218d3df5159487a7f06bc42389bbc4d6d6d13424334eb0808.jpg)



(c) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c397c2d46003f493f836c1e4c701ff60463cf6a6fec1c3a1411073d235071559.jpg)



(d) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/eff82135363cb035a60ada2e8ac7773ab577a13db99b24e0ca9186d2dca01570.jpg)



(e) Moirai<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/98973e09b43f30d83ff4a6dfb1316b9122064828eb4494b4506c517659838d8d.jpg)



(f) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/344de84c2bad5bbf18c57c02888aaa6306625da8765b8304c43e6316a768320f.jpg)



(g) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/645fad97d7b82d0d5b5cf379257a51aa69ba29705361a43d020415aa2ecac276.jpg)



(h) Chronos<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/264d9806261e51f1f080edc672661705a402b7d309feac5d22ff3f17b0a33b16.jpg)



(i) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/abf67c10d1e49fdd1ab9440e99977b810be7b5e10b810c8ec1552b23b7e5def7.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/e2ab73e09e0ae6e4cc31ef42ba3c51a4c64801b7b3c2cfeb2fb88f2de5d0d9c0.jpg)



(k) TimesFM



Figure 8: Zero-shot forecasting cases from ETTm1 by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/a596bfcf74e05f8fcb45a568c75ee4176d52ba2068b599b2b7be7ebb1f138615.jpg)



(a) Time−MoE<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/9f03aa52c5beba976e8b1cac2a70ac3cfc76e9570ec34e9f7bce2e8f43fd3085.jpg)



(b) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/51f88152b207c3584024ffa44a63a34943d5148c9ad623ff16b6022207dd464c.jpg)



(c) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c70cf0ffb96bc40a277e15503489624510e14e4e3d65fd9fec2166652b1847dd.jpg)



(d) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/b76d3810757bd5e8631dd0067a8aea5e6d5c8dea3b8db68afe63a05e6ed425ff.jpg)



(e) Moirai<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/39666b8093eebd7a7a86083f4eb2410698dfe3c86b3a86fc496b9a5eb9c3281c.jpg)



(f) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/3bd19422ef8a55a2015b4b7da3a6b93a4f043aac3302513067dac7f7bd2f1a18.jpg)



(g) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/042331adf15e631ed85a7d3057270d845d1aa0d3767b0db0a730bf3fd412e2de.jpg)



(h) Chronos<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/dce921392ee6e2f18bafd1fde98a715e5100131dc9734e7773d793fa46886b7d.jpg)



(i) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/9f415cc15726be00660371aaccf6e62aaae7bb9f52e14739702704fe8ee63bf9.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/a54dc358154819c451f947fe6fb7880d12edbeca34e95c911a799544fcbf5700.jpg)



(k) TimesFM



Figure 9: Zero-shot forecasting cases from ETTm2 by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/64a72057f234cf055d5b39cbb5708dd2559d4ee52b2c68e495d7b617340a8af7.jpg)



(a) Time−MoE<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/477591f5a43ebbe62fa2244ddd0285fbd9aabbf1e83330d585d2a669d90424c3.jpg)



(b) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/90529756a302d851158be310cf02e50592e018147035b734c473d2f155063dc4.jpg)



(c) Time−MoE<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/e21604de7085f0d2d601d1ac117080571d9a153feac035fc5cc9a9963447b90d.jpg)



(d) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c1941e8ac432df7e5b08c8563c8e0ada274b9f335b8e81031706dd3cb2d7434f.jpg)



(e) Moirai<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/67911a291bf0c0918032b14ec35ce77e14b71b922e08a7520352c4d42e7616fe.jpg)



(f) Moirai<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/78795618651522356a98b969cf42388de114bfd1c6061127b8a9caf383f32a55.jpg)



(g) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/9d169689a163e02f1f6de9aa057a4272b10b0a3449db951547f7d61a79d96f62.jpg)



(h) Chronos<sub>����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/039b1a0b1a0247129fcb1e3bbc85960a6831313a3f5dab94fd1a4a68ec1207ef.jpg)



(i) Chronos<sub>�����</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/972b953b7dcd589b48000e47cd541bc5eb04a04bcddd26f1678969cc13ba291b.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/7afd2237d2e508fc16e640476860c751a7751f7f9d736f81a7465cd75a586c4b.jpg)



(k) TimesFM



Figure 10: Zero-shot forecasting cases from Weather by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c839396f521425ca9003e1ee94dd35165c60c07dc0d5f433f30ce60fd519d70a.jpg)



(a) Time−MoE<sub>!"#$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/c6d04fbf2b2eca9bd2c0e248a92e3d56d71c07fecdc2d3517009a2fded4c9db1.jpg)



(b) Time−MoE<sub>%"&'$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/f93e1a80ae123a0fea02813c61741a34e0fd2875bc0d0ab2c2279faf5761f4bb.jpg)



(c) Time−MoE<sub>(%)&"</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/145299613784457490d95d5732048e104c61fa8c378d6e67040ae496a4fddba3.jpg)



(d) Moirai<sub>#*"%%</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/536a492f8003fa160f01b80d4b7662491bb10bf96e99f66b00522c9e221e1028.jpg)



(e) Moirai<sub>!"#$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/00e3d91c7e88b2f9bd4fbbd694976c482ba303727ebba562a5d92a8eeeb98409.jpg)



(f) Moirai<sub>%"&'$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/98fbb89f9a19478d4a32e9311fd83ab5b783979f22383a36b575b6ad787a85e8.jpg)



(g) Chronos<sub>#*"%%</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/dc3e2c1d79cafd46cad6c09953aa602ce1154e6c9d18331e8f3c95d0d0581dde.jpg)



(h) Chronos<sub>!"#$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/f4719819d54d1317d11654ce2ec798b6729c7970117281d87e98824f6fe337f3.jpg)



(i) Chronos<sub>%"&'$</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/958ca906a7e3733e4cf710c760abc8cc0e624cd584ae51638e24e7e50ff7c875.jpg)



(j) Moment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-13/eed035e0-aa1c-482b-806f-67d5c2fc108b/3823f5c2c12249dc76fc836459abf00d5ae8dcda24d2c3d6bb0123a249bb3cb6.jpg)



(k) TimesFM



Figure 11: Zero-shot forecasting cases from Global Temp by different models, with forecast horizon 96. Blue lines are the ground truths and read lines are the model predictions.
