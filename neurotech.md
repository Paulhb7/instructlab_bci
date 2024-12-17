## NEURO-GPT: TOWARDS A FOUNDATION MODEL FOR EEG

Wenhui Cui $^{1}$, Woojae Jeong$^{1}$, Philipp Tholke$^{2}$, Takfarinas Medani$^{1}$, Karim Jerbi 2 , 3 , $^{4}$, Anand A. Joshi$^{1}$, Richard M. Leahy1

1 Ming Hsieh Department of Electrical and Computer Engineering, University of Southern California, Los Angeles, CA, USA

2 Psychology Department, Universit' e de Montr' eal, Montreal, QC, Canada 3 Mila (Quebec AI research institute), Montreal, QC, Canada, 4 UNIQUE (Quebec Neuro-AI research center), QC, Canada

## ABSTRACT

To handle the scarcity and heterogeneity of electroencephalography (EEG) data for Brain-Computer Interface (BCI) tasks, and to harness the power of large publicly available data sets, we propose Neuro-GPT, a foundation model consisting of an EEG encoder and a GPT model. The foundation model is pre-trained on a large-scale data set using a self-supervised task that learns how to reconstruct masked EEG segments. We then fine-tune the model on a motor imagery classification task to validate its performance in a low-data regime (9 subjects). Our experiments demonstrate that applying a foundation model can significantly improve classification performance compared to a model trained from scratch, which provides evidence for the generalizability of the foundation model and its ability to address challenges of data scarcity and heterogeneity in EEG. The code is publicly available at https://github.com/wenhui0206/NeuroGPT .

Index Terms -Foundation Model, EEG, GPT, Encoder

## 1. INTRODUCTION

The limited scale of training data for electroencephalography (EEG) based Brain-Computer Interface (BCI) classification tasks poses challenges to applying deep learning models. These models require a large amount of training data to converge and generalize to unseen testing data. However, individual differences can lead to heterogeneous feature representations across subjects [1], which makes it difficult to generalize the model across subjects. EEG's high-dimensional nature and limited availability for specific tasks create additional barriers to the convergence of these models.

One common approach is to learn generalizable features from large amounts of data using self-supervised learning and then transfer to the task of interest [2]. Here, we address the question of whether we can train a model on large-scale EEG datasets using a self-supervised task and then transfer the pretrained knowledge to enhance performance on a downstream

task. Large language models (LLMs) in natural language processing (NLP) tasks have proven extraordinarily successful using this approach. Similar models have also shown remarkable performance in other tasks including image and video generation [3], medical question answering [4], and neural activity analysis [5, 6, 7]. Despite the popularity of LLMs, there have been relatively few attempts to adapt them to EEG data. The work in [8] employed a BERT-inspired [9] approach to pre-train a transformer model on massive EEG data using a contrastive self-supervised task. However, it exhibited limited generalizability to downstream tasks.

Here we aim to lay the groundwork for developing foundation models for EEG. We employ a Generative Pre-trained Transformer (GPT) model [10], which uses a decoder-only transformer architecture and is trained to predict the next masked token given a sequence of tokens as input (autoregressive training). In text-based tasks, a sentence is broken down into tokens as input units to the model. A token can be a few characters or a word, depending on the language and context. To adapt the GPT model to EEG data, we split the whole time series into fixed-length "chunks", treating each chunk as a token. An EEG encoder is incorporated to extract representative features from raw EEG data. NeuroGPT is a foundation model consisting of an EEG encoder to extract spatio-temporal features from EEG data, and a GPT model that uses self-supervision to predict the masked chunks. The foundation model is pre-trained on the TUH EEG dataset [11]. We fine-tune the model on a motor imagery classification task where only 9 subjects are available. Experiments showed that the EEG encoder learns meaningful features that are generalizable to the downstream task.

## 2. METHODS

In this section we introduce the architecture of Neuro-GPT, the pre-training details, and the fine-tuning strategies. We divide the raw EEG data into fixed-length chunks from which we generate a sequence of tokens corresponding to contigu-

ous data chunks. The GPT model then learns to predict masked tokens. Employing chunks of raw EEG signals directly as input tokens to the GPT model would be problematic. Given the high dimensionality and low signal-to-noise ratio of EEG data [12], predicting raw signals is particularly challenging for the GPT model, and it may not learn meaningful features given the presence of noise. Thus, we introduce an EEG encoder [13] comprising convolutional and transformer layers to extract spatio-temporal features from the raw EEG. We input chunks of EEG into the encoder to generate the embeddings. These embeddings serve as a lower-dimensional and denoised representation of the raw EEG signals, not only simplifying the prediction of the masked chunk for the GPT model but also enhancing its ability to capture informative temporal correlations and patterns. The overall Neuro-GPT pipeline is illustrated in Figure 1.

Fig. 1 . Neuro-GPT Pipeline: the EEG encoder takes chunks of EEG data as input and generates embeddings as tokens for the GPT model. The last embedded chunk in the sequence is masked. The GPT model then predicts the masked chunk and a reconstruction loss is computed between the prediction and the original embedding token.

<!-- image -->

## 2.1. Neuro-GPT Pipeline

EEG Encoder We adopt an encoder architecture incorporating both convolutional and self-attention modules. This arrangement has achieved state-of-the-art performance in BCI classification tasks [13]. We split the raw EEG signals into N chunks, each of time length T . This results in a sequence of chunks denoted { D$\_{1}$, D$\_{2}$, · · · , D$\_{N}$ } . Each chunk is of dimension C × T , where C is the number of channels. Each chunk is treated as an individual training sample in the encoder. In the convolutional module, we apply a temporal convolution filter to the time series and a spatial convolution filter to the electrodes of the EEG. Then after average pooling, the extracted local features are fed into the self-attention layers to incorporate temporal dependencies within a chunk. The self-attention mechanism combined with convolution will encode the spatio-temporal features of each chunk. The outputs of the encoder are the embedded chunks or tokens: {H ( D$\_{1}$ ) , H ( D$\_{2}$ ) , · · · , H ( D$\_{N}$ ) } , where H denotes the map-

Fig. 2 . Causal masking: consider a sequence with four tokens (chunks). We duplicate the sequence three times and progressively mask (represented in orange) one token within each duplicated sequence.

<!-- image -->

arned by the EEG encoder from raw EEG signals to embeddings.

Causal Masking We apply a novel causal masking scheme to the tokens generated by the embedding module. As illustrated in Fig. 2, we first duplicate the sequence of tokens. Starting from the second token, one token is masked and subsequent tokens are zeroed-out in each duplicated sequence. The masked token is replaced with a learnable token M of the same dimension. So, after causal masking, the input sequence to the GPT model is

{H ( D$\_{1}$ ) , M , 0 , · · · , 0 } , {H ( D$\_{1}$ ) , H ( D$\_{2}$ ) , M , 0 , · · · , 0 } , · · · , {H ( D$\_{1}$ ) , H ( D$\_{2}$ ) , H ( D$\_{3}$ ) , · · · , M} (1)

The pre-training of Neuro-GPT utilizes a self-supervised task, where the GPT model predicts every masked token in each sequence. We use a causal reconstruction loss defined in Eq. 2 as the self-supervised pre-training objective.

L = 1 N - 1 N ∑ i =2 ∥ ˆ Y$\_{i}$ - H ( D$\_{i}$ ) ∥ 2 2 (2)

where

ˆ Y$\_{i}$ = G [ M|H ( D$\_{i}$$\_{-}$$\_{1}$ ) , H ( D$\_{i}$$\_{-}$$\_{2}$ ) , · · · , H ( D$\_{1}$ )]

where G denotes the GPT model. We aggregate the reconstruction losses of masked tokens at each position. The predicted token ˆ Y$\_{i}$ produced by the GPT model is inferred based on the preceding tokens. By predicting the masked token separately from 1, 2, and 3 preceding tokens the model gains insight into the underlying temporal correlations in brain activity across different time scales. Thus the GPT model is potentially able to capture the dynamic evolution of brain activity more accurately.

GPT Model The GPT model employs a decoder-only transformer architecture consisting of a multi-layered stack of self-attention and feed-forward modules, enabling it to capture the global dependencies between tokens. Unlike BERT [9], which randomly masks some tokens in a sequence and the model predicts the masked tokens at random

positions, GPT always predicts the next masked token given preceding tokens, also known as auto-regressive training [14]. This guarantees that the prediction of EEG embeddings considers the causal temporal relationship between tokens, thus improving our model of the underlying brain activity patterns.

## 2.2. Pre-training

Pre-training Dataset : The large-scale public dataset, Temple University Hospital (TUH) EEG corpus, is used as the pre-training dataset. TUH EEG corpus comprises a diverse archive of clinical EEG recordings from 14 , 987 subjects with multiple sessions. The archive has over 40 different channel configurations and varying duration of recordings [11]. The sample frequency ranges from 250 - 1024 Hz, with the majority of recordings sampled at 250 Hz.

Preprocessing : We preprocessed the TUH EEG dataset using the Brainstorm software [15] in MATLAB (Mathworks, Inc.). Based on the channel labels, we selected 22 channels corresponding to the extended international 10-20 system (Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2). Channels with zero or missing signals throughout the recording sessions were marked as bad channels. The signals of the bad channels were interpolated by a weighted average of all neighboring channels with a maximal distance of 5cm between neighbors. EEG recordings were re-referenced to the average of 22 channels. We removed power line noise (60 Hz) using a notch filter and bandpass-filtered the data (0.5-100 Hz). All recordings were re-sampled to 250 Hz. We performed a DC offset correction and removed linear trends from the data. A z-transform was applied along the time dimension within each recording to normalize the data.

Implementation Details : During the pre-training phase, we simultaneously pre-train the entire Neuro-GPT model. After experimenting with various input configurations, we set the standard input as: 32 chunks, each with a length of 2 seconds and a 10% (0.2 second) overlap. We randomly select a starting point for each EEG recording and then sample 32 contiguous chunks. If the total length of the EEG recording is shorter than the length to be sampled (57.8 seconds), we apply zero-padding to the end of the sequence. The attention weights are set to zero for the zero-padded part. In each training batch, one sampled sequence is considered as a single training sample. The EEG encoder consists of two convolutional layers followed by six self-attention layers, with an embedding dimension of 1,080. The first convolutional layer has a kernel size of (1 , 25) , while the second has a kernel size of ( C, 1) , with C being the number of channels [13].

We employ the open-source GPT-2 [10] model provided by Hugging Face [16], which has an embedding dimension of 1024 . We specify 6 transformer decoder layers in the GPT2 model. A linear layer is added before the GPT-2 model to project the embedding dimension from 1080 to 1024. We pre-

processed 20,000 EEG recordings from the TUH EEG dataset with a total duration of 5656 hours. We train the model on 19,000 EEG recordings for 135 epochs. The remaining 1000 EEG recordings were used as a hold-out validation set.

## 2.3. Downstream Fine-tuning

Downstream Dataset : We define the downstream task as motor imagery classification, using the BCI Competition IV Dataset 2a provided by Graz University of Technology [17]. The BCI 2a dataset consists of nine subjects performing four motor imagery tasks: imagining left hand, right hand, feet, and tongue movement. Two sessions were collected on different days for each subject, using 22 Ag/AgCl electrodes at a sampling frequency of 250 Hz. Each recording has 72 trials per task, yielding a total of 288 trials. All trials from both sessions were used as training or testing samples - importantly, no subjects in the training data were included in the testing. Data was bandpass-filtered between 0.5 Hz and 100 Hz and normalized across time for each trial. We extract the sequence from t = 2 s to t = 6 s for each trial, which corresponds to the period when the cue and motor imagery tasks are performed.

Channel resampling : The downstream dataset has a different subset of 22 channel locations on the scalp from the pre-training dataset. To match the channel configuration between the two datasets, we resampled the downstream data to the pre-training dataset channel configuration using a 22 × 22 transformation matrix. The transformation matrix was computed by solving the forward and the inverse problem for the source localization, mapping from one sensor configuration to the cerebral cortex and then back to the second configuration [18, 19].

Fine-tuning Details : We fine-tune the pre-trained model on the BCI 2a dataset for the 4-class motor imagery classification task. To fully explore the potential of the foundation model, we designed three fine-tuning strategies:

- 1. Encoder-only : Remove the GPT model and fine-tune the pre-trained EEG encoder only. (Note that in this case the model still benefits from including GPT in pre-training through the self-supervised training of the encoder in combination with the GPT model.)
- 2. Encoder+GPT : Fine-tune the entire Neuro-GPT model.
- 3. Linear : Remove the GPT model, fix the EEG encoder and fine-tune only the linear head (3 linear layers).

All strategies use the same pre-trained model and involve adding a linear head consisting of 3 linear layers to the end of the model for classification. For the Encoder+GPT strategy, we maintain the same number of chunks, the same chunk length, and the same overlapping ratio as used in the pretraining stage. Since only a 4-seconds sequence is extracted from each EEG recording in the BCI 2a dataset, we apply zero-padding to the end of the sequence. In the Encoderonly strategy, we feed the model with two non-overlapping 2-second chunks, and no zero-padding is applied. For the

Linear strategy, all the pre-trained parameters from the EEG encoder are frozen during fine-tuning. We only fine-tune the linear head, which takes the output features of the EEG encoder as input. No masking is applied during fine-tuning.

## 3. EXPERIMENTS AND RESULTS

Fine-tuning Classification Performance : Unlike previous studies which only focused on within-subject classification [20, 13], we performed leave-one-subject-out crossvalidation, which is more challenging due to the high intersubject variance. We compute the average classification accuracy across subjects. To explore the benefits of applying a pre-trained foundation model, we compare the classification performance of a model trained from scratch (w/o pre-training) to that of the same model fine-tuned on the pre-trained foundation model (w/ pre-training). In addition, we compare the proposed Neuro-GPT with BENDR [8], a BERT-inspired transformer model trained on TUH EEG data using contrastive self-supervised learning and then fine-tuned on the BCI classification data. As shown in Table 1, NeuroGPT significantly improved the classification performance compared with the best performance of BENDR, and outperforms other methods for motor imagery classification using leave-one-subject-out cross-validation.

The performance of models with pre-training surpassed that of models without pre-training for both Linear and Encoder-only fine-tuning strategies, highlighting that applying a foundation model to a downstream task can lead to effective feature learning and, consequently, improved performance. Among the fine-tuning strategies, Encoder-only achieved the best performance, indicating that the encoder learned expressive and generalizable features during pretraining, thus facilitating the learning of distinguishable features for downstream tasks. The Encoder+GPT yielded worse performance, possibly because the GPT model only serves as

Table 1 . A comparison of means and stds of four-class classification accuracy among different methods. The first three rows are three fine-tuning strategies of Neuro-GPT, accuracies reported in other work are shown in the bottom rows.

| Method       | w/o Pre-train     | w/ Pre-train      |
|--------------|-------------------|-------------------|
| Linear       | 0 . 398 ± 0 . 054 | 0 . 443 ± 0 . 051 |
| Encoder-only | 0 . 606 ± 0 . 098 | 0 . 645 ± 0 . 104 |
| Encoder+GPT  | 0 . 596 ± 0 . 090 | 0 . 586 ± 0 . 098 |
| BENDR [8]    | /                 | 0 . 426           |
| SVM [21]     | 0 . 361 ± 0 . 082 | /                 |
| EEGNet [22]  | 0 . 513 ± 0 . 052 | /                 |
| CTCNN [23]   | 0 . 477 ± 0 . 151 | /                 |
| CCNN [24]    | 0 . 553 ± 0 . 101 | /                 |
| NG-CRAM [25] | 0 . 601 ± 0 . 102 | /                 |

an auxiliary component to assist the EEG encoder in encoding meaningful features from raw EEG data. The GPT model has more trainable parameters than the encoder. Fine-tuning a large model on a small data-set can lead to over-fitting. To examine whether the features learned by the foundation model are linearly separable, we input the features generated by the EEG encoder to the linear head for classification. The classification accuracy achieved by fine-tuning only the linear head is 0 . 443 vs. 0 . 398 with out pre-training, indicating that the EEG encoder can encode meaningful features through pre-training.

Hyper-parameter Evaluation in Pre-training : To explore the optimal input configurations for the foundation model during pre-training, we conducted experiments with varying numbers of chunks (4, 8, 16, 32), chunk lengths (1s, 2s, 4s), and overlapping ratios ( 10% , 50% ). Different model architectures were also investigated. Key findings include:

- · Chunks with a 1-second length are more straightforward to predict (as embedded tokens) but led to poorer downstream performance.
- · Chunks with longer lengths are more challenging to predict but enhance downstream performance.
- · Increasing the number of chunks is beneficial. Training with 32 and 16 chunks yielded better downstream results than training with 8 or 4 chunks.
- · Increasing the overlapping ratio to 50% improved reconstruction, but degraded the downstream performance.
- · Increasing the embedding dimension of GPT-2 model ( 768 → 1024 ) improved downstream performance.
- · Reducing the number of self-attention layers in the encoder ( 6 → 4 , 2 ) degraded downstream performance.
- · Adding more GPT decoder layers ( 6 → 8 , 10 ) did not improve downstream performance.

## 4. DISCUSSION

We have demonstrated that pre-training a foundation model on a large-scale EEG dataset boosts downstream task performance. Through exploring different fine-tuning strategies, we discovered that the pre-trained EEG encoder captures inherent and fundamental features of EEG that are generalizable across datasets, leading to significant improvements in classification performance.

## 5. ACKNOWLEDGMENT

This project is sponsored in part by the NIH under grant R01 EB026299 and in part by the Defense Advanced Research Projects Agency (DARPA) under cooperative agreement No. N660012324006. The content of the information does not necessarily reflect the position or the policy of the Government, and no official endorsement should be inferred.

## 6. REFERENCES

- [1] Y. Du et al., "Eeg temporal-spatial transformer for person identification," Scientific Reports , vol. 12, no. 1, pp. 14378, 2022.
- [2] C. J. Reed et al., "Self-supervised pretraining improves self-supervised pretraining," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , 2022, pp. 2584-2594.
- [3] L. Yu et al., "Language model beats diffusion - tokenizer is key to visual generation," 2023.
- [4] K. Singhal et al., "Towards expert-level medical question answering with large language models," 2023.
- [5] A. Thomas et al., "Self-supervised learning of brain dynamics from broad neuroimaging data," Advances in Neural Information Processing Systems , vol. 35, pp. 21255-21269, 2022.
- [6] J. Ortega Caro et al., "Brainlm: A foundation model for brain activity recordings," bioRxiv , pp. 2023-09, 2023.
- [7] M. Azabou et al., "A unified, scalable framework for neural population decoding," 2023.
- [8] D. Kostas, S. Aroca-Ouellette, and F. Rudzicz, "Bendr: using transformers and a contrastive self-supervised learning task to learn from massive amounts of eeg data," 2021.
- [9] J. Devlin et al., "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805 , 2018.
- [10] A. Radford et al., "Language models are unsupervised multitask learners," OpenAI blog , vol. 1, no. 8, pp. 9, 2019.
- [11] I. Obeid and J. Picone, "The temple university hospital eeg data corpus," Frontiers in neuroscience , vol. 10, pp. 196, 2016.
- [12] C. Q. Lai et al., "Artifacts and noise removal for electroencephalogram (eeg): A literature review," in 2018 IEEE Symposium on Computer Applications & Industrial Electronics (ISCAIE) . IEEE, 2018, pp. 326-332.
- [13] Y. Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization," IEEE Transactions on Neural Systems and Rehabilitation Engineering , vol. 31, pp. 710-719, 2023.
- [14] T. B. Brown et al., "Language models are few-shot learners," 2020.
- [15] F. Tadel et al., "Brainstorm: a user-friendly application for meg/eeg analysis," Computational intelligence and neuroscience , vol. 2011, pp. 1-13, 2011.
- [16] T. Wolf et al., "Huggingface's transformers: State-ofthe-art natural language processing," 2020.
- [17] C. Brunner et al., "Bci competition 2008-graz data set a," Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology , vol. 16, pp. 1-6, 2008.
- [18] J. C. Mosher et al., "Eeg and meg: forward solutions for inverse methods," IEEE Transactions on biomedical engineering , vol. 46, no. 3, pp. 245-259, 1999.
- [19] S. Baillet et al., "Electromagnetic brain mapping," IEEE Signal processing magazine , vol. 18, no. 6, pp. 14-30, 2001.
- [20] C. Zhang et al., "Eeg-inception: an accurate and robust end-to-end neural network for eeg-based motor imagery classification," Journal of Neural Engineering , vol. 18, no. 4, pp. 046014, 2021.
- [21] V. P. Oikonomou et al., "A comparison study on eeg signal processing techniques using motor imagery eeg data," in 2017 IEEE 30th international symposium on computer-based medical systems (CBMS) . IEEE, 2017, pp. 781-786.
- [22] V. J. Lawhern et al., "Eegnet: a compact convolutional neural network for eeg-based brain-computer interfaces," Journal of neural engineering , vol. 15, no. 5, pp. 056013, 2018.
- [23] R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for eeg decoding and visualization," Human brain mapping , vol. 38, no. 11, pp. 5391-5420, 2017.
- [24] S. U. Amin et al., "Deep learning for eeg motor imagery classification based on multi-layer cnns feature fusion," Future Generation computer systems , vol. 101, pp. 542554, 2019.
- [25] D. Zhang et al., "Motor imagery classification via temporal attention cues of graph embedded eeg signals," IEEE journal of biomedical and health informatics , vol. 24, no. 9, pp. 2570-2579, 2020.

## General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data

## Mohammad Javad Darvishi Bayazi

Mila, Québec AI Institute Université de Montréal mj.darvishi92@gmail.com

## Bruno Aristimunha

Inria TAU, LISN-CNRS Université Paris-Saclay

Amin Darabi Mila, Québec AI Institute Université de Montréal

Hena Ghonia Mila, Québec AI Institute

Roland Riachi Mila, Québec AI Institute

Arian Khorasani

Mila, Québec AI Institute Université de Montréal

Guillaume Dumas

Mila, Québec AI Institute Université de Montréal

Md Rifat Arefin

Mila, Québec AI Institute Université de Montréal

Irina Rish Mila, Québec AI Institute Université de Montréal irina.rish@umontreal.ca

## Abstract

Brain function represents one of the most complex systems driving our world. Decoding its signals poses significant challenges, particularly due to the limited availability of data and the high cost of recordings. The existence of large hospital datasets and laboratory collections partially mitigates this issue. However, the lack of standardized recording protocols, varying numbers of channels, diverse setups, scenarios, and devices further complicate the task. This work addresses these challenges by introducing the Brain Foundation Model (BFM) , a suite of open-source models trained on brain signals. These models serve as foundational tools for various types of time-series neuroimaging tasks. This work presents the first model of the BFM series, which is trained on electroencephalogram (EEG) and functional Magnetic Resonance Imaging (fMRI) signal data. Our results demonstrate that BFM can generate signals more accurately than baseline models. Model weights and pipelines are available at https://bit.ly/3CCI0HW.

## 1 Introduction

Foundation models trained on large-scale datasets have revolutionized the field of artificial intelligence, demonstrating emergent capabilities across various tasks beyond their original objectives. Their adaptability and transferability make them valuable as a base for a wide range of applications (Bommasani et al. [2021]). However, time series analysis has lagged due to the scarcity of wellcurated data and its inherent complexities, making it challenging to create comprehensive datasets (Rasul et al. [2024b]).

Time series analysis is vital across numerous fields, from finance to healthcare. In healthcare, one critical application is analyzing human physiological functions over time. This provides insights into our mechanisms and aids in diagnosing and treating dysfunctions and disorders. The brain is among the most complicated and essential systems governing human behavior and perception (Buzsaki [2019]). However, decoding brain activity is particularly challenging due to its extreme complexity, requiring vast amounts of data to capture its dynamics. Recording such brain signals is not only

costly but also resource-intensive (Rashid et al. [2020]; Roy et al. [2019]). A promising approach to mitigate these challenges is leveraging the transferability of foundation models to enhance brain activity analysis.

This work aims to develop a general-purpose brain activity foundation model capable of leveraging knowledge from large language models (LLMs) and general time series data. Due to its domainagnostic nature, this model can transfer knowledge across various biosignals. We focus here on electroencephalography (EEG) and functional Magnetic Resonance Imaging (fMRI) as non-invasive methods of measuring brain activity. This framework is not limited to these two modalities and can be used for all other time series. We focused on these signals because of their wide applications ranging from medical diagnosis to brain-computer interfaces (BCI) (Hossain et al. [2023]; Safayari and Bolhasani [2021]; Popa et al. [2020]; Siuly et al. [2016]). We introduce BFM, a model that learns a robust representation of brain data capable of generating realistic brain signals. We expect its potential effectiveness in several applications.

Figure 1: Overview of BFM. (Left) Schematic of EEG electrodes on the scalp and fMRI ROI capturing signals from various brain regions. (Right) Training of Time-series model.

<!-- image -->

## 2 Related Work

Time-series (TS) forecasting: Time series forecasting is essential in many domains, ranging from finance to healthcare (Zhang et al. [2017]; Jin et al. [2018]). Accurate forecasts are critical in informing decision-making processes and strategic planning [Wu et al. , 2022; Lai et al. , 2018]. Traditional approaches to time series forecasting include statistical methods such as Autoregressive Integrated Moving Average (ARIMA) and ETS models, which use autocorrelations and decompositions into explicit fundamental components, respectively, to predict future values. Despite the success of these models, they share common inherent limitations in their assumptions of linear relationships and stationary distributions - both of which are often not the case in real-world data (Liu et al. [2023]).

Modern research in time series forecasting has also seen the rise of deep learning-based methods (Benidis et al. [2022]) focusing on the use of multi-layer perceptrons (MLPs), recurrent neural networks (RNNs) and Transformers (Vaswani et al. [2017]; Nie et al. [2023]; Wu et al. [2020]; Salinas et al. [2020a]). These new developments seek to address the challenges faced by statistical models by utilizing non-linear functions and training on diverse, complex datasets. In particular, recent work has been done for foundational time series forecasting models analogous to large language models (LLMs). Time series foundation models leverage similar techniques such as self-supervised learning and scale to achieve state-of-the-art performance across a variety of domains and datasets (Woo et al. [2024]; Rasul et al. [2024a]; Ansari et al. [2024]).

Biosignals foundation models: In recent years, several foundation models have been developed to advance the analysis of diverse biosignals. Abbaspourazad et al. [2023] developed foundation models

using extensive PPG and ECG data collected via Apple Watch. Ortega Caro et al. [2023] introduced the Brain Language Model (BrainLM), which serves as a foundation model for fMRI recordings. Azabou et al. [2024] presented POYO-1, a unified, scalable framework for neural population decoding focused on invasive neural activities. Zhang et al. [2022] applied self-supervised contrastive pretraining for time series through time-frequency consistency. Cui et al. [2023] proposed Neuro-GPT, integrating an EEG encoder with a GPT model. Chen et al. [2024] introduced EEGFormer, a pre-trained model leveraging large-scale compound EEG data. Jiang et al. [2024] introduced the Large Brain Model (LaBraM), which is trained on EEG data from BCI using vector-quantization for tokenization and masked patches for learning representations.

These models face several limitations. Primarily, their tokenization methods are often specific to the type of signal or the number of channels , which hinders their scalability and generalizability. This limitation has led to evaluations on tasks that are sometimes saturated [Kiessner et al. , 2024; Darvishi-Bayazi et al. , 2024]. In this work, we aim to develop a versatile model that can leverage various types of time series data to learn robust representations and facilitate transfer learning from LLMs and general time series models to biosignals.

## 3 Background and Method

Time-series Forecasting: Consider a dataset {X$\_{i}$} S i =1 where each X$\_{i}$ = [ x$\_{1}$, . . . , x$\_{T}$ ] ∈$\_{R}$ T$\_{i}$ × N is a multivariate time series with T$\_{i}$ time steps and N channels. Given an input time series window x$\_{t}$$\_{:}$$\_{t}$$\_{+}$$\_{C}$$\_{+}$$\_{P}$ = [ x$\_{t}$, . . . , x$\_{t}$$\_{+}$$\_{C}$$\_{+}$$\_{P}$ ] of length C + P ≥ 2 for t ∈ { 1 , . . . , T - C - P } , we look to forecast the P ≥ 1 future values. In this work, we adopt a probabilistic and channel-independent approach. This means that we individually treat each channel as a univariate time series, and we do not explicitly model the dependencies between each channel. Moreover, given x$\_{t}$$\_{:}$$\_{t}$$\_{+}$$\_{C}$ we output logits ϕ and prediction ˆ y$\_{t}$ ∼$\_{P}$$\_{ϕ}$ ( ·| x$\_{t}$$\_{:}$$\_{t}$$\_{+}$$\_{C}$ ) such that ˆ y$\_{t}$ ≈ y$\_{t}$ = x$\_{t}$$\_{+}$$\_{C}$$\_{+1:}$$\_{t}$$\_{+}$$\_{C}$$\_{+1+}$$\_{P}$ .

Large Language Models (LLMs): have demonstrated remarkable performance by leveraging massive datasets and learning billions of parameters ([Dubey et al. , 2024; Brown, 2020]). These models are largely based on the Transformer architecture ([Vaswani, 2017]). One prominent example is T5: Text-to-Text Transfer Transformer [Raffel et al. , 2020; Chung et al. , 2024], an encoderdecoder, sequence-to-sequence model that exemplifies transfer learning in natural language processing tasks. As general pattern recognizers ([Mirchandani et al. , 2023]), LLMs can also effectively process time-series data [Zhou et al. , 2023; Jin et al. , 2023]. In this work, we use T5 as the backbone for our approach, though it can seamlessly be replaced with other LLMs.

Chronos: is a model that uses T5 (Raffel et al. [2020]) architecture as a backbone and were trained on publicly available diverse time series datasets. BFM uses Chronos tokenizer (Ansari et al. [2024]) and pre-trained Chronos-T5 based (Raffel et al. [2020]) models. The implementation of the Chronos tokenizer was motivated by the fact that in language tasks, tokens are derived from a finite dictionary. In contrast, with time series data, values are from an unbounded, typically continuous domain. The tokenizer uses mean scaling (Salinas et al. [2020b]) to normalize context window x$\_{1:}$$\_{C}$ to [( x$\_{1}$ - m ) /s, .. ( x$\_{C}$ - m ) /s ] where m = 0 and s = 1 C ∑ C i $\_{=1}$| x$\_{i}$ | . After mean scaling, the tokenizer applies quantization to convert them into discrete tokens. The quantization function chooses B centers, c$\_{1}$ < ... < c$\_{B}$ uniformly and B - 1 edges b$\_{i}$ separating them, c$\_{i}$ < b$\_{i}$ < c$\_{i}$$\_{+1}$ , for i ∈ { 1 , ..., B - 1 } . Dequantization function can be defined as d ( j ) = c$\_{j}$ , where j ∈ { 1 , ..., B - 1 } . As mentioned in Figure 1, we consider each location series as independent time series, which is then passed to the tokenizer.

Brain Foundation models: BFM is univariate probabilistic forecasting model, based on Chronost5-large (700M) architecture. BFM is trained using categorical cross entropy objective function between ground truth label distribution and categorical distribution predicted by the model. We use Continuous Ranked probability score (CRPS) (Gneiting and Raftery [2007]; Matheson and Winkler [1976]) to evaluate model performance, which is commonly used to evaluate probabilistic forecasts. We report the CRPS averaged across all the time series of a dataset and over the prediction horizon using 20 empirical samples.

Datasets In this work, our objective is to learn robust representations of brain signals recorded from the scalp or different regions of interest (ROIs), as illustrated in Figure 1 (left). These multivariate signals reflect the underlying electrical and bold activity of the brain. This study uses the NMT EEG

dataset, the MOABB benchmark, and resting state fMRI from the Adolescent Brain and Cognitive Development (ABCD) study.

NMT is a public, annotated dataset comprising healthy and pathological EEG recordings (Khan et al. [2022]). It consists of 2,417 recordings from unique participants, providing multichannel EEG data and labels indicating the participants' pathological state, classified as normal or abnormal. In addition, demographic information such as gender and age is included. We leverage the predefined training and testing splits based on subjects for model pretraining and signal generation. As shown in Figure 1 (Left), each EEG channel is treated as an independent time series, which is further divided into two segments: a context window for conditioning and a prediction target window.

MOABB is a comprehensive BCI library (Aristimunha et al. [2023]) that aggregates several EEG datasets. In this work, we selected large datasets-either in terms of the number of trials or the number of subjects-to avoid the bias often present in BCI studies that rely on small, single datasets (Jayaram and Barachant [2018]). Specifically, we used the BCI Competition IV 2a dataset [Tangermann et al. , 2012], Cho2015 (Cho et al. [2017]), Weibo2014 (Yi et al. [2014]), and Liu2024 (Liu et al. [2024]). These datasets vary in recording protocols, number of channels, trial lengths, and classification tasks, providing a diverse testing ground for our model.

ABCD rs-fMRI: We utilized resting-state fMRI data from the ABCD study. The voxel-wise fMRI data were reduced to the activity of 100 brain regions using dimensionality reduction based on the Schaefer-Yeo atlas (Schaefer et al. [2018]). The preprocessing steps included removing recording artifacts and subtracting the mean signal to enhance data quality.

## 4 Empirical Evaluation

BFM aims to predict future signal values based on previous time series samples. The following section compares our model's performance quantitatively against several baseline models. A qualitative analysis of the forecasting results can be found in the appendix in Figure 6.2.

Forecasting/Generation performance: We evaluated the performance of the BFM against several baseline models. The first baseline is a Naive Model, which forecasts the next value using the last observed value. The final baseline is Chronos-Original, which leverages the pre-trained Chronos model trained on a general time series. Table 4 shows that BFM improves the performance of other models in distribution on the unseen subjects of the EEG and fMRI datasets and out-of-distribution zero-shot performance on the Moabb datasets.

Table 1: Performance of BFM on various datasets (CRPS). Lower CRPS values indicate better performance ( ↓ ). CRPS for BFM is reported as mean ± std over three seeds.

| Dataset                     |   Naive |   Chronos | BFM             |
|-----------------------------|---------|-----------|-----------------|
| NMT-EEG(unseen-subjects)    |  1.2531 |   0.8306  | 0.7675+-0.0039  |
| ABCD-fMRI (unseen-subjects) |  0.0093 |   0.00567 | 0.00543+-0.0009 |
| BNCI2014\_001                |  1.5478 |   0.9275  | 0.9005 ± 0.0006 |
| BNCI2014\_004                |  1.4118 |   0.9293  | 0.8722 ± 0.0016 |
| BNCI2015\_001                |  1.5038 |   0.9327  | 0.8219 ± 0.0051 |
| Weibo2014                   |  1.1461 |   0.9882  | 0.8721 ± 0.0199 |
| Cho2017                     |  1.4568 |   0.9176  | 0.8684 ± 0.0048 |
| Liu2024                     |  1.3151 |   0.9169  | 0.8614 ± 0.0155 |

Impact of Model Size and Transfer Learning from Language and Time-Series Models: To identify the optimal model to initialize the BFM, we examine the effects of model scaling and transfer learning from pre-trained language and time-series models on forecasting performance. Specifically, we evaluate models of varying sizes to assess how scaling impacts the validation loss. As shown in Figure 2 (Left), larger models consistently achieve lower validation loss, leading us to select the largest model for subsequent experiments. Furthermore, we explore the utility of transfer learning by initializing BFM with weights from pre-trained language models and time-series models. Figure 2 (Right) demonstrates a positive transfer from language model weights, with even lower loss observed when initializing from the Chronos time-series model weights. This highlights the potential of transfer learning from other modalities in enhancing neuroimaging tasks.

Figure 2: Scaling and Transfer Behavior. (Left) Larger models show smaller validation loss. (Right) Positive transfer from language and general time series models to EEG signals.

<!-- image -->

## 5 Discussion and Conclusion

We introduced and evaluated the Brain Foundation Model (BFM), a framework capable of learning the dynamics of brain signals by leveraging knowledge from both language models and general time series data. Despite the mixed findings regarding the effectiveness of transferring knowledge from LLMs to time series (Tan et al. [2024]; Zhou et al. [2023]), our results demonstrate a positive transfer from LLMs to EEG data, with an even stronger transfer observed from general time series models to the brain. BFM exhibited strong generalization across datasets beyond its training set. We believe that this model possesses the core characteristics of a foundation model and has the potential to improve multimodal analysis of simultaneous EEG-fMRI analysis (Lioi et al. [2020]; Ciccarelli et al. [2023]), brain-body signal analysis for human state assessments (Darvishi-Bayazi et al. [2023]) or decoding speech (Défossez et al. [2023]; Millet et al. [2022]). We believe this unified framework significantly advances BCI applications, diagnostic tools, and neuroscience research through the analysis of brain signals.

## Acknowledgements

We acknowledge the support from the Canada CIFAR AI Chair Program and from the Canada Excellence Research Chairs Program. This research was made possible thanks to the computing resources on the Summit and Frontier supercomputers provided by the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. We sincerely appreciate Danilo Bzdok and Shambhavi Aggarwal valuable comments, which greatly contributed to the improvment of this paper.

## References

Salar Abbaspourazad, Oussama Elachqar, Andrew C Miller, Saba Emrani, Udhyakumar Nallasamy, and Ian Shapiro. Large-scale training of foundation models for wearable biosignals. arXiv preprint arXiv:2312.05409 , 2023.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815 , 2024.

Bruno Aristimunha, Igor Carrara, Pierre Guetschel, Sara Sedlar, Pedro Rodrigues, Jan Sosulski, Divyesh Narayanan, Erik Bjareholt, Quentin Barthelemy, Robin Tibor Schirrmeister, Emmanuel Kalunga, Ludovic Darmet, Cattan Gregoire, Ali Abdul Hussain, Ramiro Gatti, Vladislav Goncharenko, Jordy Thielen, Thomas Moreau, Yannick Roy, Vinay Jayaram, Alexandre Barachant, and Sylvain Chevallier. Mother of all bci benchmarks, 2023. Version 1.1.0. DOI: 10.5281/zenodo.10034223 .

| framework for neural population decoding. 36, 2024.                                                                                                                                                                                   | Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake Richards, Matthew Perich, Guillaume Lajoie, and Eva Dyer. A unified, scalable Advances in Neural Information Processing Systems ,   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                                                                                                       | Konstantinos Benidis, Syama Sundar Rangapuram, Valentin Flunkert, Yuyang Wang, Danielle                                                                                                                                                       |
| Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Tom B Brown. Language models are few-shot learners.                                                                                           | , 2021. arXiv preprint arXiv:2005.14165 , 2020.                                                                                                                                                                                               |
| Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportuni- ties and risks of foundation models. arXiv preprint arXiv:2108.07258                                                                  |                                                                                                                                                                                                                                               |
| Gyorgy Buzsaki. The brain from inside out                                                                                                                                                                                             | . Oxford University Press, USA, 2019.                                                                                                                                                                                                         |
| Yuqi Chen, Kan Ren, Kaitao Song, Yansen Wang, Yifan Wang, Dongsheng Li, and Lili Qiu. Eeg- former: Towards transferable and interpretable large-scale eeg foundation model.                                                           | arXiv preprint                                                                                                                                                                                                                                |
| arXiv:2401.10278 , 2024.                                                                                                                                                                                                              | Hohyun Cho, Minkyu Ahn, Sangtae Ahn, Moonyoung Kwon, and Sung Chan Jun. Eeg datasets for motor imagery brain-computer interface. GigaScience , 6(7):gix034, 2017.                                                                             |
| Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research | , 25(70):1-53, 2024.                                                                                                                                                                                                                          |
| Giuseppina Ciccarelli, Giovanni Federico, Giulia Mele, Angelica Di Cecca, Miriana Migliaccio,                                                                                                                                         |                                                                                                                                                                                                                                               |
| Ciro Rosario Ilardi, Vincenzo Alfano, Marco Salvatore, and Carlo Cavaliere. Simultaneous real- time eeg-fmri neurofeedback: A systematic review. 2023.                                                                                | Frontiers in Human Neuroscience , 17:1123014,                                                                                                                                                                                                 |
| Wenhui Cui, Woojae Jeong, Philipp Thölke, Takfarinas Medani, Karim Jerbi, Anand A Joshi, and Richard M Leahy. Neuro-gpt: Developing a foundation model for eeg. arXiv:2311.03764 , 2023.                                              | arXiv preprint                                                                                                                                                                                                                                |
| Mohammad-Javad Darvishi-Bayazi, Andrew Law, Sergio Mejia Romero, Sion Jennings, Irina Rish, and Jocelyn Faubert. differences in ab initio pilots. Scientific Reports                                                                  | Beyond performance: the role of task demand, effort, and individual , 13(1):14035, 2023. Mohammad-Javad Darvishi-Bayazi, Mohammad Sajjad Ghaemi, Timothee Lesort, Md Rifat Arefin,                                                            |
| Jocelyn Faubert, and Irina Rish. Amplifying pathological detection in eeg signaling pathways through cross-dataset transfer learning. Computers in Biology and Medicine ,                                                             | , 169:107893, 2024. Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, and Jean-Rémi King. De- Nature Machine Intelligence                                                                                                   |
| coding speech perception from non-invasive brain recordings. 5(10):1097-1107, 2023. Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha                                                          | Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.                                                                                                                                                 |
| Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association , 102(477):359-378, 2007.                                                           |                                                                                                                                                                                                                                               |
| Khondoker Murad Hossain, Md Ariful Islam, Shahera Hossain, Anton Nijholt, and Md Atiqur Rahman                                                                                                                                        | Ahad. Status of deep learning for eeg-based brain-computer interface applications. Frontiers in                                                                                                                                               |

| Wei-Bang Jiang, Li-Ming Zhao, and Bao-Liang Lu. Large brain model for learning generic represen- tations with tremendous eeg data in bci. arXiv preprint arXiv:2405.18765 , 2024.                                                                                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bo Jin, Haoyu Yang, Leilei Sun, Chuanren Liu, Yue Qu, and Jianing Tong. A treatment engine by predicting next-period prescriptions. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining , KDD '18, page 1608-1616, New York, NY, USA, 2018. Association for Computing Machinery.                                         |
| Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yux- uan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728 , 2023. Hassan Aqeel Khan, Rahat Ul Ain, Awais Mehmood Kamboh, Hammad Tanveer Butt, Saima Shafait,                    |
| Wasim Alamgir, Didier Stricker, and Faisal Shafait. The nmt scalp eeg dataset: an open-source annotated dataset of healthy and pathological eeg recordings for predictive modeling. Frontiers in                                                                                                                                                                         |
| Ann-Kathrin Kiessner, Robin T Schirrmeister, Joschka Boedecker, and Tonio Ball. Reaching the ceiling? empirical scaling behaviour for deep eeg pathology classification. Computers in Biology and Medicine , page 108681, 2024. Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In |
| The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval , SIGIR '18, page 95-104, New York, NY, USA, 2018. Association for Computing Machinery. Giulia Lioi, Claire Cury, Lorraine Perronnet, Marsel Mano, Elise Bannier, Anatole Lécuyer, and                                                                                    |
| Christian Barillot. Simultaneous eeg-fmri during a neurofeedback task, a brain imaging dataset for multimodal data integration. Scientific data , 7(1):173, 2020. Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring                                                                                                           |
| Haijie Liu, Penghu Wei, Haochong Wang, Xiaodong Lv, Wei Duan, Meijie Li, Yan Zhao, Qingmei                                                                                                                                                                                                                                                                               |
| Wang, Xinyuan Chen, Gaige Shi, et al. An eeg motor imagery dataset for brain computer interface in acute stroke patients. Scientific Data , 11(1):131, 2024.                                                                                                                                                                                                             |
| James E Matheson and Robert L Winkler. Scoring rules for continuous probability distributions. Management science , 22(10):1087-1096, 1976.                                                                                                                                                                                                                              |
| Juliette Millet, Charlotte Caucheteux, Yves Boubenec, Alexandre Gramfort, Ewan Dunbar, Christophe Pallier, Jean-Remi King, et al. Toward a realistic model of speech processing in the brain with self-supervised learning. Advances in Neural Information Processing Systems , 35:33428-33443,                                                                          |
| Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, and Andy Zeng. Large language models as general pattern machines. arXiv preprint arXiv:2307.04721 , 2023.                                                                                                                                 |
| Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers, 2023.                                                                                                                                                                                                                      |
| Josue Ortega Caro, Antonio Henrique Oliveira Fonseca, Christopher Averill, Syed A Rizvi, Matteo Rosati, James L Cross, Prateek Mittal, Emanuele Zappala, Daniel Levine, Rahul M Dhodapkar, et al. Brainlm: A foundation model for brain activity recordings. bioRxiv , pages 2023-09, 2023.                                                                              |
| Livia Livint Popa, Hanna Dragos, Cristina Pantelemon, Olivia Verisezan Rosu, and Stefan Strilciuc. The role of quantitative eeg in the diagnosis of neuropsychiatric disorders. Journal of medicine                                                                                                                                                                      |
| Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research , 21(140):1-67, 2020.                                                                                               |

| of eeg-based brain-computer interface: A comprehensive review. June 2020.                                                                                                                                                                                                                      | Mamunur Rashid, Norizam Sulaiman, Anwar P. P. Abdul Majeed, Rabiu Muazu Musa, Ahmad Fakhri Ab. Nasir, Bifta Sama Bari, and Sabira Khatun. Current status, challenges, and possible solutions Frontiers in Neurorobotics , 14,           |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                                                                                                                                                                | Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Hena Ghonia, Rishika Bhagwatkar, Arian                                                                                                                                               |
| Khorasani, Mohammad Javad Darvishi Bayazi, George Adamopoulos, Roland Riachi, Nadhir                                                                                                                                                                                                           |                                                                                                                                                                                                                                         |
| Hassen, Marin Biloš, Sahil Garg, Anderson Schneider, Nicolas Chapados, Alexandre Drouin,                                                                                                                                                                                                       | Valentina Zantedeschi, Yuriy Nevmyvaka, and Irina Rish. Lag-llama: Towards foundation models                                                                                                                                            |
| for probabilistic time series forecasting, 2024.                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                         |
| Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Hena Ghonia, Rishika Bhagwatkar, Arian Khorasani, Mohammad Javad Darvishi Bayazi, George Adamopoulos, Roland Riachi, Nadhir Hassen, et al. Lag-llama: Towards foundation models for probabilistic time series forecasting. Preprint , 2024. | Yannick Roy, Hubert Banville, Isabela Albuquerque, Alexandre Gramfort, Tiago H Falk, and Jocelyn Faubert. Deep learning-based electroencephalography analysis: a systematic review. Journal of                                          |
| David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecast- ing with autoregressive recurrent networks. 2020.                                                                                                                                        | International Journal of Forecasting , 36(3):1181-1191, International journal of forecasting , 36(3):1181-1191,                                                                                                                         |
| David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecast- ing with autoregressive recurrent networks.                                                                                                                                              |                                                                                                                                                                                                                                         |
| 2020.                                                                                                                                                                                                                                                                                          | , 28(9):3095-3114, 2018.                                                                                                                                                                                                                |
| Alexander Schaefer, Ru Kong, Evan M Gordon, Timothy O Laumann, Xi-Nian Zuo, Avram J Holmes, Simon B Eickhoff, and BT Thomas Yeo. Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity mri. Cerebral cortex                                            |                                                                                                                                                                                                                                         |
|                                                                                                                                                                                                                                                                                                | arXiv preprint arXiv:2406.16964 , 2024. Michael Tangermann, Klaus-Robert Müller, Ad Aertsen, Niels Birbaumer, Christoph Braun, Clemens                                                                                                  |
| Mingtian Tan, Mike A Merrill, Vinayak Gupta, Tim Althoff, and Thomas Hartvigsen. Are language models actually useful for time series forecasting?                                                                                                                                              | Brunner, Robert Leeb, Carsten Mehring, Kai J Miller, Gernot R Müller-Putz, et al. Review of the                                                                                                                                         |
|                                                                                                                                                                                                                                                                                                | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Advances in Neural Information Processing Systems 30 , pages 5998-6008. Curran Associates, Inc., 2017.                                  |
| H.                                                                                                                                                                                                                                                                                             | Kaiser, and Illia Polosukhin. Attention is All you Need. In I. Guyon, U.V. Luxburg, S. Bengio, Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors,                                                                            |
| A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems                                                                                                                                                                                                        | , 2017.                                                                                                                                                                                                                                 |
| Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers, 2024.                                                                                                                             |                                                                                                                                                                                                                                         |
| Neo Wu, Bradley Green, Xue Ben, and Shawn O'Banion. Deep transformer models for time series forecasting: The influenza prevalence case, 2020.                                                                                                                                                  |                                                                                                                                                                                                                                         |
| Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, 2022.                                                                                                                                    |                                                                                                                                                                                                                                         |
| Weibo Yi, Shuang Qiu, Kun Wang, Hongzhi Qi, Lixin Zhang, Peng Zhou, Feng He, and Dong Ming. Evaluation of eeg oscillatory patterns and cognitive process during simple and compound limb motor imagery. PloS one , 9(12):e114853, 2014.                                                        | Weibo Yi, Shuang Qiu, Kun Wang, Hongzhi Qi, Lixin Zhang, Peng Zhou, Feng He, and Dong Ming. Evaluation of eeg oscillatory patterns and cognitive process during simple and compound limb motor imagery. PloS one , 9(12):e114853, 2014. |

- Liheng Zhang, Charu Aggarwal, and Guo-Jun Qi. Stock price prediction via discovering multifrequency trading patterns. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '17, page 2141-2149, New York, NY, USA, 2017. Association for Computing Machinery.

Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik. Self-supervised contrastive pre-training for time series via time-frequency consistency. Advances in Neural Information Processing Systems , 35:3988-4003, 2022.

Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems , 36:43322-43355, 2023.

## 6 Appendix

## 6.1 Training Details

We trained model of 3 sizes initialized from chronos weights, tiny(8M), base(200M), and large(700M) for 2K steps with effective batch size 2048, to study the effect of model size on validation loss performance. We use context length 512 for BFM and prediction length 64, with linear scheduler for learning rate starting from 0.001. BFM large was trained for 6K steps with an effective batch size of 2048 on eight nodes(4 AMD MI250X or 8 separate GPUs, each having 64 GB of high-bandwidth memory) using DDP (Data Distributed parallelization).

## 6.2 Forecasting visualization: qualitative analysis

Figure 3 presents several examples of the predicted signals, including the observed signals, the median of the predicted signals, and the 80% prediction interval. The results qualitatively demonstrate that our models effectively learn the underlying patterns in EEG signals, generating meaningful and realistic samples.

Figure 3: Forecasting Results. Three examples of EEG signals generated by the proposed time-series model. The predicted signals are compared to the original EEG recordings to evaluate the accuracy of the model's predictions.

<!-- image -->

<!-- image -->

## The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark

Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, Bruna Junqueira Lopes, Sébastien Velut, Salim Khazem, Thomas Moreau

## To cite this version:

Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, et al.. The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. 2024. ￿hal-04537061￿

## HAL Id: hal-04537061

## https://universite-paris-saclay.hal.science/hal-04537061v1

Preprint submitted on 8 Apr 2024

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

<!-- image -->

## The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark

Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, Bruna Lopes, Sebastien Velut, Salim Khazem, Thomas Moreau

Inria TAU, LISN-CNRS, Universit' e Paris-Saclay, 91405, Orsay, France

Universit' e Cˆ ote d'Azur, Inria Cronos Team, Sophia Antipolis, France

Donders Institute, Radboud University, Nijmegen, Netherlands

University of S˜ ao Paulo, Sao Paulo, Brazil

Federal University of ABC, Santo Andre, Brazil

GeorgiaTech-CNRS IRL 2958, Centralesupelec, Metz, France

Inria Mind team, Universit' e Paris-Saclay, CEA, Palaiseau, 91120, France

E-mail: sylvain.chevallier@universite-paris-saclay.fr

## Abstract.

Objective . This study conduct an extensive Brain-computer interfaces (BCI) reproducibility analysis on open electroencephalography datasets, aiming to assess existing solutions and establish open and reproducible benchmarks for effective comparison within the field. The need for such benchmark lies in the rapid industrial progress that has given rise to undisclosed proprietary solutions. Furthermore, the scientific literature is dense, often featuring challenging-to-reproduce evaluations, making comparisons between existing approaches arduous. Approach . Within an open framework, 30 machine learning pipelines (separated into raw signal: 11, Riemannian: 13, deep learning: 6) are meticulously re-implemented and evaluated across 36 publicly available datasets, including motor imagery (14), P300 (15), and SSVEP (7). The analysis incorporates statistical meta-analysis techniques for results assessment, encompassing execution time and environmental impact considerations. Main results . The study yields principled and robust results applicable to various BCI paradigms, emphasizing motor imagery, P300, and SSVEP. Notably, Riemannian approaches utilizing spatial covariance matrices exhibit superior performance, underscoring the necessity for significant data volumes to achieve competitive outcomes with deep learning techniques. The comprehensive results are openly accessible, paving the way for future research to further enhance reproducibility in the BCI domain. Significance . The significance of this study lies in its contribution to establishing a rigorous and transparent benchmark for BCI research, offering insights into optimal methodologies and highlighting the importance of reproducibility in driving advancements within the field.

Keywords : Brain-computer interface, EEG, Reproducibility, Riemannian classifier, Deep learning

Submitted to: J. Neural Eng.

## 1. Introduction

The field of Brain-Computer Interface (BCI) aims at developing methodologies to allow interactions with devices, like prostheses or computer environments, from decoding brain signals. It is a very good candidate technology to assist people with motor disability, as it requires very limited motor capabilities of the subject. To go from brain signals to decisionmaking, BCI systems are defined by several choices: a paradigm, that specifies the cognitive tasks to perform to control the interface, an acquisition device, to record the brain activity, and an algorithmic pipeline, that processes the acquired data and predicts which action the subject intends to perform. As such, BCI is at the forefront of interdisciplinary research, integrating expertise from diverse fields such as electronics, neuroscience, human-machine interaction (HMI), signal processing, and machine learning.

## 1.1. Open data

To fasten the emergence of BCI system, the field has organized to decouple the lengthy paradigm design and data acquisition processes from the development of algorithmic pipelines to process them. The creation and publication of many accessible and openly available datasets allow for offline development and evaluation of novel processing pipelines. They also improve experiment replication and ensure the replicability of published results.

Due to the constraints on real-time acquisitions in open environments, Electroencephalography (EEG) has become the leading device to develop BCI systems with its highfrequency acquisition and limited constraints on its deployment. In EEG-based BCI, many real-world competitions and open datasets ex-

ist on a world scale to design and evaluate the best BCI systems. BCI datasets come with various paradigms, depending on the decoding task to perform from brain signals. Motor Imagery (MI) is a common paradigm choice to control a BCI, with different imagined movements, but other paradigms are also very efficient like Event Related Potential (ERP) generated by oddball stimulus or Steady State Visually Evoked Potential (SSVEP) produced by repetitive stimulations. Many open datasets are thus openly available for EEG-based BCI.

Yet, open data is not solely about availability. It also requires the use of open formats that enable easy reading and exploitation of metadata. Indeed, there exists a wide variety of EEG acquisition devices with various hardware specifications, as well as a plethora of paradigm specifications, with their experimental design, shared annotations, and code for interpreting the cued events. Retrieving this information is critical to developing valid systems that can correctly process the retrieved data. Drawing inspiration from brain imaging techniques like fMRI or MEG that rely on the Brain Imaging Data Structure (BIDS), an extension of this format has been proposed for EEG-based research [91]. Despite this proposed format, most EEG open data available online are stored in diverse structures and formats, which hinders the automation of data collection, as each dataset requires specific processing scripts.

Moreover, while numerous books on EEG data acquisition and BCI exist, it remains challenging to find well-founded guidelines for hardware requirements and BCI design. These guidelines are needed to allow practitioners to make informed decisions regarding experimental design and hardware selection. In the worst-case scenario, a poorly designed BCI or an inappropriate hardware choice can result

in unusable data. The unique characteristics of EEG-based BCIs could be guided by metaanalyses of existing datasets, leading to recommendations on the number and positioning of electrodes, sampling frequency, the required number of subject trials, and the influence of the number of classes. However, to the best of our knowledge, there is no comprehensive and systematic evaluation of these parameters available in the existing literature.

## 1.2. BCI pipelines

Based on these open datasets, the development of BCI classification pipelines for EEG signals has a long-standing research history and a very active community [75]. Pipelines initially focused on methods based on feature extraction from raw signals, such as channel variance or local variation [74], combined with complex classifiers. The first breakthrough came with the introduction of spatial filtering methods based on covariance estimation, such as XDAWN [95], CSP [2], or CCA [72]. These methods significantly improved classification accuracy when paired with simple classifiers. By using supervised filters, the separability of different BCI classes was improved while reducing the dimensionality of the input signal. As a result, these raw-signal-based pipelines with spatial filtering became the top-ranked approaches in BCI competitions. The second advancement in BCI classification involved a reformulation of spatial filtering based on covariance matrices using Riemannian geometry [12, 124]. Leveraging the inherent manifold structure of symmetric covariance matrices, this approach provides robust classification by exploiting the invariance properties of the covariance manifold. This method has demonstrated remarkable performance across various BCI tasks, even with noisy EEG sig-

nals, making it the preferred choice and quickly outperforming other pipelines in BCI competitions [33]. Pipelines based on Riemannian geometry are considered state-of-the-art methods in the current literature.

Recently, deep learning methods have been explored for EEG-based BCI classification [99]. Although deep learning has enjoyed success in computer vision tasks, it is not straightforward to adapt these techniques to handle strongly correlated time series data, such as EEG signals. As a result, the deep learning approaches performances on EEG signals have not outperformed existing approaches [103]. Indeed, the main workhorse for the success of deep learning lies in the availability of vast amounts of data. However, in the context of BCI and EEG data, there is a scarcity of data at subject-level, which could potentially explain the limited performance of deep neural networks in this domain. To address this issue, incorporating auxiliary tasks such as self-supervised learning [7] or data augmentation [97] are promising approaches. However, further investigation is still required to explore the efficacy of these methods.

## 1.3. Evaluation and Reproducibility in BCI

While the literature on EEG-based BCI pipelines is very dense, their evaluation and the interpretation of the results is a major issue for BCI. Indeed, for many studies, it is very difficult, if not impossible to compare the produced results. This stems from the fact that various factors hinder the experimental result comparison: specific preprocessing, cherry-picking datasets, subjects, or classes on a dataset, missing pipelines in the comparison, or lack of statistical analysis.

On the one hand, this issue is compounded

by the need for shared resources and comprehensive evaluations. Thorough evaluations often exceed the scope of a single research work, and it is thus crucial to find rigorous methods to assess the performance of BCI pipelines and ensure the reproducibility of their results. On the other hand, the scientific community has been grappling with a reproducibility crisis across various domains [6], and the field of BCI is not exempt. Addressing this crisis in BCI research is particularly pertinent due to the domain's unique requirements and complexities. BCI studies involve complex methodologies, intricate data acquisition techniques, and multifaceted analysis pipelines, making it challenging to replicate and validate experiments. The transdisciplinary nature of BCI research necessitates specialized knowledge from multiple disciplines, further complicating the ability to reproduce results.

Meta-analysis, a statistical technique for combining the results of multiple studies, is a powerful tool for synthesizing findings and deriving insights from a large body of research [48]. However, due to nonstandardized evaluation protocols, conducting meta-analyses in the field of BCI has proven to be challenging. The variability in datasets, experimental tasks, and analysis pipelines inhibits the aggregation of results, despite efforts to establish standardization protocols. Consequently, it is difficult to obtain a comprehensive understanding of the performance of BCIs pipelines across different paradigms through the existing literature.

In response to these challenges, the field has seen the emergence of the Mother of All BCI Benchmarks (MOABB). MOABB [4] was developed as an open-source platform to facilitate the benchmarking and assessment of new datasets and classifiers in major BCI paradigms. The first version of MOABB initi-

ated a community-driven effort to define rigorous and expert methods for conducting proper benchmarking assessments. By providing standardized evaluation procedures, MOABB enables researchers to compare and evaluate the performance of BCI methods in a transparent and reproducible manner. While its adoption by the community is broadening, the systematic evaluation of existing pipelines proposed in [54] is now outdated because it is focused on a specific BCI modality, and new pipelines have emerged in the literature.

## 1.4. Environmental impact

Addressing climate change requires comprehensive actions across all facets of human societies to adhere to the Paris Climate Agreement. Research, including the field of AI, plays a significant role in achieving this objective. Assessing the environmental impact of machine learning techniques is a challenging task [71], which has been pioneered by the natural language processing community with the prominence of conversational agents [76]. While machine learning models used in BCI are smaller in scale, assessing the performance of various models in link with their environmental impact is critical to promoting virtuous and sustainable models.

The costs associated with deploying computers or computer clusters for training machine learning models are dependent on infrastructure requirements. The environmental impact resulting from energy consumption during model training varies based on geographical criteria, as electricity production methods differ across countries. Several libraries have been developed to provide estimates of this environmental impact, measured in terms of grams of CO2 equivalent emissions [53]. Although these libraries have limitations, they offer a valuable

measurement to enhance our understanding of the training requirements for models. This facilitates a more comprehensive comparison of pipelines within a benchmark setting.

## 1.5. Contributions

This paper aims to address the lack of reproducibility studies in the field of EEG-based BCI and to go beyond a simple benchmark of existing machine learning pipelines. Indeed, it is important to provide an updated comparison of the previous benchmark [54], including deep learning pipelines. Also, new BCI paradigms for controlling systems could now be included in the evaluation to propose a global take on the current BCI state.

It is an impossible task to evaluate all openly available datasets and all published pipelines in the literature, as there are exotic or unreadable data formats, BCI protocols that are used for a unique dataset, and unavailable code. For this paper, we chose to restrict to three common BCI paradigms, namely MI, ERP, and SSVEP, that are well documented in the literature. The reason for this choice is to ensure reproducible evaluation with enough datasets and subjects to achieve decent statistical meta-analysis.

For the machine learning pipelines, we follow similar guidelines to consider only approaches that could be reimplemented with Python open-source libraries. Indeed, we could only reimplement part of the published pipelines, but we try to include to the best of our effort all pipelines that have been reused in several publications or that are often cited as reference. The MOABB framework is designed to allow seamless integration of novel pipelines for new contributors and facilitate as much as possible the reproducibility of a benchmark to compare a novel pipeline with the one

presented here.

We thought of the MOABB open science initiative as a long-term project. The objective is to endow the community with the ability to easily compare results on new datasets or new pipelines, that will be added after the paper publication. A website reproduces the results presented here and will be updated with novel additions. The goal is also to limit the environmental impact of research works by providing a simple means for scientists to copy/paste up-to-date results in their publication to compare with a reference benchmark.

Beyond the purely quantitative benchmark, it is difficult to have a good overview of the open EEG datasets available online. They have been recorded with different hardware, using similar experimental protocols with some variability in experiment design. As data sharing is becoming a common requirement in the scientific community, future works will often result in sharing new open data. It is important to be able to quickly identify existing datasets, and common design choices, to correctly position new experiments with wellinformed knowledge. This task is difficult, as open data is scattered in the literature with no common format.

To summarize, in this paper we aim to address the following open questions:

- · What is the most effective approach for classifying EEG? How do their computation time and energy consumption compare for training a model?
- · What is the best deep learning method? The best Riemannian-BCI pipeline?
- · How many trials or channels are required in a dataset to achieve correct performances?
- · For MI, which motor imagery tasks give

the best accuracy?

- · What are the open datasets in different paradigms and how do they compare?

To investigate those research questions, we conducted many experiments built on opensource tools and with the help of a large community driven by open science guidelines. This paper describes the results obtained from a wide experimental campaign and their in-depth analysis, resulting in the following contributions:

- · the largest benchmarking in EEG-based BCI in open science relying on open source libraries
- · a fair and replicable evaluation with expert knowledge in BCI
- · a deep analysis of the benchmark results, with guidelines for proposing new machine learning pipelines and new datasets or experiments.

## 2. Benchmark methodology

In this section, we describe the methodology for our benchmark, including which methods are considered, how they are trained and evaluated, and how we analyze the results.

## 2.1. Analysis pipelines inclusion

A major difficulty in the BCI literature, apart from the data and code availability, is the importance of signal processing in EEG. Many toolboxes are available, in different software platforms, like Matlab, Python, C/C++, Java, C#, Julia, Delphi, and many more. Some toolboxes are sold as commercial products, with undisclosed code or signal processing techniques that are hidden or obfuscated for intellectual property reason. The choices made in those toolboxes, let apart all single projects

maintained by only one person, are very different regarding signal filtering or electrode referencing and yield very different outcomes. Despite the fact that complex preprocessing pipelines, and often arguably overcomplicated ones, are detrimental for EEG interpretation and classification [37], most of the toolboxes include them and promote their application in tutorials and guidelines.

In this paper, our analysis pipeline relies on the Python language for its large adoption in the scientific computing, neuroscience and machine learning communities, supported by robust and extensively validated libraries such as numpy [47], scipy [118], MNE [43], scikitlearn [89] and pyriemann [14]. Manipulation of EEG recordings is facilitated through MNE, enabling the extraction of hardware events and the conversion of recordings into numpy arrays. As a light preprocessing, the signal is processed with a 4th-order Butterworth bandpass IIR filter, applied with a forward-backward pass, using standard MNE parametrization. The specific bandpass frequencies are subsequently provided as they depend on the chosen BCI paradigm. Machine learning pipelines are based on scikit-learn and pyriemann estimators.

## 2.2. Evaluation method

Another difficulty for anyone who wants to reproduce literature results is that reporting classification performances on EEG-based BCI tasks is not standardized. A crucial methodological consideration pertains to the selection of the evaluation metric (whether it be Area Under the Curve (AUC), precision/recall, F1-score, or accuracy), a decision that holds notable significance in the assessment of outcomes. Moreover, in many studies, the authors focus on specific subsets of subjects

within established datasets, or selectively use a restricted part of the cognitive tasks conducted during experiments. These choices make comparing findings across papers becomes inherently cumbersome.

To maintain consistent terminology concerning EEG signals, we will establish the following definitions. A session refers to a series of runs where EEG electrodes remain attached to a subject's head, and the overarching experimental parameters remain constant. The term run denotes the period during which an experiment is conducted without interruption or pause. Within a session, several runs are performed with potential intervals between them. An epoch or a trial signifies a segment within a run during which an atomic event occurs, triggered by an external stimulus for eventrelated potential or steady-state evoked potential or internal volition for motor imagery. These epochs are positioned in time relative to an onset, which corresponds to the start of a stimulus or a task.

In the BCI framework, different evaluation methodologies exist for partitioning the data into training and test datasets, each tailored to address specific challenges. We differentiate between within-session as shown in Figure 1, cross-session , and cross-subject evaluations, respectively illustrated in the annexes in Figure A1 and Figure A2.

In the context of within-session evaluation, the primary concern lies in identifying algorithms that can effectively mitigate overfitting within a single session. In line with common practices in machine learning for cross-validation, all trials from a session are shuffled before being split in k-folds, to evaluate the generalization performance on unseen epochs. Consequently, the pipelines are trained on trials sampled throughout the session duration, which helps mitigate the im-

Figure 1: Within-session evaluation, small rectangles indicate a sample or EEG trial, pastel colors on the two top lines shows the chronological order, bright color on the last three lines indicates training and testing samples/trials.

<!-- image -->

pact of intra-session variability in an individual's EEG. While this allows a more statistically accurate benchmark of a pipeline, it provides an upper bound for the evaluation metrics when compared to online evaluation. Despite the difference with experimental applications of BCI system, this training methodology is commonly employed in the existing literature [13, 111, 84, 75], influencing our decision to adopt it for this reproducibility study.

In contrast, the cross-session (resp. crosssubject) evaluation employs a leave-one-out cross-validation technique, where only one session (resp. subject) is designated as the testing dataset. The results from crosssession or cross-subject evaluation methods put more emphasis on transfer learning to cope with subjects' variability. However, the questions raised by transfer learning in a BCI context [119, 121] are manyfold - dealing with

subject alignment, training models for each subject or a single one, at a session, subject or dataset level - and are outside the scope of this article. For this reason, this paper focuses on the most common evaluation strategy in the literature, which is within-session evaluation.

Nonetheless, one should acknowledge that within-session evaluation has its limitations. It addresses only partly the complications stemming from variations in sessions and subjects. Such variations may arise from internal factors, like minor electrode displacements between sessions, different calibrations of EEG hardware, or external factors, like the dynamic nature of EEG measurements in an individual based on the cognitive states [98], such as alertness, drowsiness or fatigue.

Within-session evaluation of a pipeline is conducted for each subject and each session by splitting the epochs into training and test epoch sets using five stratified K-fold splits. In each fold, the testing set contains 20% of the session epochs while maintaining the class distribution. The final evaluation score corresponds to the average over the five splits. Depending on whether the classification problem is binary or involves more than two classes, evaluation scores correspond to the Area Under the Receiver Operating Characteristic Curve (ROC) or classification accuracy, respectively.

## 2.3. Grid search

Another important aspect in BCIs is how the hyperparameters are selected. They can significantly affect the performance of the algorithm and, consequently, the accuracy of the system's predictions. It is therefore crucial to employ a method that facilitates the search for optimal parameter values. Grid search stands out as a popular method for

hyperparameter tuning in machine learning algorithms. In particular, it is essential to search for the correct hyper-parameters for each scenario, considering variations in evaluation procedures, datasets, subjects, and sessions.

Using nested cross-validation for hyperparameters selection, we mitigate the risk of overfitting, as recommended in existing literature [25]. This approach is implemented by using an inner 3-fold cross-validation. Additionally, we have devised tailored grid search functions for each evaluation procedure, as detailed in subsection Appendix A.1. This standardized framework streamlines hyperparameter tuning, enabling seamless performance comparison across diverse machine-learning models and promoting experiment reproducibility across varied datasets.

## 2.4. Statistical analysis

The statistical comparison of two different pipelines can be conducted either at a datasetwise level or across multiple datasets .

The dataset-wise comparison involves assessing the statistical differences between two pipelines within the same dataset. This is done using effect sizes and p-values. To optimize computational efficiency, the number of subjects N in the dataset determines the method of estimating p-values. For datasets with N < 13, one-tailed permutation-based paired t -tests are used with all possible permutations. For datasets with 13 ⩽ N ⩽ 20, 10000 random permutations are employed. For datasets with N > 20, the Wilcoxon signedrank test [122] is used. The effect size between two pipelines is measured via Standardized Mean Difference (SMD) [48] estimated over the subjects.

For multiple datasets , the comparison of

statistical differences between two pipelines over D is done by combining effect sizes { s$\_{i}$ } D i =1 and p-values { p$\_{i}$ } D i =1 with Stouffer's Zscore method [96] that take in account the different sizes of the datasets. Combined p-values are obtained by estimating Z = ∑ D i $\_{=1}$w$\_{i}$ Z$\_{i}$ , where Z$\_{i}$ = Φ - $^{1}$(1 - p$\_{i}$ ) with Φ the standard normal cumulative distribution function. This weighted Z-score relies on weights w$\_{i}$ = √ N$\_{i}$ ∑ D i $\_{=1}$w 2 i that are proportional to N$\_{i}$ the number of subjects in the dataset i . Combined measures of the effect sizes are obtained by S = ∑ D i $\_{=1}$w$\_{i}$ s$\_{i}$ weighted average of SMD.

## 2.5. Code Carbon

The assessment of the environmental impact of research is gathering momentum, yet the methodology remains a topic of heavy debate, particularly in computer science and machine learning [71].

Within these fields, a significant environmental impact stems from the production processes of devices (such as acquisition devices, computers, GPUs, and clusters) and the energy consumption during their usage. Evaluating the impact of device production poses challenges, given the limited information shared by manufacturers, potential inaccuracies in estimations, and the frequent sharing and reusing of research equipment across multiple projects. This issue is notably prevalent for CPU and GPU clusters, where numerous experiments run in parallel [76].

Energy consumption during usage is commonly acknowledged as a key indicator of the environmental impact, although it represents only a fraction of the overall impact. This measurement is typically expressed as grams of CO2 equivalent (gCO2) emissions released into the atmosphere and heavily relies

on factors such as the power grid setup and energy production localization. National power companies typically offer estimates of the carbon footprint of consumed energy, articulated in gCO2 per Watt-hour. This value can significantly vary between countries based on energy production sources; for instance, countries reliant on coal or fuel power plants tend to have a higher carbon footprint compared to those utilizing hydroelectric power or solar panels.

For optimal measurement of energy consumption and carbon footprint, power meters are ideal, albeit encompassing the entire computer's energy consumption. To gain more precise insights, especially for evaluating specific programs or machine learning pipelines, software-based power meters prove useful [53]. Various tools exist for this purpose, many of which are open source. In our study, we opted to employ Code Carbon [35], a Pythonbased tool that seamlessly integrates into experiments conducted on individual computers or clusters.

When estimating the carbon footprint of a machine learning pipeline, it is critical to analyze both the training and inference phases. During k-fold validation, inference constitutes a substantial portion of energy consumption. Consequently, the estimated carbon footprint offers valuable insights into the trade-offs between different algorithms, with considerations for CPU consumption, execution time, and the parallelization of algorithms. Notably, wellparallelized algorithms can achieve lower execution times when distributed across a large number of CPUs. The carbon footprint serves as an insightful metric for assessing the computational complexity of a pipeline, particularly when considering its adaptability to embedded or dedicated architectures, common in commercial neurotechnology products.

## 3. Datasets

There are different types of BCI applications, referred to as paradigms throughout this article, each relying on distinctive neurological patterns to facilitate communication between the brain and the BCI system. Depending on the selected paradigm, the raw EEG recordings are transformed into trials for machine learning pipelines. These transformations include bandpass filtering, signal cropping based on stimulus events, and potential resampling to adjust the sampling frequency if needed by the machine learning pipelines. Employing standardized preprocessing procedures enables a fair comparison among different algorithms.

## 3.1. Motor Imagery

The MI paradigm involves a cognitive process where an individual mentally simulates the execution of a motor action without actually performing it. This paradigm is widely used in neuroscience to delve into the neural mechanisms governing motor control and learning. Moreover, it finds applications in neurorehabilitation to assist in restoring motor functions for individuals with neurological disorders or injuries. Various tasks are associated with this paradigm in Mother Of All BCI Benchmark (BCI), with common examples including left-hand, right-hand, and feet imagery. The choice of evaluation metrics for classification performance varies based on the number of tasks involved in the classification process: ROC-AUC is used for two-task classifications, whereas accuracy metrics are employed for multiclass scenarios. With the MI paradigm, signal processing includes bandpass filtering within the [8 - 32] Hz frequency range [86].

The Table 1 provides a comprehensive

overview of MI datasets, with class name abbreviations including RH (Right Hand), LH (Left Hand), F (Feet), H (Hands), T (Tongue), R (Resting State), LHRF (Left Hand Right Foot), and RHLF (Right Hand Left Foot). The column 'No. trials' denotes the number of trials per class, session, and run. For instance, BNCI2014 001 comprises 12 trials per class across 4 classes, 6 runs, and 2 sessions, resulting in a total of 12 × 4 × 6 × 2 = 576 trials in the dataset. The only exception is for the PhysionetMI dataset, where in the first 3 runs, there are RH, LH, and R events; in the last 3 runs, there are H, F, and R events.

## 3.2. P300/ERP

The P300 paradigm, which is a specific ERP paradigm, serves as a framework for categorizing psychophysical experiments [77]. It provides a visual representation of a distinct component characterized by a prominent positive deflection occurring approximately 300 ms after stimulus onset. Typically, this component is elicited in cognitive tasks involving unpredictable and infrequent changes in stimuli. This is improperly called P300 ERP in the BCI community (see P300 BCI or P300 speller), whereas the cognitive components are more diverse than just the P300 wave. See [77] for a detailed discussion about this point.

In the present study, our primary focus is on P3b, a specific subtype that examines stimulus changes relevant to the task at hand. This particular brain wave response manifests when the cognitive tasks involve stimuli that are predictable to some extent but are still imbued with an element of unpredictability. As is commonly the case in the literature, we classify events as targets or non-targets, i.e. , at the epoch level, resulting in a binary classification task. We evaluate

Table 1: Overview of the Motor Imagery EEG datasets available in$\_{MOABB}$.

| Motor imagery           | Motor imagery   | Motor imagery   | Motor imagery   | Motor imagery             | Motor imagery   | Motor imagery   | Motor imagery         | Motor imagery   | Motor imagery                     |
|-------------------------|-----------------|-----------------|-----------------|---------------------------|-----------------|-----------------|-----------------------|-----------------|-----------------------------------|
| Dataset                 | No. subj.       | No. ch.         | No. classes     | No. trials /session/class | Trial len.(s)   | S.freq. (Hz)    | No. sessions          | No. runs        | Classes                           |
| AlexMI [8]              | 8               | 16              | 2(3)            | 20 ± 0                    | 3               | 512             | 1                     | 1               | RH, F, (R)                        |
| BNCI2014 001 [111]      | 9               | 22              | 3(4)            | 72 ± 0                    | 4               | 250             | 2                     | 6               | RH, LH, F, (T)                    |
| BNCI2014 002 [109]      | 14              | 15              | 2               | 80 ± 0                    | 5               | 512             | 1                     | 8               | RH, F                             |
| BNCI2014 004 [70]       | 9               | 3               | 2               | 72.4 ± 9.5                | 4.5             | 250             | 5                     | 1               | RH, LH                            |
| BNCI2015 001 [38]       | 12              | 13              | 2               | 100 ± 0                   | 5               | 512             | 3 subj. 8-11 2 others | 1               | RH, F                             |
| BNCI2015 004 [102]      | 9               | 30              | 2(5)            | 39.4 ± 1.6                | 7               | 256             | 2                     | 1               | RH, F                             |
| Cho2017 [30]            | 52              | 64              | 2               | 101.2 ± 4.7               | 3               | 512             | 1                     | 1               | RH, LH                            |
| Lee2019 MI [69]         | 54              | 62              | 2               | 50                        | 4               | 1000            | 2                     | 1               | RH, LH                            |
| GrosseWentrup2009 [44]  | 10              | 128             | 2               | 150 ± 0                   | 7               | 500             | 1                     | 1               | RH, LH                            |
| PhysionetMI [101]       | 109             | 64              | 4(5)            | 22.6 ± 1.3                | 3               | 160             | 1                     | 6***            | RH, LH, H, F, (R)                 |
| Schirrmeister2017 [103] | 14              | 128             | 3(4)            | 240.8 ± 37.7              | 4               | 500             | 1                     | 2               | RH, LH, F, (R)                    |
| Shin2017A [105]         | 29              | 30              | 2               | 10 ± 0                    | 10              | 200             | 3                     | 1               | RH, LH                            |
| Weibo2014 [125]         | 10              | 60              | 4(7)            | 79 ± 3                    | 4               | 200             | 1                     | 1               | RH, LH, H, F, (LHRF), (RHLF), (R) |
| Zhou2016 [128]          | 4               | 14              | 3               | 50 ± 3.5                  | 5               | 250             | 3                     | 2               | RH, LF, F                         |

its performance with the ROC-AUC metric, which handles the inherent imbalance in the problem at hand. With the ERP paradigm, the signal is bandpass filtered to the 1-24 Hz frequency band [77].

The Table 2 shows an overview of all the ERP datasets. The column "No. epochs NT/T" indicates the number of NonTarget and Target epochs per session and run.

## 3.3. SSVEP

Steady State Visually Evoked Potentials are generated when presenting repetitive sensory stimuli to the subject. While tactile and auditory stimulations are seldom used, visual stimulation is very common, both for control [28] or cognitive probes [123]. The frequency of the stimulus repetition induces a brain oscillation in the associated sensory area. The amplitude of the generated oscillation follows the 1/f law,

meaning that stimulation in low frequency (57 Hz) induces responses of higher amplitude than higher frequency (20-25 Hz). Stimulation above 40 Hz could be difficult to detect due to the weak generated oscillations. In BCI applications, SSVEPs have been used for building spellers [83] and for button-pressing [56], but those applications are limited by the number of available frequencies of stimulation. It is possible, using systems with precisely synchronized stimulation and recording, to encode information both in frequency and phase, therefore multiplying the choices possible [85]. Similarly to the MI paradigm, we evaluate the classifiers using the ROC-AUC metric if only two classes are used and the accuracy metric if there are more. With the SSVEP paradigm, the signal is bandpass filtered to the 7-45 Hz frequency band [29]. The Table 3 encompasses all the SSVEP datasets considered in this study.

Table 2: Overview of the ERP EEG datasets available in$\_{MOABB}$.Table 3: Overview of the SSVEP EEG datasets available in$\_{MOABB}$.

| P300 / ERP              | P300 / ERP   | P300 / ERP   | P300 / ERP                 | P300 / ERP    | P300 / ERP   | P300 / ERP               | P300 / ERP   | P300 / ERP   |
|-------------------------|--------------|--------------|----------------------------|---------------|--------------|--------------------------|--------------|--------------|
| Dataset                 | No. subj.    | No. ch.      | No. epochs NT/T /session   | Epoch len.(s) | S.freq. (Hz) | No. sessions             | No. runs     | Keyboard     |
| BI2012 [116]            | 25           | 16           | 638.2 ± 1.9/127.6 ± 0.7    | 1             | 128          | 1                        | 1            | 36 aliens    |
| BI2013a [115, 10, 32]   | 24           | 16           | 400.3 ± 2.3/80.1 ± 0.5     | 1             | 512          | 8 subj. 1-7 1 subj. 8-24 | 1            | 36 aliens    |
| BI2014a [64]            | 64           | 16           | 794.5 ± 276.7/158.9 ± 55.3 | 1             | 512          | 1                        | 1            | 36 aliens    |
| BI2014b [65]            | 37           | 32           | 201.3 ± 61.5 /40.3 ± 12.3  | 1             | 512          | 1                        | 1            | 36 aliens    |
| BI2015a [62]            | 43           | 32           | 461.8 ± 220.9/92.3 ± 44.1  | 1             | 512          | 3                        | 1            | 36 aliens    |
| BI2015b [63]            | 44           | 32           | 2158.7 ± 6.3/479.9 ± 0.3   | 1             | 512          | 1                        | 4            | 36 aliens    |
| BNCI2014 008 [94, 39]   | 8            | 8            | 3500 ± 0/700 ± 0           | 1             | 256          | 1                        | 1            | 36 char.     |
| BNCI2014 009 [3]        | 10           | 16           | 480 ± 0/96 ± 0             | 0.8           | 256          | 3                        | 1            | 36 char.     |
| BNCI2015 003 [46]       | 10           | 8            | 2250 ± 1500 /270 ± 60      | 0.8           | 256          | 1                        | 2            | 36 char.     |
| EPFLP300 [49]           | 8            | 32           | 685.2 ± 16.9/137.2 ± 3.5   | 1             | 2048         | 4                        | 6            | 6 images     |
| Huebner2017 [50]        | 13           | 31           | 3275.3 ± 2.1 /1007.8 ± 0.6 | 0.9           | 1000         | 2 subj. 6 3 others       | 9            | 42 char.     |
| Huebner2018 [51]        | 12           | 31           | 3638.4 ± 7.7 /1119.6 ± 2.5 | 0.9           | 1000         | 3                        | 10           | 42 char.     |
| Lee2019 ERP [69]        | 54           | 62           | 3450/690                   | 1             | 1000         | 2                        | 1            | 36 char.     |
| Sosulski2019 [106, 108] | 13           | 31           | 75 ± 0 /15 ± 0             | 1.2           | 1000         | 4 subj. 1 3 others       | 20           | 2 tones      |
| Cattan2019 VR [24]      | 21           | 16           | 600 ± 0/120 ± 0            | 1             | 512          | 2                        | 60           | 36 crosses   |

| SSVEP              | SSVEP     | SSVEP        | SSVEP       | SSVEP                                                                                     | SSVEP         | SSVEP        | SSVEP   | SSVEP     | SSVEP                                     | SSVEP             |
|--------------------|-----------|--------------|-------------|-------------------------------------------------------------------------------------------|---------------|--------------|---------|-----------|-------------------------------------------|-------------------|
| Dataset            | No. subj. | No. channels | No. classes | No. trials /session/class                                                                 | Trial len.(s) | S.freq. (Hz) |         | No. sess. | No. runs                                  | Classes           |
| Lee2019 SSVEP [69] | 54        | 62           | 4           | 25                                                                                        | 1             | 1000         |         | 2         | 1                                         | 4 (5.45-12)       |
| MAMEM1 [87]        | 10        | 256          | 5           | 16.8 ± 3.5 classes 8.57,10.0 21.0 ± 4.4 classes 6.66,7.5,12.0                             | 3             | 250          |         | 1         | 3 subj. 1,3,8; 4 subj. 4,6 5 others       | 5 (6.66-12.00)    |
| MAMEM2 [87]        | 10        | 256          | 5           | 20 class 12.0; 30 class 8.57 25 others                                                    | 3             | 250          |         | 1         | 5                                         | 5 (6.66-12.00)    |
| MAMEM3 [87]        | 10        | 14           | 4           | 20.0 ± 0.0 class 6.66; 25.0 ± 0.0 class 8.57 30.0 ± 0.0 class 10.0; 25.0 ± 0.0 class 12.0 | 3             | 128          | 1       |           | 10                                        | 4 (6.66-12.00)    |
| Nakanishi2015 [84] | 9         | 8            | 12          | 15.0 ± 0.0                                                                                | 4.15          | 256          |         | 1         | 1                                         | 12 (9.25-14.75)   |
| Kalunga2016 [56]   | 12        | 8            | 4           | 20.0 ± 7.7                                                                                | 2             | 256          | 1       |           | 5 subj. 12; 4 subj.10 3 subj. 7; 2 others | 4 (13,17,21,rest) |
| Wang2016 [120]     | 34        | 62           | 40          | 6.0 ± 0.0                                                                                 | 5             | 250          | 1       |           | 1                                         | 40 (8-15.8)       |

The Figure 2 displays an embedding of all the datasets in 2 dimensions, using UMAP for dimensionality reduction on all feature information regarding the datasets, as listed in Tables 1, 2 and 3. The color indicates the paradigms (blue for Motor Imagery, green for Event Related Potential, and red for Steady State Visually Evoked Potential), and the name of each dataset is written on top of the circle. The color intensity is related to

the number of electrodes, datasets with a low number of electrodes are in lighter color, and the size of the circles is proportional to the number of subjects. The UMAP embedding preserves the local topology. This highlights that, despite different paradigms, datasets with many electrodes and many subjects are in a central position. Datasets with fewer subjects and fewer electrodes are closer to the border of the figure. The

BNCI2014 001 dataset, commonly called BCI Competition IV dataset 2a, is the most widely used in BCI literature. There are closely related datasets with roughly the same number of subjects, electrodes, and trials (BNCI2015 004, BNCI2015 001), with more subjects (Shin2017A) or with fewer subjects (Zhou2016). This group of datasets is useful for the fast evaluation of new pipelines and to see how well results generalize with the same kind of dataset. It is also possible to ensure a good coverage of the dataset features, using only a few datasets. With MI, a selection of BNCI2014 001, BNCI2014 004, Schirrmeister2017, and PhysioNetMI might be sufficient to evaluate an approach on datasets with very different configurations.

## 4. Pipelines

As discussed in section 1, most classification algorithms in BCI research for EEG signals fall into one of three main categories: those based on raw signals (referred to as "Raw" hereafter), algorithms relying on covariance matrices seen as elements of a Riemannian manifold (denoted "Riemannian") and the Deep Learning (DL) approaches.

The Raw signal methods typically employ supervised spatial filters to simultaneously enhance the component related to the cognitive task while reducing the dimensionality of the EEG data. In contrast, Riemannian pipelines consider the signal through its estimated covariance matrices, leveraging the natural metric acting on the curved geometry of SPD matrices, which remains invariant by congruence transformations [124, 33]. Those approaches are thus mostly invariant to any spatial transformations applied on the signal, making them highly effective in BCI applications.

Lastly, deep neural networks learn spatial

and temporal filters directly from raw EEG data. Although there is a wealth of literature on deep learning models, there are few models available or evaluated with reproducible frameworks. Initiatives like Braindecode [103] or, to a lesser extent for BCI, torchEEG [127], offer an open-source implementation of the most efficient models. This study considers a set of raw, Riemannian, and deep learning models, which are detailed in Table 4, along with the hyperparameters used for the grid-search approach, listed in the annexes within Table A1.

## 4.1. Raw signal

The category of pipelines referred to as Raw consists of BCI classifiers employing traditional statistical analysis, as well as temporal and/or spatial filtering tools to extract features. Variance-based pipelines represent one of the initial BCI pipeline concepts proposed for motor imagery. Several approaches have highlighted the value of utilizing intra-channel variance for online decoding tasks. These pipelines calculate the variance of each electrode within an epoch to create a positive definite realvalued vector. To address noise or artifacts, it is common practice to logarithmically transform the observed variance [74]. This approach essentially boils down to only considering the diagonal elements of the covariance matrices. Classifiers such as Linear Discriminant Analysis (LDA) or Support Vector Machine (SVM) can be utilized on the resulting vector to predict the label of the epoch.

Common Spatial Pattern (CSP) approach learns spatial filtering matrices in a supervised manner, minimizing the variance of the band power feature vectors within the same

Dataset visualization

Figure 2: Visualization of the MOABB datasets, with Motor Imagery in green, Event Related Potential in pink/purple and Steady State Visually Evoked Potential in yellow/brown. The size of the circle is proportional to the number of subjects and the contrast depends on the number of electrodes.

<!-- image -->

Table 4: Pipelines considered in this study, the color indicates the paradigm. Green is for motor imagery, pink for P300 and orange for SSVEP.

| Pipeline Name                | Category   | References   | Pipeline Name       | Category      | References   |
|------------------------------|------------|--------------|---------------------|---------------|--------------|
| LogVar + LDA                 | Raw        |              | TS + LR             | Riemannian    | [12]         |
| LogVar + SVM                 | Raw        |              | TS + SVM            | Riemannian    | [12]         |
| CSP + LDA                    | Raw        | [61, 19]     | ACM + TS + SVM      | Riemannian    | [22]         |
| CSP + SVM                    | Raw        | [61, 19]     | ShallowConvNet      | Deep Learning | [103]        |
| TRCSP + LDA                  | Raw        | [73]         | DeepConvNet         | Deep Learning | [103]        |
| DLCSPauto + shLDA            | Raw        | [61, 19],    | EEGNet 8 2          | Deep Learning | [67]         |
| FBCSP+SVM                    | Raw        | [2]          | EEGTCNet            | Deep Learning | [52]         |
| FgMDM                        | Riemannian | [12]         | EEGITNet            | Deep Learning | [100]        |
| MDM                          | Riemannian | [13]         | EEGNeX 8 32         | Deep Learning | [27]         |
| TS + EL                      | Riemannian | [34]         |                     |               |              |
| XDAWN + LDA                  | Raw        | [95]         | XDAWNCov + TS + SVM | Riemannian    | [28]         |
| XDAWNCov + MDM               | Riemannian | [9]          | ERPCov + MDM        | Riemannian    | [10]         |
| ERPCov( svd$\_{n}$ = 4) + MDM | Riemannian | [10]         |                     |               |              |
| TRCA                         | Raw        | [85]         | SSVEP MDM           | Riemannian    | [29]         |
| CCA                          | Raw        | [72]         | SSVEP TS + LR       | Riemannian    | [29]         |
| MsetCCA                      | Raw        | [126]        | SSVEP TS + SVM      | Riemannian    | [29]         |

class, while maximizing the between-class variance [81, 19]. To enhance the robustness of the CSP against noise and overfitting, Tikhonov Regularized CSP (CSP) approach has been proposed in [73]. In contrast to the classical CSP, which uses band-pass filtered EEG signals that may vary per subject, Filter-Bank CSP (CSP) addresses this issue by extracting CSP features for each band-pass filter from the filter bank [2]. The subsection Appendix B.1 details the CSP algorithm.

Canonical Correlation Analysis (CCA) has emerged as a prominent approach for classifying SSVEP signals, initially introduced in the work by [72]. Subsequently, CCA has been successfully employed in numerous notable SSVEP-based BCI studies, such as those conducted by [18] and [83]. SSVEP signals exhibit correlation to flickering visual stimuli, with their signal phase and frequency corresponding to stimulus characteristics. Leveraging this relationship, CCA aids in extracting EEG spatial components with the strongest correlation to SSVEP stimuli. Further details on the different pipelines can be found in subsection Appendix B.2.

## 4.2. Riemannian geometry

The introduction of Riemannian geometry into BCI processing marked a pivotal moment for the BCI community [13]. The fundamental concept underlying this approach is to represent the signal using covariance matrices or their derivatives. As covariance matrices are Symmetric Positive Definite (SPD) matrices, they live in a Riemannian space [124]. We provide here a general description of the framework needed to define algorithms based on the Riemann distance for classification tasks. For a more in-depth description of the Riemannian

framework, we refer the reader to [20] which gives a pedagogical introduction to these concepts.

Due to the curvature of the SPD matrix space, traditional Euclidean geometry is ill-suited and introduces a swelling effect. Particularly, the Euclidean distance could result in a wrong characterization of the relationship between SPD matrices. Instead, Riemannian methods rely on a distance that respects the geometry of the SPD matrices space, based on geodesics, i.e., the shortest path that connects 2 elements and stays in the space of SPD matrices. A common choice of distance is the affine-invariant one [79]. Considering the manifold of SPD matrices M$\_{n}$ = { P ∈ R n × $^{n}$| P = P $^{⊤}$and x $^{⊤}$P x > 0 , ∀ x ∈ R $^{n}$} , the affine-invariant distance for P$\_{1}$ , P$\_{2}$ ∈ M is defined as

δ$\_{R}$ ( P$\_{1}$ , P$\_{2}$ ) = [ n ∑ i =1 log 2 λ$\_{i}$ $^{(}$P - 1 1 P$\_{2}$ ) ] 1 / 2 (1)

where λ$\_{i}$ ( P ) is the i -th eigenvalues of P .

Similarly, the concept of the mean in Riemannian geometry must be redefined to ensure it belongs to the manifold; it is known as the Frechet mean [79] and is defined as:

ˆ G = argmin P ∈ P ( n ) m ∑ i =1 δ 2 $\_{R}$( P , P$\_{i}$ ) (2)

As we are operating within a Riemannian manifold, traditional machine-learning classification algorithms cannot be directly applied. Instead, there are two options: either create new algorithms to classify SPD matrices on the Riemannian manifold or map the matrices to an associated Euclidean space and then apply standard classification algorithms. In the former scenario, algorithms like Minimum Distance to Mean give robust accuracy. In the latter scenario, it is possible to rely on the tangent

space to a point of the manifold. It is possible to circulate between the manifold and the tangent space using the Log and Exp map functions. The Log (resp. Exp) maps the manifold to the tangent space (resp. the tangent space to the manifold):

Exp$\_{P}$ ( S$\_{i}$ ) = P 1 / $^{2}$Exp $^{(}$P - 1 / $^{2}$S$\_{i}$ P - 1 / 2 $^{)}$P 1 / 2 (3)

Log$\_{P}$ ( P$\_{i}$ ) = P 1 / $^{2}$Log $^{(}$P - 1 / $^{2}$P$\_{i}$ P - 1 / 2 $^{)}$P 1 / 2 (4)

Projected in the tangent space, any machine learning algorithm could be applied. One limitation is the size of the considered space, that is n ( n +1) 2 for M$\_{n}$ . Machine learning algorithms like Support Vector Machine (SVM), ElasticNet (EL) or Logistic Regression (LR) are among the most popular for classification in the tangent space. Details regarding the pipelines implementation are available in subsection Appendix B.3.

## 4.3. Deep learning

DL methods have demonstrated in recent years considerable promise in various tasks that involve handling massive volumes of digital data and those in different fields such as computer vision [66] and Natural Language Processing [117, 104].

This also holds for BCI applications. The BCI field has been significantly impacted by the integration of DL techniques [99, 36], which exhibit strong generalization capabilities, with transfer learning emerging as a key focus in BCI research. One notable advantage of DL models is their capacity to leverage vast datasets, a task typically challenging for classical Machine Learning (ML) algorithms. Moreover, by conducting all processing and classification steps within a neural network, DL models enable optimized end-to-end

learning. In this paper, we will focus specifically on DL methods that have been applied to MI paradigms.

The predominant Python libraries for DL are$\_{Tensorflow}$ [1] and PyTorch [88]. To facilitate broader access to $\_{MOABB}$, we developed integration for both libraries using wrappers from Scikeras [42] (for $\_{Tensorflow}$) and$\_{Skorch}$ [114] (for PyTorch). We were also keen on integrating the Braindecode [103] library, which incorporates several DL algorithms for EEG processing using $\_{PyTorch}$.

Most DL architectures for EEG decoding operate on minimally pre-processed (bandpass filters) or raw epochs. To address the spatio-temporal complexity of EEG signals, convolutional layers are commonly employed with separable 1D convolution along the temporal and spatial dimension (i.e., EEG channels). By segregating the spatial and temporal convolutions, these models account for the fact that all EEG channels observe all brain sources.

Although the goal is to minimize preprocessing steps before passing data to DL pipelines, a few steps are usually necessary. Bandpass filtering is applied to ensure a fair comparison between ML and DL methods. Neural network architectures reviewed here are designed for decoding EEG signals at a certain sampling frequency. Resampling datasets to match DL architectures' expected frequencies is performed to avoid interfering with the relative temporal length of their kernels. Standard rescaling of the signal is implemented before DL pipelines in adherence to common deep learning practices [68].

Data augmentation, a technique generating synthetic training samples through transformations, is well-established in computer vision for producing state-of-the-art results [26]. However, in the realm of EEG applications,

data augmentation poses unique challenges. Despite its potential to reduce overfitting and enable complex algorithms, it is still an active area of research. Assessing whether a generated signal accurately captures the physiological attributes of EEG remains an open question. Thus, this study abstained from employing data augmentation procedures in the context of EEG decoding. For a comprehensive examination of various existing techniques for EEG, we refer the reader to [97].

Consistent parameters were used across all DL experiments, training networks for 300 epochs with a batch size of 64, Adam optimizer [60] with a base learning rate of 0 . 001 and cross-entropy as the loss function. An early stopping strategy with a patience parameter of 75 epochs was crucial to prevent overfitting. While the considered DL architecture hyperparameters align with state-of-the-art standards through signal resampling, further optimization could enhance model performance via a grid-search approach tailored to individual scenarios.

## 5. Experimental results

The experimental results outlined in this section encapsulate the key insights gathered during this benchmark study. While we provide only the most important findings to the reader in this section, we also report all the raw results to ensure proper reproducibility. Due to space limitations, all the detailed evaluations, including ROCAUC or accuracy scores, are available in the appendix. Tables D1, D2, D3, D4 and D5 display the average scores of each pipeline across all subjects and sessions within a specific dataset. Additionally, Figures C1, C2, C3, C4 and C5 present the pipeline groups' scores for each subject and session across all

datasets to aid in result interpretation.

## 5.1. Riemannan pipelines outperforms others pipelines

The Riemannian distance-based classification pipeline consistently outperforms results obtained through DL and Raw pipelines across all datasets, on all paradigms. Figure 3 illustrates this superiority in the context of righthand vs left-hand classification, SSVEP and P300. The ROC-AUC and accuracy results are shown based on each pipeline's performance relative to its category (Raw, Riemannian, DL). It should be emphasized that each dot on the plots represents the average of all pipelines from a category on a dataset, each dataset encompassing 10 to 100 subjects. In the distributions, each dataset has the same weight, regardless of the number of subjects or sessions. The results presented here are thus summarizing the largest BCI study to date.

The plots demonstrate the dominance of the Riemannian approaches, showcasing its superior performance not only in overall averages but also consistently across all datasets examined. This trend is further reinforced by the noticeable peak shift in the distribution of results, providing additional confirmation of the effectiveness of the Riemannian approaches. Similar findings are observed across various classification tasks within the Motor Imagery paradigm, as could be seen in the appendix on Figures C1 and C2.

The suboptimal performance of DL pipelines can be attributed to two main factors. Firstly, the hyperparameters of DL pipelines were not fine-tuned for each dataset; instead, the parameters described in the original articles were employed. This marks a significant divergence from the Riemannian and Raw approaches, which had their hyperpa-

rameters optimized through a nested crossvalidation strategy. We could not complete a similar hyperparameter search for DL as the search space is too large and it involves complex changes in the architecture shape, such as the kernel size, or the activation functions that could not be automatized easily. Secondly, we chose not to include any data augmentation steps in our research, as explained in Sect. 4.3. It is noteworthy that such procedures have been demonstrated to exert a substantial impact on performance, as evidenced by previous studies [97]. Still, it required a lengthy and dataset-specific parametrization. It is important to recall that the objective of this study is to assess existing and published algorithms, to evaluate their off-the-shelve performances, and not to investigate how to properly tune them.

## 5.2. Riemannian pipelines work well with limited number of electrodes

Riemannian pipelines perform best in scenarios involving a reduced number of channels for MI tasks. Employing a limited number of electrodes in EEG recordings offers numerous advantages for practical BCI experiments. This approach simplifies the setup, reducing both complexity and cost. Additionally, it enhances user comfort, providing greater flexibility and ease of use in diverse settings, including homebased or mobile applications. However, using fewer electrodes may compromise signal quality and spatial resolution, potentially resulting in lower classification accuracy and reduced robustness against artifacts and noise.

To address these challenges, it becomes crucial to design classification algorithms capable of delivering high performance with a limited number of electrodes. As depicted in Figure 4-(a), the Riemannian pipelines excel in performance with datasets containing

[0 , 25] electrodes. Notably, their performances tend to decrease as the number of channels increases. Conversely, DL pipelines require a substantial amount of information to achieve satisfactory performance, while facing the limitations described earlier, emphasizing the importance of balancing electrode count and classification effectiveness.

As observed, there is a decline in performance in settings with a moderate number of channels. This decrease in effectiveness can be attributed to several factors. Primarily, an increased number of electrodes elevates the problem's complexity, as the ML algorithm needs to extract relevant information more efficiently. Furthermore, this category encompasses datasets with varying average scores, impacting overall performance. Conversely, scenarios with a high number of electrodes exhibit less pronounced detrimental effects. This could be a side effect; a large number of electrodes implies highergrade equipment and specialized technicians. The enhanced data recording procedures and superior data quality associated with a larger number of channels might thus alleviate previous issues.

Classification algorithms based on Riemannian distances can be implemented in two distinct approaches. The first method involves performing classification directly on the Riemannian manifold, using algorithms like the MDM (Minimum Distance to Mean) algorithm. The alternative approach involves projecting data onto the tangent space and conducting classification using conventional ML algorithms (SVM, LR, EL). Comparing the two strategies, it is observed that the Riemannian method based on Tangent space projection consistently outperforms the approach centered on the Riemannian surface as shown in Figure 4-(b).

<!-- image -->

Figure 3: Average performance of pipelines grouped by category (Deep Learning, Riemannian, and Raw) across the MI (right-hand vs left-hand), SSVEP, and ERP paradigms displayed as raincloud plots. Each point in the plot corresponds to the average score of one dataset across all pipelines within a specific category, encompassing all subjects and sessions.

<!-- image -->

Figure 4: (a) AUC scores are averaged across all sessions, subjects, and datasets within the right-hand vs left-hand MI paradigm for each category (Deep Learning, Riemannian, Raw), segmented by the number of channels on the y-axis. Box plots overlaid with strip plots show individual ROC-AUC scores. (b) Distribution of AUC scores for the Riemannian MI pipelines is depicted for the right-hand vs left-hand classification task. The boxes and horizontal black bars denote quartile ranges.

<!-- image -->

## 5.3. Deep learning requires a high number of trials

It is essential to emphasize that the lower performance of DL pipelines, in comparison to the other pipeline categories, is closely linked to the specific DL architecture under consideration. This observation is illustrated in Figure 5-(a). Amongst DL pipelines, ShallowConvNet stands out with the highest AUC. Interestingly, a distinctive dichotomy appears within DL models. The first group, consisting of ShallowConvNet, EEGNet-8.2, and DeepConvNet, exhibits superior performance. In contrast, the second group, gathering EEGTCNet, EEGITNet, and EEGNeX, shows lower performance levels. This divergence in performance is likely attributed to the optimization of hyperparameters in the models. It demonstrates that DL models that have been more extensively tested, and hence correctly parametrized, yield higher classification performance on all datasets.

The number of trials employed in training algorithms carries particular importance, specifically in the context of DL pipelines. A clear trend emerges when evaluating DL algorithm performance, indicating that achieving satisfactory results typically requires more than 150 trials per class, as shown in Figure 5(b). Again we observe the same two distinct groups described above, the group comprising ShallowConvNet, EEGNet-8.2, and DeepConvNet exhibits a higher resilience to the impact of the number of trials. Notably, this group manages to achieve already satisfactory performance with as few as 50 trials per class, showcasing a relatively robust response to variations in the training dataset size, highlighting the distinct capabilities of certain architectures to yield superior performance with a more limited number of trials.

## 5.4. Recommended number of trials depends on the task complexity

In scenarios where tasks exhibit a clear distinguishability, such as the MI task involving right-hand/feet movements, achieving impressive AUC performance with a limited number of trials is feasible, as clearly depicted in Figure 6a. Notably, both the Raw and Riemannian pipelines demonstrate exceptional classification scores even with fewer than 50 trials. Increasing the trial count beyond 50 does not yield significant improvements in AUC results for these pipelines. On the other hand, for DL pipelines to reach good AUC, datasets associated with more than 150 trials are imperative to achieve satisfactory AUC scores, highlighting the pivotal role trial quantity plays in DL model performance. This discrepancy highlights the varying requirements and efficiencies of different pipeline approaches based on the task complexity and nature of the data.

When dealing with tasks of higher complexity, such as MI paradigms involving 3 to 7 distinct classes (refer to Section 3.1 for detailed explanation), obtaining optimal accuracy becomes considerably more challenging. In such intricate tasks, a larger number of trials is imperative for pipelines to achieve the highest levels of accuracy. This phenomenon is illustrated in Figure 6b, where the accuracy variability across different datasets becomes apparent. The fluctuation in accuracy levels observed here is intricately tied to the unique characteristics and quality of individual datasets, portraying the nuanced dynamic between dataset quality, trial quantity, and the performance of the classification pipelines. Specifically, the noticeable decline in accuracy for datasets with over 200 trials compared to datasets with 100 to 200 trials underscores the diverse demands and responses of different

Figure 5: (a) Distributions of AUC scores averaged over all datasets for the right-hand vs lefthand classification task within the DL pipelines. (b) AUC scores averaged across all sessions, subjects, and datasets within the right-hand vs feet MI paradigm for the DL pipelines, segmented based on the number of epochs on the y-axis.

<!-- image -->

pipelines to varying trial quantities in complex classification tasks.

## 5.5. Best practices in motor imagery

Motor imagery is a widely utilized paradigm in the BCI community, and a detailed analysis was conducted on the MI results obtained in this benchmark to assist practitioners in experimental design and pipeline selection.

The highest performance levels in MI are achieved in the binary classification distinguishing between right-hand and feet movements. This task notably surpasses the performance of the right-hand/left-hand classification, highlighting a significant disparity in performance between the two tasks. This performance discrepancy is visible when comparing the results of right-hand/left-hand and righthand/feet tasks in the supplementary Fig-

ures C1 and C2, respectively. This trend is consistent across all pipeline categories Raw, Riemannian and DL - and is prevalent across datasets, with 9 datasets for righthand/feet tasks and 10 for right-hand/lefthand tasks. Five datasets are common to both tasks, with the AUC notably higher for the right-hand/feet task.

This results holds significant implications for the BCI community, as it provides for the first time valuable insight into selecting the task that yields the highest accuracy in MI. The findings presented here are highgly reliable, drawn from a diverse range of subjects recorded under various protocols and using different hardware setups. Furthermore, since this trend is observable even when analyzing subjects performing both tasks, any confounding effects related to recording

<!-- image -->

(a) right-hand/feet

Figure 6: (a) AUC scores for datasets segmented by the number of trials in the right-hand vs feet MI paradigm for each pipeline category ( Deep learning, Riemannian, Raw ). (b) Same results for MI using all available classes and the accuracy metric.

<!-- image -->

conditions can be effectively eliminated. This insight holds particular importance for BCI practitioners when designing experimental protocols.

To delve deeper into how different pipelines compare in the right-hand/feet classification task, Figure 7-(a) illustrates the AUC scores of the top three pipelines in each category, arranged by their average scores per dataset. The very good performances of Raw and Riemannian pipelines are visible on this plot. The performances of the DL pipelines are below Raw and Riemannian pipelines for the reasons outlined previously.

While the AUC performance measurements offer valuable insights, Figure7-(a) reveals significant subject variability. In practical BCI applications, the choice of the pipeline is often influenced by the algorithm's ranking, with the best-performing algorithm selected for each subject. To evaluate how pipelines fare based on this criterion, pipelines were ranked in each session for all subjects, ranging from 1 (the best) to 16 (the worst) based on

their relative scores. Figure 7-(b)visualizes the frequency of sessions (y-axis) where a pipeline achieves a specific rank (x-axis), using distinct colors for each pipeline.

While the results align with the average AUC scores presented in Figure 7-(a), they provide different qualitative insights. Firstly, the ACM+TS+SVM is outperforming other pipelines, consistently securing top positions and rarely falling below the 7th spot. Secondly, the TS+LR and TS+EL Riemannian pipelines exhibit comparable AUC scores, yet TS+EL consistently outperforms other pipelines, whereas TS+LR seldom claims the top spot but frequently ranks among the top 3 pipelines. Lastly, DL pipelines generally rank below the 9th position, excluding the notable exception of the ShallowConvNet pipeline, which attains top 3 rankings in several sessions, despite its lower average AUC score placing it behind CSP-based pipelines.

Accuracy is a primary criterion for selecting a BCI pipeline, but it's also essential to consider calibration time. Pipelines

are often trained immediately following a calibration run or updated after running multiple trials, making execution time a crucial factor in pipeline selection. User interaction is typically paused during pipeline training on the training dataset, emphasizing the significance of efficient execution. This aspect is crucial not only for real-time BCI operation but also for offline evaluation and hyperparameter optimization, as it directly impacts experiment duration.

To address this issue, the average execution times for pipeline categories (Raw, Riemannian, and DL) are presented in Figure 8(a). These findings, focused on MI right-hand vs feet classification, are applicable across other tasks and paradigms as well. The measurements were conducted using the French Jean Zay CPU/GPU HPC environment, featuring Intel Cascade Lake 6248 CPUs and Nvidia V100 16 GB RAM, encompassing both training and inference phases for a single fold of cross-validation.

Raw pipelines demonstrate the shortest computational time requirements, closely followed by Riemannian pipelines. DL pipelines, on the other hand, exhibit longer training durations on average but remain within an acceptable 30-second range for experimental systems. It's noteworthy that these results were obtained using an early stopping strategy to prevent overfitting, which also contributes to reducing training time.

Environmental impact assessment is crucial in AI-related domains given the exponential growth in these areas. In this study, the direct environmental impact, measured in gCO2 equivalent generated during training and inference phases, is evaluated. The absolute values are significantly influenced by the country's energy production methods, hence the CPU/GPU server localization are important to

measure the generated gCO2 equivalent. Precisely measuring algorithm energy consumption is challenging, as existing libraries differ in solutions and measurements. Using Code Carbon [35], the environmental footprint, expressed in gCO2 equivalent emissions, for Riemannian TS+EL, Raw CSP+SVM, and ShallowConvNet pipelines is documented in Figure 8-(b). This provides a unified measure of required computational resources, illustrating that the Riemannian pipeline consumes less energy despite its longer training times as shown in Figure 8-(a).

## 6. Future Directions for reproducible BCI machine learning pipelines

The path towards open and reproducible approaches in BCI improved with initiatives in open EEG hardware [40, 21], libraries for experimental design [93, 90, 31] and, indeed, machine learning pipelines [4, 14]. For the latter, there is still room for improvements, along two main axes. The first one is to get closer to experimental situations and the second one is to allow fast benchmarking of the most recent BCI decoding techniques. An initiative ‡ to tackle the second axes relies on benchopt [80] and aims to provide an easy environment to evaluate novel BCI techniques along with reproducible evaluation conditions.

## 6.1. Pseudo-online benchmarking

The first limitation is that, in order to maintain compatibility with numerous datasets, the inherent chronology between epochs is disregarded, and the evaluation within each session consists of a simple 5-fold cross-validation over shuffled epochs. However, by leveraging historical knowledge, certain unsupervised

<!-- image -->

Figure 7: (a) AUC scores are presented for the best three motor imagery pipelines in each category for the right-hand vs feet classification task, ordered by their average score per dataset. (b) Pipeline rankings within individual sessions for the right-hand vs feet task, with each pipeline color-coded. The x-axis displays different ranks achieved by pipelines, while the y-axis indicates the number of sessions each pipeline achieves a specific rank.

<!-- image -->

Figure 8: (a) Average execution times in seconds per dataset for the MI right-hand vs feet paradigm, segmented by pipeline category (DL, Riemannian, Raw). (b) Carbon emissions in gCO2 equivalent for the high-ranking pipelines (Raw CSP+SVM, Riemannian TS+EL, DL ShallowConvNet) in the MI right-hand vs feet task.

<!-- image -->

classification techniques can rival supervised ones [51, 107]. This evaluation presents a significant challenge as it completely overlooks the non-stationarity of the data, causing some classifiers to perform well under these conditions but fail in an online scenario. Consequently, in the near future, it is important to integrate pseudo-online benchmarking of algorithms, like along the lines proposed in [23].

In real online experiments, we anticipate a decrease in accuracy with excessively long calibration periods. User feedback could prove to be highly motivating, a factor that remains undisclosed in offline evaluations.

Regarding pseudo-online evaluation, we hypothesize that a gradual distribution shift occurs during the session. Achieving high accuracy becomes considerably more challenging as training data only capture a limited range of subject variability. Additionally, we anticipate observing learning curves that ascend and level off due to this shift; a plateau phenomenon not observed in offline analyses, where accuracy appears to increase as more data is amassed.

## 6.2. CVEP paradigm

Most open data in BCI include MI, ERP, and SSVEP paradigms. A recent addition to the landscape of evoked potentials, alongside the well-established ERP and SSVEP scenario, is the emerging Code-modulated Visually Evoked Potentials (c-VEP) paradigm [78]. Drawing an analogy to telecommunications, these three evoked paradigms can be likened to time-domain, frequency-domain, and codedomain multiple access schemes [41]. Notably, research has demonstrated that the c-VEP paradigm exhibits superior performance compared to both ERP and SSVEP paradigms [17], garnering increasing attention and leading to the development of high-speed

BCIs measured through Information Transfer Rate (ITR); see, for instance, [82, 113, 110]).

## 6.3. Character-level benchmarking of ERP and c-VEP

The decoding algorithms for ERP (and soon c-VEP) are mostly benchmarked at the epoch classification level, typically addressing a binary problem where the algorithms need to predict whether an epoch corresponds to a target or a non-target in the original application. However, this decoding approach overlooks the specific character or target being attended to in the original application, despite the availability of information regarding the stimulus sequence used for each character. Recent advancements have introduced methods that leverage application-level information to reduce the number of target hypotheses [59, 16, 15]. Moreover, unsupervised classifiers have been proposed that capitalize on the structural sequence information, showcasing the potential to surpass established, even supervised algorithms [107, 112]. Unfortunately, the emphasis on binary decoding has led to incomplete availability of the necessary structural information in all existing ERP datasets.

## 6.4. Cross-dataset transfer learning

Transfer learning has consistently posed a significant challenge in the realm of BCI [55, 57]. While current support includes benchmarks for cross-session and cross-subject scenarios, the absence of benchmarks across datasets remains a gap. Recent advancements in DL models have yielded remarkably high performances in solving cross-dataset transfer learning challenges, particularly in MI paradigms [58, 45, 121, 5]. This surge in interest toward crossdataset transfer no longer stems solely from

fundamental research but also unfolds as a promising avenue for future BCIs.

## 7. Conclusion

This contribution represents the largest reproducibility study in EEG-based BCI, leveraging the$\_{MOABB}$ library. By utilizing openly available data collected from different hardware sources in varied formats and structures, a systematic benchmarking process was undertaken. Machine learning pipelines from published works were re-implemented in a unified and open framework, aligning with the established standards of the machine learning community. This effort extends to DL pipelines, considering the rapidly evolving processing standards for time series data.

The study's strength lies in the extensive number of subjects analyzed across diverse datasets, enabling robust assessments through meta-analysis statistical techniques. The pipelines undergo evaluation using 5-fold crossvalidation, employing the AUC metric for binary classification tasks and accuracy for datasets with multiple classes. Furthermore, the environmental impact of the pipelines is assessed and factored into the reported results.

The primary outcome is a comprehensive and meticulous benchmarking of prominent pipelines from within the BCI literature. Resources are provided to reproduce these results and facilitate comparisons with future works, including result tables in the annexes and on a dedicated online platform to streamline comparisons and avoid unnecessary duplications. The Riemannian pipelines demonstrate the highest accuracy, whereas DL pipelines, while achieving admirable accuracy with extensive trial data, show limitations across most datasets. Although data augmentation techniques and advanced parameterization can en-

hance their performance, improvements are still required for these off-the-shelf pipelines.

As the benchmark incorporates various pipelines and datasets, recommendations can be formulated regarding the optimal number of trials or channels for designing BCI experiments to achieve peak performance. A detailed analysis of all considered datasets is presented, aiding practitioners in tailoring their experimental designs or selecting specific datasets for evaluating novel pipelines.

Future development avenues are outlined along two key directions. Firstly, benchmarks could progress towards evaluations that mirror real-world experimental conditions, aiming to narrow the disparity between offline assessment and practical online BCI applications. Secondly, the integration of novel BCI paradigms like CVEP and transfer learning approaches across datasets is suggested for further exploration and integration.

## 8. Acknowledgements

SC, BA and SS were supported by DATAIA Convergence Institute as part of the "Programme d'Investissement d'Avenir", (ANR17-CONV-0003) operated by LISN-CNRS. This work was granted access to the HPC resources of IDRIS under the allocation 2023AD011014322 made by GENCI.

## References

- [1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man' e, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens,

- B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Vi' egas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL https://www.tensorflow. org/ .
- [2] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan. Filter bank common spatial pattern (FBCSP) in brain-computer interface. In IEEE IJCNN , pages 23902397, 2008.
- [3] P. Aric' o, F. Aloise, F. Schettini, S. Salinari, D. Mattia, and F. Cincotti. Influence of P300 latency jitter on event related potential-based brain-computer interface performance. Journal of neural engineering , 11(3):035008, 2014.
- [4] B. Aristimunha, I. Carrara, P. Guetschel, S. Sedlar, P. Rodrigues, J. Sosulski, D. Narayanan, E. Bjareholt, B. Quentin, R. T. Schirrmeister, E. Kalunga, L. Darmet, C. Gregoire, A. Abdul Hussain, R. Gatti, V. Goncharenko, J. Thielen, T. Moreau, Y. Roy, V. Jayaram, A. Barachant, and S. Chevallier. Mother of all BCI Benchmarks, 2023. URL https: //github.com/NeuroTechX/moabb .
- [5] B. Aristimunha, R. Y. de Camargo, W. H. L. Pinaya, S. Chevallier, A. Gramfort, and C. Rommel. Evaluating the structure of cognitive tasks with transfer learning. arXiv preprint arXiv:2308.02408 , 2023.
- [6] M. Baker. 1,500 scientists lift the lid on reproducibility. Nature , 533(7604):452454, May 2016.
- [7] H. Banville, O. Chehab, A. Hyvarinen, D.-A. Engemann, and A. Gramfort.
- Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of Neural Engineering , 18(4): 046020, 2021.
- [8] A. Barachant. Commande robuste d'un effecteur par une interface cerveau machine EEG asynchrone . PhD thesis, Grenoble, 2012.
- [9] A. Barachant. MEG decoding using Riemannian geometry and unsupervised classification. Grenoble University: Grenoble, France , 2014.
- [10] A. Barachant and M. Congedo. A plug&play P300 BCI using information geometry. arXiv preprint arXiv:1409.0107 , 2014.
- [11] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Common spatial pattern revisited by riemannian geometry. In IEEE International Workshop on Multimedia Signal Processing , pages 472476, 2010. doi: 10.1109/MMSP.2010. 5662067.
- [12] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Riemannian geometry applied to BCI classification. Lva/Ica , 10:629-636, 2010.
- [13] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Multiclass braincomputer interface classification by Riemannian geometry. IEEE Transactions on Biomedical Engineering , 59(4):920928, 2011.
- [14] A. Barachant, Q. Barth' elemy, J.-R. King, A. Gramfort, S. Chevallier, P. L. C. Rodrigues, E. Olivetti, V. Goncharenko, G. W. vom Berg, G. Reguig, A. Lebeurrier, E. Bjareholt, M. S. Yamamoto, P. Clisson, and M.-C. Corsi. pyriemann, June 2023. URL https: //doi.org/10.5281/zenodo.593816 .

- [15] Q. Barth' elemy, S. Chevallier, R. Bertrand-Lalo, and P. Clisson. End-to-end P300 BCI using Bayesian accumulation of riemannian probabilities. Brain-Computer Interfaces , 10(1): 50-61, 2023.
- [16] L. Bianchi, C. Liti, G. Liuzzi, V. Piccialli, and C. Salvatore. Improving P300 speller performance by means of optimization and machine learning. Annals of Operations Research , pages 1-39, 2021.
- [17] G. Bin, X. Gao, Y. Wang, B. Hong, and S. Gao. VEP-based brain-computer interfaces: time, frequency, and code modulations [research frontier]. IEEE Computational Intelligence Magazine , 4 (4):22-26, 2009.
- [18] G. Bin, X. Gao, Z. Yan, B. Hong, and S. Gao. An online multi-channel SSVEP-based brain-computer interface using a canonical correlation analysis method. Journal of neural engineering , 6(4):046002, 2009.
- [19] B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, and K.-R. Muller. Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal processing magazine , 25(1):41-56, 2007.
- [20] N. Boumal. An introduction to optimization on smooth manifolds . Cambridge University Press, 2023.
- [21] Y. N. Cardona-' Alvarez, A. M. ' AlvarezMeza, D. A. C' ardenas-Pe˜ na, G. A. Casta˜ no-Duque, and G. CastellanosDominguez. A novel OpenBCI framework for EEG-based neurophysiological experiments. Sensors , 23(7):3763, 2023.
- [22] I. Carrara and T. Papadopoulo. Classification of BCI-EEG based on augmented
- covariance matrix. arXiv preprint arXiv:2302.04508 , 2023.
- [23] I. Carrara and T. Papadopoulo. Pseudoonline framework for BCI evaluation: a MOABB perspective using various MI and SSVEP datasets. Journal of Neural Engineering , 21(1):016003, 2024.
- [24] G. Cattan, A. Andreev, P. Rodrigues, and M. Congedo. Dataset of an EEGbased BCI experiment in virtual reality and on a personal computer. arXiv preprint arXiv:1903.11297 , 2019.
- [25] G. C. Cawley and N. L. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal of Machine Learning Research , 11:2079-2107, 2010.
- [26] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning , pages 1597-1607. PMLR, 2020.
- [27] X. Chen, X. Teng, H. Chen, Y. Pan, and P. Geyer. Toward reliable signals decoding for electroencephalogram: A benchmark study to EEGNeX. arXiv preprint arXiv:2207.12369 , 2022.
- [28] S. Chevallier, G. Bao, M. Hammami, F. Marlats, L. Mayaud, D. Annane, F. Lofaso, and E. Azabou. Brainmachine interface for mechanical ventilation using respiratory-related evoked potential. In Artificial Neural Networks and Machine Learning-ICANN 2018: 27th International Conference on Artificial Neural Networks, Rhodes, Greece, October 4-7, 2018, Proceedings, Part III 27 , pages 662-671. Springer, 2018.
- [29] S. Chevallier, E. Kalunga, Q. Barth' elemy, and F. Yger. Rie-

- annian classification for SSVEP based BCI: offline versus online implementations. In Brain-Computer Interfaces Handbook: Technological and Theoretical Advances . Taylor & Francis, 2018.
- [30] H. Cho, M. Ahn, S. Ahn, M. Kwon, and S. C. Jun. EEG datasets for motor imagery brain-computer interface. GigaScience , 6(7):gix034, 2017.
- [31] P. Clisson, R. Bertrand-Lalo, M. Congedo, G. Victor-Thomas, and J. ChatelGoldman. Timeflux: an open-source framework for the acquisition and near real-time processing of signal streams. In BCI 2019-8th International BrainComputer Interface Conference , 2019.
- [32] M. Congedo, M. Goyat, N. Tarrin, G. Ionescu, L. Varnet, B. Rivet, R. Phlypo, N. Jrad, M. Acquadro, and C. Jutten. Brain invaders: a prototype of an open-source P300-based video game working with the OpenViBE platform. In BCI 2011-5th International BrainComputer Interface Conference , pages 280-283, 2011.
- [33] M. Congedo, A. Barachant, and R. Bhatia. Riemannian geometry for EEGbased brain-computer interfaces; a primer and a review. Brain-Computer Interfaces , 4(3):155-174, 2017.
- [34] M.-C. Corsi, S. Chevallier, F. D. V. Fallani, and F. Yger. Functional connectivity ensemble method to enhance BCI performance (FUCONE). IEEE Transactions on Biomedical Engineering , 69 (9):2826-2838, 2022.
- [35] B. Courty, V. Schmidt, Goyal-Kamal, MarionCoutarel, B. Feld, J. Lecourt, LiamConnell, SabAmine, kngoyal, inimaz, M. L' eval, L. Blanche, A. Cruveiller, ouminasara, F. Zhao, A. Joshi,
- A. Bogroff, A. Saboni, H. de Lavoreille,
- N. Laskaris, E. Abati, D. Blank,
- Z. Wang, A. Catovic, alencon,
- M. Stechly, JPW, MinervaBooks, N. Carkaci, and J. Crall. Codecarbon, 2024.
- [36] A. Craik, Y. He, and J. L. ContrerasVidal. Deep learning for electroencephalogram EEG classification tasks: a review. Journal of neural engineering , 16 (3):031001, 2019.
- [37] A. Delorme. EEG is better left alone. Scientific reports , 13(1):2372, 2023.
- [38] J. Faller, C. Vidaurre, T. SolisEscalante, C. Neuper, and R. Scherer. Autocalibration and recurrent adaptation: Towards a plug and play online ERD-BCI. IEEE Transactions on Neural Systems and Rehabilitation Engineering , 20(3):313-319, 2012.
- [39] L. A. Farwell and E. Donchin. Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. Electroencephalography and clinical Neurophysiology , 70(6):510-523, 1988.
- [40] J. Frey. Comparison of an openhardware electroencephalography amplifier with medical grade device in brain-computer interface applications. In PhyCS-International Conference on Physiological Computing Systems . SCITEPRESS, 2016.
- [41] S. Gao, Y. Wang, X. Gao, and B. Hong. Visual and auditory brain-computer interfaces. IEEE Transactions on Biomedical Engineering , 61(5):1436-1447, 2014.
- [42] A. Garcia Badaracco. SciKeras , 2020. URL https://github.com/adriangb/ scikeras .

- [43] A. Gramfort, M. Luessi, E. Larson, D. A. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, and M. S. Hamalainen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience , 7(267):1-13, 2013. doi: 10.3389/fnins.2013.00267.
- [44] M. Grosse-Wentrup, C. Liefhold, K. Gramann, and M. Buss. Beamforming in noninvasive brain-computer interfaces. IEEE Transactions on Biomedical Engineering , 56(4):1209-1219, 2009.
- [45] P. Guetschel and M. Tangermann. Transfer learning between motor imagery datasets using deep learning - validation of framework and comparison of datasets, Nov. 2023.
- [46] C. Guger, S. Daban, E. Sellers, C. Holzner, G. Krausz, R. Carabalona, F. Gramatica, and G. Edlinger. How many people are able to control a P300based brain-computer interface (BCI)? Neuroscience letters , 462(1):94-98, 2009.
- [47] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del R'ıo, M. Wiebe, P. Peterson, P. G'erardMarchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant. Array programming with NumPy. Nature , 585(7825): 357-362, Sept. 2020.
- [48] L. V. Hedges and I. Olkin. Statistical methods for meta-analysis . Academic press, 2014.
- [49] U. Hoffmann, J.-M. Vesin, T. Ebrahimi, and K. Diserens. An efficient P300-based brain-computer interface for disabled
- subjects. Journal of Neuroscience methods , 167(1):115-125, 2008.
- [50] D. Hubner, T. Verhoeven, K. Schmid, K.-R. Muller, M. Tangermann, and P.-J. Kindermans. Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees. PloS one , 12(4):e0175856, 2017.
- [51] D. Huebner, T. Verhoeven, K.-R. Mueller, P.-J. Kindermans, and M. Tangermann. Unsupervised learning for brain-computer interfaces based on event-related potentials: Review and online comparison. IEEE Computational Intelligence Magazine , 13(2): 66-77, 2018.
- [52] T. M. Ingolfsson, M. Hersche, X. Wang, N. Kobayashi, L. Cavigelli, and L. Benini. EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain-machine interfaces. In IEEE International Conference on Systems, Man, and Cybernetics (SMC) , pages 2958-2965. IEEE, 2020.
- [53] M. Jay, V. Ostapenco, L. Lef'evre, D. Trystram, A.-C. Orgerie, and B. Fichel. An experimental comparison of software-based power meters: focus on CPU and GPU. In IEEE/ACM international symposium on cluster, cloud and internet computing , pages 1-13, 2023.
- [54] V. Jayaram and A. Barachant. MOABB: trustworthy algorithm benchmarking for bcis. Journal of neural engineering , 15 (6):066011, 2018.
- [55] V. Jayaram, M. Alamgir, Y. Altun, B. Scholkopf, and M. Grosse-Wentrup. Transfer learning in brain-computer in-

- terfaces. IEEE Computational Intelligence Magazine , 11(1):20-31, 2016.
- [56] E. K. Kalunga, S. Chevallier, Q. Barth' elemy, K. Djouani, E. Monacelli, and Y. Hamam. Online SSVEPbased BCI using Riemannian geometry. Neurocomputing , 191:55-68, 2016.
- [57] E. K. Kalunga, S. Chevallier, and Q. Barth' elemy. Transfer learning for SSVEP-based BCI using riemannian similarities between users. In EUSIPCO , pages 1685-1689, 2018.
- [58] S. Khazem, S. Chevallier, Q. Barth' elemy, K. Haroun, and C. Noˆ us. Minimizing subject-dependent calibration for BCI with Riemannian transfer learning. In International IEEE/EMBS Conference on Neural Engineering (NER) , pages 523-526. IEEE, 2021.
- [59] P.-J. Kindermans, H. Verschore, D. Verstraeten, and B. Schrauwen. A P300 BCI for the masses: Prior information enables instant unsupervised spelling. Advances in neural information processing systems , 25, 2012.
- [60] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [61] Z. J. Koles, M. S. Lazar, and S. Z. Zhou. Spatial patterns underlying population differences in the background EEG. Brain topography , 2:275-284, 1990.
- [62] L. Korczowski, M. Cederhout, A. Andreev, G. Cattan, P. L. C. Rodrigues, V. Gautheret, and M. Congedo. Brain Invaders calibration-less P300-based BCI with modulation of flash duration Dataset (bi2015a) , 2019.
- [63] L. Korczowski, M. Cederhout, A. Andreev, G. Cattan, P. L. C. Rodrigues,
- V. Gautheret, and M. Congedo. Brain Invaders Cooperative versus Competitive: Multi-User P300-based BrainComputer Interface Dataset (bi2015b) , 2019.
- [64] L. Korczowski, E. Ostaschenko, A. Andreev, G. Cattan, P. L. C. Rodrigues, V. Gautheret, and M. Congedo. Brain Invaders calibration-less P300-based BCI using dry EEG electrodes Dataset (bi2014a) , 2019.
- [65] L. Korczowski, E. Ostaschenko, A. Andreev, G. Cattan, P. L. C. Rodrigues, V. Gautheret, and M. Congedo. Brain Invaders Solo versus Collaboration: Multi-User P300-based BrainComputer Interface Dataset (bi2014b) , 2019.
- [66] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM , 60(6):8490, 2017.
- [67] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of neural engineering , 15(5):056013, 2018.
- [68] Y. LeCun, L. Bottou, G. B. Orr, and K.R. Muller. Efficient backprop. In Neural networks: Tricks of the trade , pages 950. Springer, 2002.
- [69] M.-H. Lee, O.-Y. Kwon, Y.-J. Kim, H.-K. Kim, Y.-E. Lee, J. Williamson, S. Fazli, and S.-W. Lee. EEG dataset and openbmi toolbox for three BCI paradigms: An investigation into BCI illiteracy. GigaScience , 8(5):giz002, 2019.

- [70] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof, and G. Pfurtscheller. Braincomputer communication: motivation, aim, and impact of exploring a virtual apartment. IEEE Transactions on Neural Systems and Rehabilitation Engineering , 15(4):473-482, 2007.
- [71] A.-L. Ligozat, J. Lefevre, A. Bugeau, and J. Combaz. Unraveling the hidden environmental impacts of AI solutions for environment life cycle assessment of AI solutions. Sustainability , 14(9):5172, 2022.
- [72] Z. Lin, C. Zhang, W. Wu, and X. Gao. Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. IEEE transactions on biomedical engineering , 53(12):2610-2614, 2006.
- [73] F. Lotte and C. Guan. Regularizing common spatial patterns to improve BCI designs: unified theory and new algorithms. IEEE Transactions on biomedical Engineering , 58(2):355-362, 2010.
- [74] F. Lotte, M. Congedo, A. L' ecuyer, F. Lamarche, and B. Arnaldi. A review of classification algorithms for EEG-based brain-computer interfaces. Journal of neural engineering , 4(2):R1, 2007.
- [75] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger. A review of classification algorithms for EEG-based braincomputer interfaces: a 10 year update. Journal of neural engineering , 15(3): 031005, 2018.
- [76] A. S. Luccioni, S. Viguier, and A.-L. Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language
- model. Journal of Machine Learning Research , 24(253):1-15, 2023.
- [77] S. J. Luck. An introduction to the eventrelated potential technique . The MIT Press, 2014.
- [78] V. Mart'ınez-Cagigal, J. Thielen, E. Santamaria-Vazquez, S. P' erezVelasco, P. Desain, and R. Hornero. Brain-computer interfaces based on code-modulated visual evoked potentials (c-VEP): A literature review. Journal of Neural Engineering , 18(6):061002, 2021.
- [79] M. Moakher. A differential geometric approach to the geometric mean of symmetric positive-definite matrices. SIAM journal on matrix analysis and applications , 26(3):735-747, 2005.
- [80] T. Moreau, M. Massias, A. Gramfort, P. Ablin, P.-A. Bannier, B. Charlier, M. Dagr' eou, T. D. la Tour, G. Durif, C. F. Dantas, Q. Klopfenstein, J. Larsson, E. Lai, T. Lefort, B. Mal' ezieux, B. Moufad, B. T. Nguyen, A. Rakotomamonjy, Z. Ramzi, J. Salmon, and S. Vaiter. Benchopt: Reproducible, efficient and collaborative optimization benchmarks. In Advances in Neural Information Processing Systems (NeurIPS) , volume 36, New-Orleans, LA, USA, Nov. 2022. Curran Associates, Inc.
- [81] J. Muller-Gerking, G. Pfurtscheller, and H. Flyvbjerg. Designing optimal spatial filters for single-trial EEG classification in a movement task. Clinical neurophysiology , 110(5):787-798, 1999.
- [82] S. Nagel and M. Spuler. World's fastest brain-computer interface: combining EEG2Code with deep learning. PloS one , 14(9):e0221909, 2019.

- [83] M. Nakanishi, Y. Wang, Y.-T. Wang, Y. Mitsukura, and T.-P. Jung. Enhancing unsupervised canonical correlation analysis-based frequency detection of SSVEPs by incorporating background EEG. In 2014 36th Annual International Conference of the IEEE Engineering in Medicine and Biology Society , pages 3053-3056. IEEE, 2014.
- [84] M. Nakanishi, Y. Wang, Y.-T. Wang, and T.-P. Jung. A comparison study of canonical correlation analysis based methods for detecting steady-state visual evoked potentials. PloS one , 10(10): e0140703, 2015.
- [85] M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung. Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis. IEEE Transactions on Biomedical Engineering , 65(1):104-112, 2017.
- [86] C. S. Nam, A. Nijholt, and F. Lotte. Brain-computer interfaces handbook: technological and theoretical advances . CRC Press, 2018.
- [87] V. P. Oikonomou, G. Liaros, K. Georgiadis, E. Chatzilari, K. Adam, S. Nikolopoulos, and I. Kompatsiaris. Comparative evaluation of state-of-theart algorithms for SSVEP-based BCIs. arXiv preprint arXiv:1602.00904 , 2016.
- [88] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in
- Neural Information Processing Systems 32 , pages 8024-8035, 2019.
- [89] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research , 12:2825-2830, 2011.
- [90] J. Peirce, R. Hirst, and M. MacAskill. Building experiments in PsychoPy . Sage, 2022.
- [91] C. R. Pernet, S. Appelhoff, K. J. Gorgolewski, G. Flandin, C. Phillips, A. Delorme, and R. Oostenveld. EEGBIDS, an extension to the brain imaging data structure for electroencephalography. Scientific data , 6(1):103, 2019.
- [92] H. Ramoser, J. Muller-Gerking, and G. Pfurtscheller. Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE transactions on rehabilitation engineering , 8(4):441-446, 2000.
- [93] Y. Renard, F. Lotte, G. Gibert, M. Congedo, E. Maby, V. Delannoy, O. Bertrand, and A. L' ecuyer. Openvibe: An open-source software platform to design, test, and use brain-computer interfaces in real and virtual environments. Presence , 19(1):35-53, 2010.
- [94] A. Riccio, L. Simione, F. Schettini, A. Pizzimenti, M. Inghilleri, M. O. Belardinelli, D. Mattia, and F. Cincotti. Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis. Frontiers in human neuroscience , 7:732, 2013.
- [95] B. Rivet, A. Souloumiac, V. Attina, and G. Gibert. xDAWN algorithm to en-

- hance evoked potentials: application to brain-computer interface. IEEE Transactions on Biomedical Engineering , 56 (8):2035-2043, 2009.
- [96] S. RMJ. The american soldier, vol. 1: Adjustment during army life, 1949.
- [97] C. Rommel, J. Paillard, T. Moreau, and A. Gramfort. Data augmentation for learning predictive models on EEG: a systematic comparison. Journal of Neural Engineering , 19(6):066020, 2022.
- [98] R. N. Roy. Neuroergonomics and physiological computing contributions to human-machine interaction . PhD thesis, Universit' e Paul Sabatier, 2022.
- [99] Y. Roy, H. Banville, I. Albuquerque, A. Gramfort, T. H. Falk, and J. Faubert. Deep learning-based electroencephalography analysis: a systematic review. Journal of neural engineering , 16(5): 051001, 2019.
- [100] A. Salami, J. Andreu-Perez, and H. Gillmeister. EEG-ITNet: An explainable inception temporal convolutional network for motor imagery classification. IEEE Access , 10:36672-36685, 2022.
- [101] G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw. BCI2000: a general-purpose braincomputer interface (BCI) system. IEEE Transactions on biomedical engineering , 51(6):1034-1043, 2004.
- [102] R. Scherer, J. Faller, E. V. Friedrich, E. Opisso, U. Costa, A. Kubler, and G. R. Muller-Putz. Individually adapted imagery improves brain-computer interface performance in end-users with disability. PloS one , 10(5):e0123727, 2015.
- [103] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann,
- F. Hutter, W. Burgard, and T. Ball. Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping , 38(11): 5391-5420, 2017.
- [104] S. Schneider, A. Baevski, R. Collobert, and M. Auli. wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862 , 2019.
- [105] J. Shin, A. von Luhmann, B. Blankertz, D.-W. Kim, J. Jeong, H.-J. Hwang, and K.-R. Muller. Open access dataset for EEG+ NIRS single-trial classification. IEEE Transactions on Neural Systems and Rehabilitation Engineering , 25(10): 1735-1745, 2016.
- [106] J. Sosulski and M. Tangermann. Spatial filters for auditory evoked potentials transfer between different experimental conditions. In GBCIC , 2019.
- [107] J. Sosulski and M. Tangermann. UMM: Unsupervised mean-difference maximization. arXiv preprint arXiv:2306.11830 , 2023.
- [108] J. Sosulski, D. Hubner, A. Klein, and M. Tangermann. Online optimization of stimulation speed in an auditory brain-computer interface under time constraints. arXiv preprint arXiv:2109.06011 , 2021.
- [109] D. Steyrl, R. Scherer, J. Faller, and G. R. Muller-Putz. Random forests in non-invasive sensorimotor rhythm braincomputer interfaces: a practical and convenient non-linear classifier. Biomedical Engineering/Biomedizinische Technik , 61(1):77-86, 2016.
- [110] Q. Sun, L. Zheng, W. Pei, X. Gao, and Y. Wang. A 120-target brain-computer interface based on code-modulated vi-

- sual evoked potentials. Journal of Neuroscience Methods , 375:109597, 2022.
- [111] M. Tangermann, K.-R. Muller, A. Aertsen, N. Birbaumer, C. Braun, C. Brunner, R. Leeb, C. Mehring, K. J. Miller, G. Mueller-Putz, et al. Review of the BCI competition IV. Frontiers in neuroscience , page 55, 2012.
- [112] J. Thielen, P. van den Broek, J. Farquhar, and P. Desain. Broad-band visually evoked potentials: re (con) volution in brain-computer interfacing. PloS one , 10(7):e0133797, 2015.
- [113] J. Thielen, P. Marsman, J. Farquhar, and P. Desain. From full calibration to zero training for a code-modulated visual evoked potentials for brain-computer interface. Journal of Neural Engineering , 18(5):056007, 2021.
- [114] M. Tietz, T. J. Fan, D. Nouri, B. Bossan, and skorch Developers. skorch: A scikit-learn compatible neural network library that wraps PyTorch , July 2017. URL https://skorch. readthedocs.io/en/stable/ .
- [115] E. Vaineau, A. Barachant, A. Andreev, P. C. Rodrigues, G. Cattan, and M. Congedo. Brain invaders adaptive versus non-adaptive P300 braincomputer interface dataset. arXiv preprint arXiv:1904.09111 , 2019.
- [116] G. Van Veen, A. Barachant, A. Andreev, G. Cattan, P. C. Rodrigues, and M. Congedo. Building brain invaders: EEG data of an experimental validation. arXiv preprint arXiv:1905.05182 , 2019.
- [117] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, GLYPH<suppress>L. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [118] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, ˙ I. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods , 17:261-272, 2020.
- [119] Z. Wan, R. Yang, M. Huang, N. Zeng, and X. Liu. A review on transfer learning in EEG signal analysis. Neurocomputing , 421:1-14, 01 2021.
- [120] Y. Wang, X. Chen, X. Gao, and S. Gao. A benchmark dataset for SSVEP-based brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering , 25(10):1746-1752, 2016.
- [121] X. Wei, A. A. Faisal, M. GrosseWentrup, A. Gramfort, S. Chevallier, V. Jayaram, C. Jeunet, S. Bakas, S. Ludwig, K. Barmpas, et al. 2021 BEETL competition: Advancing transfer learning for subject independence & heterogenous EEG data sets. In NeurIPS 2021 Competitions and Demonstrations Track , pages 205-219. PMLR, 2022.
- [122] F. Wilcoxon. Individual comparisons by ranking methods. In Breakthroughs in Statistics: Methodology and Distribution , pages 196-202. Springer, 1992.
- [123] Z. Wu and D. Yao. The influence of cognitive tasks on different frequencies

steady-state visual evoked potentials. Brain topography , 20:97-104, 2007.

[124] F. Yger, M. Berar, and F. Lotte. Riemannian approaches in brain-computer interfaces: a review. IEEE Transactions on Neural Systems and Rehabilitation Engineering , 25(10):1753-1762, 2016.

[125] W. Yi, S. Qiu, K. Wang, H. Qi, L. Zhang, P. Zhou, F. He, and D. Ming. Evaluation of EEG oscillatory patterns and cognitive process during simple and compound limb motor imagery. PloS one , 9(12):e114853, 2014.

[126] Y. Zhang, G. Zhou, J. Jin, X. Wang, and A. Cichocki. Frequency recognition in SSVEP-based BCI using multiset canonical correlation analysis. International journal of neural systems , 24(04): 1450013, 2014.

[127] Z. Zhang, S. hua Zhong, and Y. Liu. TorchEEG, 2024. URL https:// torcheeg.readthedocs.io .

[128] B. Zhou, X. Wu, Z. Lv, L. Zhang, and X. Guo. A fully automated trial selection method for optimization of motor imagery based brain-computer interface. PloS one , 11(9):e0162657, 2016.

## Appendix A. Detailed pipelines evaluation

This section provides details regarding the automatic parametrization and the evaluation conducted in the benchmark.

## Appendix A.1. Specialized grid search

The parametrization of evaluated pipelines should be generic to ensure a fair evaluation and automatic to avoid information leak-

age. A dictionary structure containing the parameter to search for each element of the machine learning pipeline is implemented to ensure this process when an evaluation is launched. As shown below, the param grid structure, with names matching the pipeline structure, could be passed to the function evaluation.process() .

```
pipelines = {} pipelines["GridSearchEN"] = Pipeline( steps =[ ("Covariances", Covariances("cov")), ("Tangent\_Space", TangentSpace(metric="riemann")), ( "LogistReg", LogisticRegression( penalty="elasticnet", l1\_ratio =0.70 , intercept\_scaling =1000.0 , solver="saga", max\_iter =1000 , ), ), ] ) param\_grid = {} param\_grid["GridSearchEN"] = { "LogistReg\_\_l1\_ratio": [0.15, 0.30, 0.45, 0.60, 0.75] , } evaluation = WithinSessionEvaluation( paradigm=paradigm , datasets=dataset , overwrite=True , random\_state =42, hdf5\_path=path , n\_jobs=-1, ) result = evaluation.process(pipelines , param\_grid)
```

Appendix A.2. Other evaluation types

The proposed benchmark focus on withinsession evaluation, but it is possible to conduct

Table A1: Parameter used in Grid Search. For the ACM+TS+SVM pipeline, we reduce the hyperparameter search due to computational constraint (both order and lag to [1 - 5] for datasets with more than 60 electrodes - Cho2017, Lee2019-MI, PhysionetMI and Weibo2014. While we select parameters to [1 - 3] for datasets with more than 100 electrodes - Schirrmeister2017 and GrosseWentrup2009)

| Pipeline                                       | Parameter                                                    | Value                                             |
|------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------|
| CSP + SVMGrid                                  | csp nfilter svc C svc kernel                                 | [2 - 8] [0.5, 1, 1.5] ["rbf", "linear"]           |
| EnGrid                                         | logisticregression l1 ratio                                  | [0.20, 0.30, 0.45, 0.65, 0.75]                    |
| LogVarGrid                                     | svc C                                                        | [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]         |
| Tangent Space SVM (SVM) Grid                   | svc C svc kernel                                             | [0.5, 1, 1.5] ["rbf", "linear"]                   |
| ACM + TANG + Support Vector Machine (SVM) Grid | augmenteddataset order augmenteddataset lag svc C svc kernel | [1 - 10] [1 - 10] [0.5, 1, 1.5] ["rbf", "linear"] |

Figure A1: Cross-session evaluation

<!-- image -->

other evaluation types like cross-session or cross-subject. The structure of the crosssession evaluation is shown in Figure A1 and Figure A2 shows the structure of the crosssubject one.

## Appendix B. Machine learning algorithms

This section describes the machine learning pipelines and the choices of implementation.

Figure A2: Cross-subject evaluation

<!-- image -->

## Appendix B.1. CSP-baseline pipelines

Mathematically, the CSP algorithm seeks to find spatial filters by solving a generalized eigenvalue problem. Let X$\_{1}$ and X$\_{2}$ be the data matrices of the band-pass filtered EEG signals for two conditions, each with dimensions of (time × channel). The CSP algorithm constructs discriminative and common activity matrices, S$\_{d}$ and S$\_{c}$ , respectively. These matrices are defined as follows:

S$\_{d}$ = Σ$\_{1}$ - Σ$\_{2}$ and S$\_{c}$ = Σ$\_{1}$ + Σ$\_{2}$ , (B.1)

where Σ$\_{1}$ = X T $\_{1}$X$\_{1}$ and Σ$\_{2}$ = X T $\_{2}$X$\_{2}$ are the estimates of the condition covariance matrices. Then, the objective of CSP is to find spatial filters v$\_{j}$ ∈$\_{R}$ C that maximize the EEG signal bandpower variance between examples from different conditions while simultaneously minimizing its variance between examples from the same condition. This translates into:

argmax$\_{V}$ V $^{T}$S$\_{d}$ V V $^{T}$S$\_{c}$ V (B.2)

which is optimized by solving the following generalised eigenvalue problem [92]:

S$\_{d}$v = λS$\_{c}$v, (B.3)

and selecting the filters v that yield the largest eigenvalues λ .

For classification purposes, CSP utilizes log-variance features extracted from the filtered signals projected onto the CSP filters. Typically, a small number of patterns (2 to 6) are selected based on the corresponding eigenvalues. The patterns, denoted as a$\_{j}$ , provide insights into the specific information captured by the corresponding filters v$\_{j}$ . Each filter v$\_{j}$ extracts the activity spanned by pattern, a$\_{j}$ while canceling out other activities spanned by different patterns. This allows for discrimination between different mental states based on the log-variance features. Linear classifiers, such as linear discriminant analysis, are commonly used due to the approximately Gaussian distribution of the log-variance features. In$\_{MOABB}$, we implemented two different decision head, the SVM and Linear Discriminant Analysis (LDA), with a$\_{GridSearch}$ in some parameters [11].

## Appendix B.2. CCA-based pipelines

Based on the work of Hotelling, the CCA aims at finding a canonical space where

the correlation of two sets of variables is maximized. Considering the total covariance matrix C of two sets of variables x and y , the within-set covariances matrices are respectively denoted C$\_{xx}$ and C$\_{yy}$ and the between-sets covariance matrices are C$\_{xy}$ = C ⊤ $\_{yx}$. The canonical correlation is defined as:

ρ = max$\_{w}$$\_{x}$$\_{,w}$$\_{y}$ w ⊤ $\_{x}$C$\_{xy}$ w$\_{y}$ $^{√}$w ⊤ $\_{x}$C$\_{xx}$ w$\_{x}$w ⊤ $\_{y}$C$\_{yy}$ w$\_{y}$ (B.4)

with w$\_{x}$ and w$\_{y}$ the projection vectors that maximizes canonical correlation. The solution is provided by a generalized eigenproblem formulated as C$\_{xy}$C - 1 y yC$\_{yx}$w$\_{x}$ = λ $^{2}$C$\_{xx}$ w$\_{x}$ . As C$\_{xx}$ and C$\_{yy}$ are symmetric positive definite matrices, it is possible to rewrite the problem as symmetric standard eigenproblem Ax = λx that could be solved with any linear algebra library.

In SSVEP, the CCA is applied on a set of x EEG trials and a set y of reference sinusoid signals as described in [72]. The frequency of the sine and cosine reference signals should match the stimulus frequency and its harmonics. The obtained w$\_{x}$ could be seen as filter acting on EEG signal to enhance the signal components synchronized with the visual stimulus.

Indeed, the choice reference signals is crucial to obtain robust results and could be complex to parameterize. The multiset canonical correlation analysis (MsetCCA) generalizes the CCA for multiple references[126], with the objective to learn the optimal reference signals without constraint on the sine/cosine shape of the reference set.

The goal of filters learned from CCA is to enhance the EEG activity generated by cortical stimulus response. The cortical signal component unrelated to the task could be filtered out as well, as introduced in taskrelated component analysis (TRCA) [85] for

SSVEP. In this case, an optimization process is designed to find the w$\_{x}$ that maximized the inter-trial covariance of EEG signals.

## Appendix B.3. Riemannian pipelines

As described in subsection 4.2, the Riemannian pipelines could be directly defined on the manifold. This is the case of the Minimum Distance to Mean (MDM) classifier, that computes the average of each class with Eq. (2) and use the distance defined in Eq. (1) between an unseen sample and each class center to make a prediction.

To mitigate the effect of increased dimensions, i.e., for EEG recorded with a high number of electrodes, a geodesic filtering could be applied before MDM classification, namely the FgMDM. This filtering step implements a linear discriminant analysis in the tangent space to project all trials on a single hyperplane and the projection back to the manifold.

The second option is to project the samples in the tangent space, vectorize samples that are symmetric matrices and train a classifier on those vectorized data. Algorithms such as logistic regression (LR), logistic regression under ℓ$\_{1}$ and ℓ$\_{2}$ norm penalties (ElasticNet, EL) or support vector machine (SVM) have been described in the literature.

When considering MI, Riemannian classifiers operate directly on covariance matrices estimated from the EEG signals. The transient information of ERP require to first estimate a average ERP or, better, an average ERP filtered XDAWN. For SSVEP, the relevant information is spectral and the covariances need to be estimated on the bandpass filtered signal for each stimulation frequency. All those preprocessing are described in great detail in [124, 29].

## Appendix C. Experimental results

This section provides a global overview of the pipeline scores for the different paradigms with raincloud plots. The pipelines are grouped by categories: Deep learning, Riemannian, and Raw. The small points correspond to the scores obtained on the individual sessions, with an exception for the last row, i.e. Average , where they correspond to the average score over one dataset. The curves above indicate a density estimation of these individual scores. The diamond shapes connected by vertical lines indicate the within-dataset average. Finally, the boxes and horizontal black bars indicate the quartiles.

## Appendix D. Detailed results for each pipeline

This section provides the tables indicating the ROC-AUC for binary classification problems and the accuracy for multi-class tasks. The main objective is to provide a reference benchmark that could be easily reproduced to verify the results or used as-is to compare with new pipelines. It is thus possible to save energy and resources, avoiding reproducing already existing validated results with a simple copy-paste. To this end, the tables provide here the average result of pipelines across all subjects for a given dataset using withinsession evaluation. We did not provide subject-by-subject results for space reasons and we know that this benchmark is meant to evolve with new pipelines, datasets, and evaluation methods. We mirror those tables on a website that will allow further additions and available up-to-date results.

Table D1: Summary of performances via average on all the motor imagery datasets, for classification using all the labels. Intra-session validation. Bold numbers represent the best score in each dataset.Table D2: Summary of performances via average on all the P300 datasets, for classification using left vs. right motor imagery task. Intra-session validation. Bold numbers represent the best score in each dataset.

| pipeline        | AlexandreMotorImagery   | BNCI2014-001   | PhysionetMotorImagery   | Schirrmeister2017   | Weibo2014     | Zhou2016      |   Average |
|-----------------|-------------------------|----------------|-------------------------|---------------------|---------------|---------------|-----------|
| ACM+TS+SVM      | 69.37 ± 15.07           | 77.82 ± 12.23  | 55.44 ± 14.87           | 82.50 ± 10.20       | 63.89 ± 11.01 | 85.25 ± 4.06  |     72.38 |
| CSP+LDA         | 61.04 ± 17.22           | 65.99 ± 15.47  | 47.73 ± 14.35           | 72.97 ± 10.42       | 39.45 ± 11.87 | 82.96 ± 5.20  |     61.69 |
| CSP+SVM         | 62.92 ± 16.89           | 66.88 ± 15.22  | 48.52 ± 14.62           | 75.89 ± 10.55       | 44.08 ± 11.95 | 83.08 ± 5.33  |     63.56 |
| DLCSPauto+shLDA | 60.63 ± 17.91           | 66.31 ± 15.36  | 46.85 ± 14.65           | 72.82 ± 10.44       | 38.84 ± 11.97 | 82.06 ± 5.57  |     61.25 |
| DeepConvNet     | 37.71 ± 4.56            | 35.29 ± 8.26   | 27.68 ± 3.91            | 56.78 ± 18.11       | 24.17 ± 9.80  | 55.69 ± 5.61  |     39.55 |
| EEGITNet        | 36.04 ± 3.43            | 35.55 ± 6.35   | 26.15 ± 4.95            | 70.44 ± 14.68       | 25.78 ± 8.00  | 50.68 ± 16.27 |     40.77 |
| EEGNeX          | 37.71 ± 9.64            | 45.62 ± 15.29  | 26.69 ± 5.64            | 67.56 ± 14.15       | 30.22 ± 11.02 | 56.42 ± 11.29 |     44.03 |
| EEGNet-8,2      | 43.96 ± 8.62            | 60.46 ± 20.20  | 29.04 ± 7.03            | 76.99 ± 13.05       | 35.35 ± 14.05 | 83.34 ± 3.58  |     54.86 |
| EEGTCNet        | 34.17 ± 1.86            | 41.65 ± 13.73  | 25.79 ± 3.85            | 71.11 ± 11.96       | 17.95 ± 3.88  | 37.19 ± 2.57  |     37.98 |
| FBCSP+SVM       | 65.00 ± 17.56           | 66.53 ± 12.05  | 45.49 ± 12.54           | 75.94 ± 8.59        | 45.21 ± 10.05 | 81.99 ± 4.65  |     63.36 |
| FgMDM           | 65.63 ± 15.63           | 70.14 ± 15.13  | 55.04 ± 14.17           | 82.97 ± 10.08       | 56.94 ± 9.26  | 83.07 ± 4.96  |     68.97 |
| MDM             | 60.62 ± 13.69           | 61.60 ± 14.20  | 42.96 ± 12.98           | 52.03 ± 10.11       | 33.41 ± 8.67  | 76.05 ± 7.10  |     54.45 |
| ShallowConvNet  | 50.00 ± 12.94           | 72.47 ± 16.50  | 41.87 ± 12.50           | 85.13 ± 9.57        | 48.94 ± 10.36 | 85.02 ± 3.78  |     63.91 |
| TS+EL           | 69.79 ± 13.75           | 72.38 ± 14.85  | 59.93 ± 14.07           | 85.53 ± 9.40        | 63.84 ± 8.77  | 84.54 ± 4.93  |     72.67 |
| TS+LR           | 69.17 ± 14.79           | 71.97 ± 15.46  | 58.55 ± 14.06           | 84.60 ± 9.28        | 62.76 ± 8.39  | 84.88 ± 4.63  |     71.99 |
| TS+SVM          | 67.92 ± 12.74           | 70.76 ± 15.08  | 58.46 ± 15.15           | 84.41 ± 9.56        | 61.47 ± 9.62  | 83.66 ± 4.55  |     71.11 |
| Average         | 55.73                   | 61.34          | 43.51                   | 74.85               | 43.27         | 74.74         |     58.91 |

| pipeline        | BNCI2014-001   | BNCI2014-004   | Cho2017       | GrosseWentrup2009   | Lee2019-MI    | PhysionetMotorImagery   | Schirrmeister2017   | Shin2017A     | Weibo2014     | Zhou2016      |   Average |
|-----------------|----------------|----------------|---------------|---------------------|---------------|-------------------------|---------------------|---------------|---------------|---------------|-----------|
| ACM+TS+SVM      | 91.71 ± 10.30  | 82.67 ± 15.33  | 73.56 ± 14.54 | 86.60 ± 15.12       | 83.05 ± 13.97 | 63.55 ± 21.24           | 85.82 ± 13.98       | 68.97 ± 23.45 | 84.78 ± 13.33 | 95.03 ± 4.76  |     81.57 |
| CSP+LDA         | 82.34 ± 17.26  | 80.10 ± 14.93  | 71.38 ± 14.54 | 76.44 ± 20.95       | 76.88 ± 17.41 | 65.75 ± 17.37           | 77.23 ± 18.43       | 72.30 ± 21.79 | 80.72 ± 15.29 | 93.15 ± 6.88  |     77.63 |
| CSP+SVM         | 83.07 ± 16.53  | 79.27 ± 15.68  | 71.92 ± 14.25 | 77.81 ± 21.27       | 77.27 ± 16.73 | 65.71 ± 17.90           | 79.24 ± 20.07       | 70.11 ± 22.19 | 79.84 ± 15.86 | 92.96 ± 7.86  |     77.72 |
| DLCSPauto+shLDA | 82.75 ± 16.69  | 79.87 ± 15.11  | 71.16 ± 14.53 | 76.40 ± 20.83       | 76.69 ± 17.23 | 65.07 ± 17.68           | 77.02 ± 18.48       | 70.34 ± 23.30 | 80.16 ± 15.23 | 92.56 ± 7.21  |     77.2  |
| DeepConvNet     | 82.07 ± 15.52  | 72.36 ± 18.53  | 71.67 ± 12.91 | 82.38 ± 15.39       | 70.65 ± 15.76 | 59.57 ± 16.77           | 81.23 ± 17.39       | 56.03 ± 19.18 | 73.64 ± 15.78 | 94.42 ± 6.21  |     74.4  |
| EEGITNet        | 75.27 ± 16.37  | 65.10 ± 15.32  | 57.20 ± 12.21 | 72.19 ± 14.71       | 59.17 ± 11.72 | 52.71 ± 11.11           | 74.66 ± 20.52       | 52.18 ± 16.78 | 59.35 ± 14.06 | 69.41 ± 14.66 |     63.72 |
| EEGNeX          | 66.28 ± 13.22  | 66.53 ± 17.10  | 53.28 ± 10.60 | 57.00 ± 7.52        | 55.12 ± 10.05 | 51.20 ± 10.63           | 68.58 ± 19.37       | 49.02 ± 17.58 | 57.97 ± 15.65 | 61.56 ± 14.60 |     58.65 |
| EEGNet-8,2      | 77.15 ± 19.33  | 69.50 ± 19.50  | 66.79 ± 16.34 | 83.02 ± 18.08       | 65.67 ± 16.43 | 59.55 ± 15.95           | 80.20 ± 18.13       | 57.99 ± 17.28 | 66.46 ± 21.78 | 94.84 ± 2.83  |     72.12 |
| EEGTCNet        | 67.46 ± 20.81  | 69.70 ± 19.55  | 58.34 ± 12.63 | 68.45 ± 16.27       | 55.68 ± 12.75 | 55.90 ± 12.74           | 75.62 ± 22.33       | 51.26 ± 16.77 | 63.16 ± 18.32 | 82.24 ± 9.40  |     64.78 |
| FBCSP+SVM       | 84.44 ± 16.00  | 80.39 ± 16.05  | 67.91 ± 15.63 | 79.65 ± 18.63       | 75.07 ± 16.97 | 58.45 ± 13.93           | 81.44 ± 17.89       | 65.63 ± 21.64 | 76.81 ± 18.88 | 92.64 ± 5.01  |     76.24 |
| FgMDM           | 86.53 ± 12.14  | 79.28 ± 15.25  | 72.90 ± 12.70 | 87.02 ± 13.20       | 81.34 ± 13.93 | 68.46 ± 19.06           | 86.71 ± 13.79       | 70.86 ± 23.36 | 78.41 ± 14.85 | 92.54 ± 6.67  |     80.41 |
| LogVar+LDA      | 77.96 ± 15.09  | 78.51 ± 15.25  | 64.49 ± 10.08 | 78.71 ± 11.69       | 66.21 ± 12.06 | 61.94 ± 14.41           | 78.44 ± 13.76       | 61.78 ± 22.77 | 74.13 ± 10.40 | 88.39 ± 8.57  |     73.06 |
| LogVar+SVM      | 75.86 ± 16.45  | 78.30 ± 15.18  | 65.46 ± 11.71 | 81.73 ± 12.40       | 73.83 ± 13.85 | 62.35 ± 16.87           | 79.42 ± 13.66       | 61.38 ± 22.68 | 74.85 ± 11.33 | 88.47 ± 8.50  |     74.17 |
| MDM             | 81.69 ± 14.94  | 77.66 ± 15.78  | 63.39 ± 13.69 | 64.29 ± 8.04        | 70.23 ± 13.87 | 54.76 ± 16.79           | 61.53 ± 16.41       | 62.99 ± 21.25 | 58.80 ± 16.13 | 90.70 ± 7.11  |     68.6  |
| ShallowConvNet  | 86.17 ± 13.74  | 72.36 ± 18.05  | 73.84 ± 14.95 | 86.53 ± 13.00       | 75.83 ± 15.04 | 65.19 ± 15.80           | 84.82 ± 15.29       | 60.80 ± 19.27 | 79.10 ± 12.63 | 95.65 ± 5.55  |     78.03 |
| TRCSP+LDA       | 79.84 ± 16.28  | 79.78 ± 15.22  | 71.85 ± 13.84 | 78.29 ± 16.66       | 76.26 ± 15.41 | 67.24 ± 17.23           | 79.14 ± 15.91       | 67.30 ± 23.19 | 79.33 ± 14.43 | 93.53 ± 6.38  |     77.25 |
| TS+EL           | 86.44 ± 13.20  | 79.75 ± 15.44  | 76.23 ± 14.21 | 89.25 ± 12.00       | 84.74 ± 13.19 | 67.91 ± 20.03           | 88.65 ± 12.98       | 68.68 ± 23.64 | 85.29 ± 12.10 | 94.35 ± 6.04  |     82.13 |
| TS+LR           | 87.41 ± 12.58  | 80.09 ± 15.01  | 75.01 ± 13.71 | 87.60 ± 13.20       | 83.09 ± 13.46 | 67.28 ± 19.19           | 87.22 ± 13.83       | 69.31 ± 23.06 | 83.62 ± 13.88 | 94.16 ± 6.33  |     81.48 |
| TS+SVM          | 86.48 ± 13.58  | 79.41 ± 15.26  | 74.62 ± 14.19 | 88.08 ± 13.58       | 83.57 ± 14.08 | 68.18 ± 19.92           | 87.64 ± 13.48       | 68.45 ± 24.25 | 83.72 ± 14.28 | 93.37 ± 6.30  |     81.35 |
| Average         | 81.1           | 76.35          | 68.47         | 79.02               | 73.18         | 62.15                   | 79.72               | 63.44         | 74.74         | 89.47         |     74.76 |

Table D3: Summary of performances via average on all the P300 datasets, for classification using right hand vs. feet tasks motor imagery task. Intra-session validation. Bold numbers represent the best score in each dataset.Table D4: Summary of performances via average on all the P300 datasets, for classification using all the labels. Intra-session validation. Bold numbers represent the best score in each dataset.

| pipeline        | AlexandreMotorImagery   | BNCI2014-001   | BNCI2014-002   | BNCI2015-001   | BNCI2015-004   | PhysionetMotorImagery   | Schirrmeister2017   | Weibo2014     | Zhou2016      |   Average |
|-----------------|-------------------------|----------------|----------------|----------------|----------------|-------------------------|---------------------|---------------|---------------|-----------|
| ACM+TS+SVM      | 86.56 ± 12.26           | 97.32 ± 3.35   | 88.60 ± 10.71  | 93.01 ± 8.09   | 62.60 ± 14.62  | 93.33 ± 8.46            | 98.67 ± 3.06        | 93.25 ± 4.12  | 97.18 ± 3.00  |     90.06 |
| CSP+LDA         | 77.19 ± 17.58           | 91.52 ± 10.39  | 80.98 ± 14.79  | 88.52 ± 10.75  | 54.02 ± 11.33  | 86.41 ± 13.96           | 97.02 ± 5.17        | 88.59 ± 6.36  | 95.20 ± 3.17  |     84.38 |
| CSP+SVM         | 78.59 ± 20.14           | 91.04 ± 10.35  | 81.21 ± 15.30  | 89.19 ± 10.08  | 52.08 ± 11.05  | 88.04 ± 12.57           | 97.50 ± 4.90        | 88.64 ± 5.90  | 94.95 ± 3.53  |     84.58 |
| DLCSPauto+shLDA | 77.03 ± 18.93           | 91.54 ± 10.37  | 80.45 ± 15.52  | 88.87 ± 10.42  | 53.02 ± 10.75  | 86.81 ± 13.34           | 96.95 ± 5.22        | 88.48 ± 6.53  | 94.43 ± 3.41  |     84.18 |
| DeepConvNet     | 61.88 ± 19.05           | 88.27 ± 12.19  | 87.56 ± 11.25  | 88.12 ± 13.19  | 57.08 ± 12.29  | 71.49 ± 15.88           | 95.90 ± 7.14        | 79.29 ± 12.63 | 95.92 ± 3.66  |     80.61 |
| EEGITNet        | 47.50 ± 9.46            | 75.98 ± 13.09  | 70.90 ± 17.50  | 71.95 ± 16.76  | 51.41 ± 6.40   | 54.69 ± 11.97           | 96.04 ± 8.62        | 62.54 ± 12.32 | 80.40 ± 17.12 |     67.93 |
| EEGNeX          | 52.34 ± 14.81           | 64.36 ± 13.49  | 69.95 ± 20.12  | 72.34 ± 19.83  | 53.02 ± 9.69   | 51.77 ± 12.06           | 89.49 ± 16.91       | 60.18 ± 11.70 | 64.80 ± 16.64 |     64.25 |
| EEGNet-8,2      | 64.22 ± 16.01           | 88.55 ± 14.92  | 83.93 ± 16.31  | 90.43 ± 11.75  | 54.20 ± 8.20   | 73.78 ± 15.59           | 96.50 ± 8.07        | 78.15 ± 14.46 | 94.58 ± 3.21  |     80.48 |
| EEGTCNet        | 61.09 ± 22.06           | 75.21 ± 18.53  | 73.92 ± 19.02  | 77.21 ± 18.55  | 51.22 ± 5.84   | 57.03 ± 13.25           | 97.15 ± 7.70        | 62.37 ± 12.42 | 85.46 ± 16.42 |     71.19 |
| FBCSP+SVM       | 80.78 ± 18.86           | 93.55 ± 6.29   | 80.39 ± 16.83  | 91.57 ± 7.66   | 52.51 ± 9.82   | 83.97 ± 12.43           | 97.40 ± 4.18        | 88.27 ± 7.91  | 94.63 ± 3.94  |     84.78 |
| FgMDM           | 79.84 ± 17.80           | 93.52 ± 8.18   | 84.77 ± 11.26  | 90.18 ± 9.77   | 58.31 ± 12.63  | 89.67 ± 10.65           | 98.48 ± 3.45        | 88.56 ± 4.63  | 96.04 ± 2.67  |     86.6  |
| MDM             | 74.22 ± 21.19           | 89.13 ± 10.38  | 77.48 ± 14.11  | 86.20 ± 12.99  | 48.45 ± 9.62   | 81.78 ± 11.64           | 84.67 ± 13.13       | 65.18 ± 9.75  | 92.21 ± 4.31  |     77.7  |
| ShallowConvNet  | 64.22 ± 18.33           | 93.00 ± 8.05   | 87.60 ± 12.05  | 91.41 ± 10.88  | 57.23 ± 12.36  | 74.75 ± 14.98           | 98.06 ± 4.35        | 88.70 ± 5.60  | 97.06 ± 1.86  |     83.56 |
| TS+EL           | 81.41 ± 21.36           | 94.45 ± 6.74   | 85.98 ± 11.38  | 91.19 ± 8.49   | 58.70 ± 13.37  | 94.09 ± 7.17            | 98.56 ± 3.01        | 92.32 ± 3.98  | 96.59 ± 2.82  |     88.14 |
| TS+LR           | 83.75 ± 17.47           | 94.45 ± 7.06   | 85.86 ± 11.01  | 91.09 ± 8.71   | 61.01 ± 14.22  | 93.15 ± 7.40            | 98.60 ± 3.08        | 91.53 ± 4.53  | 96.76 ± 2.58  |     88.47 |
| TS+SVM          | 82.66 ± 18.16           | 94.01 ± 7.60   | 86.19 ± 11.50  | 90.81 ± 8.95   | 62.55 ± 15.30  | 94.27 ± 7.19            | 98.72 ± 2.92        | 91.84 ± 4.25  | 96.11 ± 2.99  |     88.57 |
| Average         | 72.08                   | 88.49          | 81.61          | 87.01          | 55.46          | 79.69                   | 96.23               | 81.74         | 92.02         |     81.59 |

Table D5: Summary of performances via average on all the P300 datasets, for classification using left vs. right motor imagery task. Intra-session validation. Bold numbers represent the best score in each dataset.

| pipeline          |              |               |               |               |              | BNCI2014-008 BNCI2014-009 BNCI2015-003 BrainInvaders2012 BrainInvaders2013a BrainInvaders2014a BrainInvaders2014b BrainInvaders2015a BrainInvaders2015b Cattan2019-VR   |               |               |               |               | EPFLP300      |              |              | Huebner2017 Huebner2018 Lee2019-ERP Sosulski2019   | Average             |
|-------------------|--------------|---------------|---------------|---------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|---------------|---------------|--------------|--------------|----------------------------------------------------|---------------------|
| ERPCov+MDM        | 74.30 ± 9.77 | 81.16 ± 10.13 | 76.79 ± 10.95 | 78.77 ± 10.32 | 80.59 ± 9.36 | 71.62 ± 11.17                                                                                                                                                           | 78.57 ± 12.36 | 80.02 ± 10.07 | 75.04 ± 15.85 | 80.76 ± 10.07 | 71.97 ± 10.88 | 94.47 ± 8.26 | 95.15 ± 3.72 | 74.43 ± 13.26                                      | 68.17 ± 13.59 78.79 |
| ERPCov(svdn4)+MDM | 75.42 ± 9.91 | 84.52 ± 8.83  | 76.93 ± 11.26 | 79.02 ± 10.53 | 82.07 ± 8.46 | 72.11 ± 11.64                                                                                                                                                           | 76.48 ± 12.83 | 77.92 ± 10.33 | 77.09 ± 15.81 | 80.67 ± 9.47  | 71.44 ± 10.20 | 96.21 ± 6.50 | 96.61 ± 1.89 | 82.47 ± 12.56                                      | 70.63 ± 13.79 79.97 |
| XDAWN+LDA         | 82.24 ± 5.26 | 64.03 ± 3.91  | 78.62 ± 7.19  | 64.41 ± 4.14  | 76.74 ± 7.16 | 66.60 ± 7.54                                                                                                                                                            | 83.73 ± 10.62 | 76.02 ± 10.46 | 77.22 ± 13.73 | 67.16 ± 6.11  | 62.98 ± 5.38  | 97.74 ± 2.84 | 97.54 ± 1.58 | 96.45 ± 3.93                                       | 67.49 ± 7.44 77.27  |
| XDAWNCov+MDM      | 77.62 ± 9.81 | 92.04 ± 5.97  | 83.08 ± 7.55  | 88.22 ± 5.90  | 90.97 ± 5.52 | 80.88 ± 11.01                                                                                                                                                           | 91.58 ± 10.02 | 92.57 ± 5.03  | 83.48 ± 12.05 | 88.53 ± 7.34  | 83.20 ± 9.05  | 98.07 ± 2.09 | 97.78 ± 1.04 | 97.70 ± 2.68                                       | 86.07 ± 7.15 88.79  |
| XDAWNCov+TS+SVM   | 85.61 ± 4.43 | 93.43 ± 5.11  | 82.95 ± 8.57  | 90.99 ± 4.79  | 92.71 ± 4.92 | 85.77 ± 9.75                                                                                                                                                            | 91.88 ± 9.94  | 93.05 ± 4.98  | 84.56 ± 12.09 | 90.68 ± 6.29  | 84.29 ± 8.53  | 98.69 ± 1.78 | 98.47 ± 0.97 | 98.41 ± 2.03                                       | 87.28 ± 6.92 90.58  |
| Average           | 79.04        | 83.03         | 79.67         | 80.28         | 84.61        | 75.4                                                                                                                                                                    | 84.45         | 83.91         | 79.48         | 81.56         | 74.78         | 97.04        | 97.11        | 89.89                                              | 75.93 83.08         |

| pipeline   | Kalunga2016   | Lee2019-SSVEP   | MAMEM1        | MAMEM2        | MAMEM3        | Nakanishi2015   | Wang2016      |   Average |
|------------|---------------|-----------------|---------------|---------------|---------------|-----------------|---------------|-----------|
| CCA        | 25.40 ± 2.51  | 23.86 ± 3.72    | 19.17 ± 5.01  | 23.60 ± 4.10  | 13.80 ± 7.47  | 8.15 ± 0.74     | 2.48 ± 1.01   |     16.64 |
| MsetCCA    | 22.67 ± 4.23  | 25.10 ± 3.81    | 20.50 ± 2.37  | 22.08 ± 1.76  | 27.60 ± 3.01  | 7.10 ± 1.50     | 4.00 ± 1.10   |     18.43 |
| MDM        | 70.89 ± 13.44 | 75.38 ± 18.38   | 27.31 ± 11.64 | 23.12 ± 6.29  | 34.40 ± 9.96  | 78.77 ± 19.06   | 54.77 ± 21.95 |     52.09 |
| TS+LR      | 70.86 ± 11.64 | 89.44 ± 13.84   | 53.71 ± 24.25 | 39.36 ± 12.06 | 42.10 ± 14.33 | 87.22 ± 15.96   | 67.52 ± 20.04 |     64.32 |
| TS+SVM     | 68.95 ± 13.73 | 88.58 ± 14.47   | 50.58 ± 23.34 | 34.80 ± 11.76 | 40.20 ± 14.41 | 86.30 ± 15.88   | 59.58 ± 20.57 |     61.28 |
| TRCA       | 24.84 ± 7.24  | 64.01 ± 15.27   | 24.24 ± 6.65  | 24.24 ± 2.93  | 23.70 ± 3.49  | 83.21 ± 10.80   | 2.79 ± 1.03   |     35.29 |
| Average    | 47.27         | 61.06           | 32.58         | 27.87         | 30.3          | 58.46           | 31.86         |     41.34 |

Figure C1: Distributions of ROC-AUC scores on the right hand vs left hand MI task of the pipelines grouped by category.

<!-- image -->

Figure C2: Distributions of ROC-AUC scores on the right hand vs feet MI task.

<!-- image -->

Figure C3: Accuracy scores averaged over all subjects and session of each SSVEP dataset, per pipeline category.

<!-- image -->

Figure C4: Box-plot representing classification accuracy averaged over all the sessions of all the subjects of all the datasets of the SSVEP paradigm and over all pipelines of the corresponding category ( Riemannian, Raw ). Box-plots are overlaid with strip-plots, where each point represents the classification accuracy of one within-session evaluation.

<!-- image -->

Pipeline XDAWNCov + MDM (Riemannian) ERPCov + MDM (Riemannian) XDAWN + LDA (Raw) XDAWNCov + TS + SVM (Riemannian) ERPCov(svd\_n=4) + MDM (Riemannian)

Figure C5: ROC-AUC scores on the ERP classification task of the different pipelines.

<!-- image -->

<!-- image -->

## Deep Riemannian Neural Architectures for Domain Adaptation in Burst cVEP-based Brain Computer Interface

Sébastien Velut, Sylvain Chevallier, Marie-Constance Corsi, Frédéric Dehais

## To cite this version:

Sébastien Velut, Sylvain Chevallier, Marie-Constance Corsi, Frédéric Dehais. Deep Riemannian Neural Architectures for Domain Adaptation in Burst cVEP-based Brain Computer Interface. ESANN 2024, Oct 2024, Bruges (Belgium) and online, France. pp.571-576, ￿10.14428/esann/2024.ES2024-112￿. ￿hal04720928￿

## HAL Id: hal-04720928

## https://hal.science/hal-04720928v1

Submitted on 4 Oct 2024

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

## Deep Riemannian Neural Architectures for Domain Adaptation in Burst cVEP-based Brain Computer Interface

Velut S' ebastien 1 , $^{2}$, Chevallier Sylvain$^{2}$, Corsi Marie-Constance$^{3}$, Dehais Fr' ed' eric 1

1- F' ed' eration ENAC ISAE-SUPAERO ONERA, Universit' e de Toulouse, 10 avenue Edouard Belin, 31400, Toulouse - France

2- A&O - LISN - Universit' e Paris-Saclay

1 rue Ren' e Thom, 91190 Gif-sur-Yvette - France

3- Sorbonne Universit' e, Institut du Cerveau - Paris Brain Institute -ICM, CNRS, Inria, Inserm, AP-HP, Hopital de la Piti' e Salpetriere, F-75013, Paris, France

Abstract . Code modulated Visually Evoked Potentials (cVEP) is an emerging paradigm for Brain-Computer Interfaces (BCIs) that offers reduced calibration times. However, cVEP-based BCIs still encounter challenges related to cross-session/subject variabilities. As Riemannian approaches have demonstrated good robustness to these variabilities, we propose the first study of deep Riemannian neural architectures, namely SPDNets, on cVEP-based BCIs. To evaluate their performance with respect to subject variabilities, we conduct classification tasks in a domain adaptation framework using a burst cVEP open dataset. This study demonstrates that SPDNet yields the best accuracy with single-subject calibration and promising results in domain adaptation.

## 1 Introduction

Code modulated Visually Evoked Potentials (cVEP) have gained popularity in the Brain-Computer Interface (BCI) community [1]. This approach employs pseudo-random visual flickers, providing advantages such as shorter calibration times, as only one code needs to be learned. Alternative decoding methods, like bitwise-decoding [2], have enabled self-paced BCI with flexible decoding period. Despite these advancements, cVEP-based BCIs remain primarily studied in lab settings due to the persistent need for recalibration before each use. This limitation is related to cross-session and cross-subject variabilities common to all BCI paradigms. These sources of variability in BCI are diverse [3], encompassing anatomical differences such as variations in grey matter quantity, human factors like differences in education level and lifestyle habits, or physiological factors like fatigue, concentration levels, and stress levels. Additionally, neurophysiological disparities, such as variations in modulations of spectral power across specific frequencies, also contribute to these variabilities. To address these sources of variability, extensive research has been conducted [4, 3] to propose new approaches. There are two main settings for evaluating transfer learning approaches, depending on the quantity of information available for a target subject. In the most independent setting, refer to as Domain Generalization , no information from the target subject is at hand thus the model is trained on data

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

from n -1 subjects and tested on the n -th (target) subject. If some information could be obtained from the target subject, the so-called Domain Adaptation setting uses a small portion of the n -th subject's data for training [5]. In the context of Deep Learning (DL) algorithms, the model could be initially trained on n -1 subjects, followed by freezing one or more layers and retraining the model on the data from the last subject [6]. To achieve reasonable performances in DG/DA settings, an important preprocessing step is data alignment, which aims to align the data from all subjects to ensure a similar feature space across subjects [5]. While these settings have shown promise in addressing inter-subject variability, there is still room for improvement in terms of accuracy and robustness. In this context, Riemannian geometry has emerged as a powerful tool for enhancing the performance of BCIs. Riemannian techniques have the ability to capture the intrinsic structure of the data on a curved manifold, enabling more effective and efficient classification algorithms [7, 8]. Moreover, recentering the data before classification has been proven to enhance accuracy when using transfer learning techniques [9].

In this study, we propose a novel approach that adapts Riemannian deep learning model, namely SPDNet, for cVEP-based BCIs to address cross-subject variability. We aim to compare our method with state-of-the-art CNN model for cVEP in terms of accuracy and robustness across subjects. For this study, we adapted SPDNet [10] and its batch normalization version [11] (SPDBNNet) for a cVEP dataset [2] with 12 participants. We evaluated these models on two different transfer learning settings: Domain Generalization (DG) and Domain Adaptation (DA). They are compared with Single Subject (SS) training baseline (no transfer learning), where one model per subject is trained with longer calibration phase than DA. In the following sections, we will first provide in Sect. 2 some preliminaries to redefine key definitions of Riemannian manifolds and context related to the cVEP. Next, we will present in Sect. 3 the methodology used in our study, followed in Sect. 4 by the results obtained and the implications of our findings. Finally, we conclude the paper with Sect. 5.

## 2 Preliminaries

## 2.1 Riemannian manifold

Let P$\_{n}$ be the set of the n × n symmetric definite-positive (SPD) matrices : P$\_{n}$ : { P ∈ R n × $^{n}$,P ⊤ = P,u $^{⊤}$Pu = 0 , ∀ u ∈ R $^{n}$} endowed with the affineinvariant Riemannian (AIR) distance. While other distance could be considered, the properties of the AIR and the Frechet mean (center of mass of a SPD matrices) are interesting for processing EEG and their definition can be found in [8],[7]

## 2.2 cVEP and Burst cVEP

Code modulated VEP relies on pseudo-random code of 0s and 1s, often referred to as an m-sequence or Gold codes, and is generated using a linear feedback

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

shift register. Once the initial sequence is produced, the subsequent sequences are then generated by phase-shifting this initial sequence. Indeed, it is crucial for these sequences to be as uncorrelated as possible to ensure a maximal discriminability between the sequences. The burst cVEP is a variation of the cVEP. Burst cVEP employs short bursts of irregular visual flashes at a slower rate [2]. This method evokes a distinct brain response, known as P100 (i.e., a positive deflection that occurs 100 ms after the brief burst of flash) with double the amplitude of classical cVEP m-sequences, as shown in Figure 1. Those clearer responses to onsets facilitate the detection and classification of the code. Additionally, those code could be more comfortable for the user, as it is possible to reduce the flash amplitude, generating thus less visual fatigue in response to those rapidly blinking stimuli.

<!-- image -->

Fig. 1: cVEP relying on m-sequence or burst. Left: The black line illustrates a prototypical m-sequence cVEP, animating the presentation of the flash with alternating plateaus of '1' (flash on) and '0' (flash off). The red line represents the averaged and normalized cerebral response to this alternating on/off visual stimulation. Right: Example of Burst cVEP graphs, consisting of brief flash presentations. The blue line depicts the averaged and normalized cerebral response to this alternating on/off visual stimulation. Adapted from [2]

<!-- image -->

## 3 Material and Method

## 3.1 Datasets

The data used for this study come from the dataset created with the study of Cabrera-Castillos et al. 1 [2]. In this experiment, 12 participants were instructed to concentrate on four targets cued sequentially for 2.2 seconds in random order for each of 15 blocks (60 trials in total). To retrieve the dataset easily and to facilitate the reproducibility of this study, we have implemented this dataset in MOABB [12].

## 3.2 Method

The emergence and advantages of Riemannian methods in recent years have led to significant improvements in the performance of BCIs, particularly in the

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

context of motor imagery tasks [8]. However, there have been relatively few studies exploring the potential of Riemannian classifiers in the domain of cVEPbased BCIs. Motivated by this gap in the literature, we sought to investigate the performance of Riemannian classifiers (3 layers of BiMap-ReEig, followed by a LogEig layer, a flatten and a Linear layer; for the SPDBNNet, the batch normalisation layers are not domain specific) in comparison to a state-of-the-art deep learning algorithm, an optimized CNN which has demonstrated excellent results for Burst cVEP [2].

We evaluated different settings for transfer learning, namely domain generalization and domain adaptation. Let k be the target participant index (evaluated in testing set) and I = [1 ,... 12] the participants' index. For domain generalization, let be Ω DG S = { X$\_{i}$, ∀ i ∈ I \ k } the training set with X$\_{i}$ the data of the i -th participant. The model is trained on Ω$\_{S}$ and tested on the testing set Ω DG T = { X$\_{k}$ } . In Domain Adaptation setting, a subset X DA k ∈ X$\_{k}$ of trials from subject k are available in Ω DA S = Ω DG S ∪ X DA k and, indeed, excluded from testing set Ω DA T = Ω DG T \ X DA k . In this study, X DA k are the first 16 (out of 60) trials of X$\_{k}$ , representing roughly 32 sec of calibration for a subject. The baseline is a single subject setting, using X SS k 32 (out of 60) trials for training, that represents a circa 110s calibration to train a subject/session-specific model. For the training we used the following parameters : lr=0.001,optimiser=Adam or RiemannianAdam,loss=CrossEntropy,epochs=20,batchsize=64.

All channels of the 32-electrode EEG cap are used, EEG data is bandpass filtered between 1-25 Hz, and the signals are re-referenced to the average. Epochs were created by extracting 0.25s windows following each frame and labeling them with the corresponding bit of the code. For the Riemannian algorithms, a spatial XDAWN filter, on class 1, is estimated and applied before computing covariance matrices with Ledoit-Wolf estimator. A recentering step is applied to whiten the covariance matrices. For the CNN, the data were normalized as commonly done with neural nets and a recentering step is applied too. Due to the unbalanced nature of Burst cVEP, the dataset has circa 4.85 times more 0 than 1. To address this imbalance, we randomly removed 0-labeled epochs to match 1-labeled epochs in training set. We did not balance the testing set, to reflect situation occurring in a real world online BCI application. To avoid bias, we repeated 10 times the process of each pipelines and then average the different scores.

## 4 Results and discussion

The statistical differences shown in this section are obtained via the Stouffer's method that combines p-values resulting from the Wilcoxon signed-rank test. Then a Bonferroni correction was performed.

Figure 2 illustrates the performance comparison of different DL models in different settings. The DA setting exhibits significantly higher accuracy than DG setting( p ⩽ 0 . 05) but, no difference is found between DG and SS setting or between DA and SS setting. This result is something in contrary to what we expected compared to other studies. There are much more training samples

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

Fig. 2: Average accuracy scores for all subjects (10 repetitions). The box range is between the first and third quartile. The black line in the box is the median and the dark color square point is the mean

<!-- image -->

in DG and DA settings than in SS setting. It can explained that SS setting is not significantly higher. It indicates that riemannian algorithms and the CNN performs equally well between the settings.

When comparing the same setting (DG/DA/SS) across different models, all three models show comparable performance without significant differences, with one exception: in the SS setting, the CNN and the SPDBNNet achieves better accuracy than the SPDNet. Notably, we successfully matched the accuracy of an optimized CNN using Riemannian DL models, achieving very good performance (circa 93% accuracy). It should be noted the hyperparameters of the CNN are well optimized as reported in [2], while we did not conduct any hyperparameter search for Riemannian DL models, it is thus likely that the SPDNets could match the CNN performances when carefully tuned. Indeed, we will conduct such optimization and propose an ablation study in a near future.

However, although the CNN appears more consistent across different settings, it exhibits more outliers compared to the SPDNet and SPDBNNet. This indicates a preferable behavior for the Riemannian DL models in a context of transfer learning. These results were obtained by predicting a code in a fixed time window of less than 2 seconds. For instance, the mean prediction time for SPDBNNet in the SS settings was 1.438 seconds, while for CNN in the SS settings, it was 1.458 seconds. However, it is important to note that the training time of SPDBNNet is longer than that of SPDNet, which was 1.442 seconds, which in turn is longer than the training time of the CNN.

ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024, i6doc.com publ., ISBN 978-2-87587-090-2. Available from http://www.i6doc.com/en/.

## 5 Conclusion

This article presented two Riemannian deep learning (DL) models that were on par with the state-of-the-art model in terms of accuracy and decision time for cVEP-based BCIs. These models were more robust to outliers and had better accuracy for the lowest-performing subjects. This first study, which employed a vanilla SPDNet architecture without optimization, showed promising results. We plan to work on more complex approaches for transfer learning to tackle cross-subject variabilities. In this study, we restricted our analysis to crosssubject evaluation, but we are confident that cross-session evaluations for a given subject might greatly benefit from these findings.

## References

- [1] V. Martinez-Cagigal, J. Thielen, E. Santamaria-Vazquez, S. Perez-Velasco, P. Desain, and R. Hornero. Brain-computer interfaces based on code-modulated visual evoked potentials (cVEP): a literature review. Journal of Neural Engineering , page 22, 2021.
- [2] K. Cabrera Castillos, S. Ladouce, L. Darmet, and F. Dehais. Burst c-VEP based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience. NeuroImage , page 11, 2023.
- [3] S. Saha and M. Baumert. Intra- and inter-subject variability in EEG-based sensorimotor brain computer interface: A review. Frontiers in Human Neuroscience , page 8, 2020.
- [4] S. Khazem, S. Chevallier, Q. Barth' elemy, K. Haroun, and C. Noˆ us. Minimizing subjectdependent calibration for bci with riemannian transfer learning. In IEEE/EMBS Conference on Neural Engineering (NER) , pages 523-526, 2021.
- [5] F. Fahimi, Z. Zhang, W. Boon Goh, T. Lee, K. Keng Ang, and C. Guan. Inter-subject transfer learning with end-to-end deep convolutional neural network for EEG-based BCI. Journal of Neural Engineering , page 13, 2018.
- [6] B. Aristimunha, R. Y. de Camargo, W. H. Lopez Pinaya, S. Chevallier, A. Gramfort, and C. Rommel. Evaluating the structure of cognitive task with transfer learning. In NeurIPS workshop AI4Science , page 19, 2023.
- [7] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing , 112:172-178, 2013.
- [8] F. Yger, M. Berar, and F. Lotte. Riemannian approaches in brain-computer interfaces: A review. IEEE TNSRE , 25(10):1753-1762, 2017.
- [9] P. Zanini, M. Congedo, C. Jutten, S. Said, and Y. Berthoumieu. Transfer learning: a riemannian geometry framework with applications to brain-computer interfaces. IEEE TBME , page 12, 2018.
- [10] Z. Huang and L. Van Gool. A riemannian network for SPD matrix learning. In AAAI , page 9, 2017.
- [11] R. Kobler, J. Hirayama, Q. Zhao, and M. Kawanabe. SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG. In NeurIPS , 2022.
- [12] S. Chevallier, I. Carrara, B. Aristimunha, P. Guetschel, S. Sedlar, B. Lopes, S. Velut, S. Khazem, and T. Moreau. The largest eeg-based bci reproducibility study for open science: the moabb benchmark. arXiv preprint arXiv:2404.15319 , 2024.

<!-- image -->

## What is the best model for decoding neurophysiological signals? Depends on how you evaluate

Bruno Aristimunha, Thomas Moreau, Sylvain Chevallier, Raphael Y de Camargo, Marie-Constance Corsi

## To cite this version:

Bruno Aristimunha, Thomas Moreau, Sylvain Chevallier, Raphael Y de Camargo, Marie-Constance Corsi. What is the best model for decoding neurophysiological signals? Depends on how you evaluate. CNS 2024 - 33rd Annual Computational Neuroscience Meeting, Jul 2024, Natal, Brazil. ￿hal-04743845￿

## HAL Id: hal-04743845

## https://inria.hal.science/hal-04743845v1

Submitted on 18 Oct 2024

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

<!-- image -->

## What is the best model for decoding neurophysiological signals? Depends on how you evaluate

Bruno Aristimunha$^{1,2}$, Thomas Moreau$^{3}$, Sylvain Chevallier$^{1}$, Raphael Y. de Camargo$^{2}$, Marie-Constance Corsi 4

- 1. Inria TAU, LISN-CNRS, Université Paris-Saclay, 91405, Orsay, France
- 2. Center for Mathematics, Computing and Cognition, Universidade Federal do ABC, Santo André, Brazil
- 3. Inria Mind team, Université Paris-Saclay, CEA, Palaiseau, 91120, France
- 4. Sorbonne Université, Institut du Cerveau - Paris Brain Institute -ICM, CNRS, Inria, Inserm, AP-HP, Hôpital de la Pitié Salpêtrière, Paris, France
- 5. Inria NERV team, Paris, France

Non-invasive brain-computer interface (BCI) is a framework that establishes direct communication between a computational external device and the brain activity, mostly via electroencephalography (EEG) signals. Despite its clinical applications, EEG-based BCI presents several challenges, such as performance variability across subjects and a low amount of data. To tackle these issues, many approaches have been proposed to better highlight and understand the neural dynamics reflected in the signals, including signal processing tools (e.g. band-pass filters, alignments), alternative features such as functional connectivity (Corsi M.-C. et al. 2022), or more sophisticated classification models (based on manifolds or deep learning). Despite all these efforts, many results fail to provide a consistent answer to which type of model is the best to understand brain dynamics; even when they use the same data, they use different evaluation schemes. In this study, we are interested in the following questions: (i) Does the way the model is evaluated impact the ranking of the best model? (ii) Does the amount of data impact the decoding of the brain signals, and is this reflected in the ranking? (iii) Do the best models also deliver better interpretability? Here, we systematically evaluated different methods, 7 using deep learning, 10 using Riemannian Manifold, and Common Spatial Patterns across six EEG-based BCI datasets during the sensory-motor rhythms tasks. All these methods were benchmarked using the same data split, with the classification task determining which motor imagery task occurred during a trial. The results were consistent with prior reports (Chevallier, S. et al. 2024). For instance, the best deep learning model, Attention Net (Wimpff, M., Gizzi, et al. 2024), outperformed the best Riemannian model, Fucone (Corsi M.-C. et al. 2022), by 16% in five-fold cross-validation at the subject level on the BCI 2014 competition dataset. However, the ranking remained inconsistent when we changed the evaluation method. The amount of data

used as input in the model was decisive for deep learning models, while the manifold models proved to be more invariant to this factor when trained with one model for a subject. Finally, the inherent interpretability of the functional connectivity models was not effective for scenarios with many subjects. These results emphasize the necessity of a systematic comparison of brain decoding models, drawing a parallel with the benchmark approaches that built the foundation in deep learning fields, which now could be adopted in neuroscience.

## References

Corsi, M. C., Chevallier, S., De Vico Fallani, F., & Yger, F. (2022). Functional connectivity ensemble method to enhance BCI performance (FUCONE). IEEE Transactions on Biomedical Engineering , 69 (9), 2826-2838.

Wimpff, M., Gizzi, L., Zerfowski, J. and Yang, B., 2024. EEG motor imagery decoding: A framework for comparative analysis with channel attention mechanisms. Journal of Neural Engineering , 21 (3), p.036020.

Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Sedlar, S., Lopes, B., ... & Moreau, T. (2024). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv preprint arXiv:2404.15319 .

## Acknowledgments

The work of BA was supported by DATAIA Convergence Institute as part of the ''Programme d'Investissement d'Avenir'', (ANR-17-CONV-0003) operated by LISN.

## E-mails

b.aristimunha@gmail.com thomas.moreau@inria.fr sylvain.chevallier@universite-paris-saclay.fr raphael.camargo@ufabc.edu.br marie-constance.corsi@inria.fr

<!-- image -->

Bruno Aristimunha$^{1,2}$, Thomas Moreau$^{3}$, Sylvain Chevallier$^{1}$, Raphael Y. de Camargo$^{2}$, Marie-Constance Corsi 4

1. Inria TAU, LISN-CNRS, Université Paris-Saclay,| 2. CMCC, UFABC, Santo André, Brazil | 3. Inria Mind, CEA, Université Paris-Saclay, France | 4. Sorbonne Université, ICM, CNRS, Inria NERV, Inserm, France

## What is decoding neurophysiological

## signals?

Here, the neurophysiological time series x depends on the stimulus y

<!-- image -->

When you train a machine learning model, you learn how to decode an task.

## What is it benchmark?

Benchmarking is an emerging science , and we understand it as the iron rule to tame anything goes. All disputes must be settled by competitive empirical testing:

- 1) Agree on metric;
- 2) Agree on benchmark data;
- 3) Compete (Compute).

## Dataset and models (what we agree)

Here, we selected four motor-imagery

## datasets:

And we selected 17 machine learning models:

| NAME            |   YEAR | CATEGORY   |
|-----------------|--------|------------|
| BIOT            |   2023 | DEEP       |
| AUG-COV         |   2024 | FC         |
| ATCNet          |   2022 | DEEP       |
| AttentionNet    |   2024 | DEEP       |
| EEGITNet        |   2022 | DEEP       |
| EEGInception    |   2020 | DEEP       |
| EEGNetv4        |   2018 | DEEP       |
| ShallowFBCSPNet |   2016 | DEEP       |
| TIDNet          |   2020 | DEEP       |
| Cov-CSP-LDA     |   2008 | FC         |
| Cov-CSP-LDA     |   2008 | FC         |
| Cov-FgMDM       |   2010 | FC         |
| Cov-MDM         |   2010 | FC         |
| Cov-Tang-LogReg |   2010 | FC         |
| Fucone          |   2022 | FC         |
| Cov-Tang-SVM    |   2010 | FC         |
| Var-LDA Log     |   2008 | FC         |
| Var-SVM Log     |   2008 | FC         |

Deep learning

Functional Connectivity

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

## How do we evaluate?

## Inter vs Intra models

<!-- image -->

## Experimental results

1.

We can decode the time series! One subject for model |

multi subject for model

| solver name      | BNCI2014 001   | BNCI2014 004   | BNCI2015\_001   | Weibo2014   | BNCI2014\_001   | BNCI2014\_004   | BNCI2015\_001   | Weibo2014   |
|------------------|----------------|----------------|----------------|-------------|----------------|----------------|----------------|-------------|
| ATCNet           | 38.93412.80    | 63.01418.96    | 55.79+13.37    | 27.2146.41  | 35.07+4.11     | 74.78+4.46     | 53.27+2.90     | 18.4512.62  |
| AttentionBaseNet | 39.85413.73    | 57.65416.93    | 57.39+16.03    | 26.70+6.87  | 47.41+6.61     | 67.36+7.31     | 66.03+5.57     | 18.7011.95  |
| AUG-COV          | 67.47418.67    | 70.22+18.04    | 81.43414.08    | 46.53410.92 | 33.70+3.77     | 60.95+6.27     | 59.12+5.17     | 18.46+4.11  |
| BIOT             | 33.33410.47    | 55.75416.01    | 58.00+14.33    | 27.94+6.80  | 26.65+1.72     | 66.44+2.24     | 53.28+0.93     | 17.43+1.81  |
| Cov-CSP-LDA shr  | 61.53417.93    | 71.15417.55    | 78.46+15.14    | 36.83+11.66 | 32.9545.68     | 65.2442.88     | 54.42+2.06     | 16.67+3.23  |
| Cov-CSP-LDA svd  | 62.87417.41    | 72.29416.20    | 77.36416.65    | 38.34+12.05 | 35.4545.15     | 63.69+3.25     | 59.04+6.00     | 17.93+4.45  |
| Cov-FgMDM        | 69.20+16.29    | 70.78418.21    | 81.00413.94    | 60.54+11.44 | 32.7643.85     | 61.61+3.66     | 55.44+1.39     | 18.48+3.17  |
| Cov-MDM          | 60.27+15.26    | 69.03+17.92    | 75.43+17.26    | 32.56411.08 | 30.4545.67     | 66.83+3.31     | 57.6545.98     | 18.98+2.53  |
| Cov-Tang-LogReg  | 70.19+16.49    | 72.10+17.45    | 81.96414.49    | 64.36+9.41  | 36.35+6.75     | 64.56+3.31     | 57.00+2.69     | 18.30+2.91  |
| Fucone           | 70.27+15.49    | 70.64+18.16    | 81.50+13.61    | 63.25410.96 | 32.38+3.73     | 60.45+4.46     | 54.59+1.48     | 17.80+2.43  |
| Cov-Tang-SVM     | 67.32+15.83    | 69.80+17.53    | 82.36413.89    | 65.30+9.92  | 30.24+4.75     | 65.20+3.57     | 54.7643.20     | 17.72+2.75  |
| DUMMY            | 15.71+4.24     | 40.53+7.81     | 41.82+6.27     | 9.03+3.04   | 25.00+0.00     | 50.00+0.00     | 50.00+0.00     | 14.29+0.00  |
| EEGInception     | 30.77412.07    | 54.41+16.92    | 53.89+12.58    | 21.66+6.79  | 30.21+1.95     | 70.41+4.81     | 58.20+1.69     | 16.39+2.72  |
| EEGNetv4         | 32.91+9.77     | 52.13413.97    | 51.89+9.99     | 22.98+7.97  | 42.73+5.08     | 74.7443.42     | 65.01+4.45     | 18.1313.53  |
| LogVar-LDA       | 56.36414.77    | 69.02+17.51    | 72.43416.84    | 51.5548.31  | 35.69+5.47     | 64.80+5.49     | 55.98+3.30     | 20.29+1.84  |
| LogVar-SVM       | 42.76416.46    | 66.52+20.04    | 63.86417.39    | 41.94+7.51  | 33.19+2.29     | 63.3341.38     | 58.39+4.41     | 18.16+2.71  |
| ShallowFBCSPNet  | 48.24+16.57    | 67.69+19.03    | 67.07+17.03    | 45.47410.30 | 56.09+6.17     | 73.2642.24     | 64.67+4.11     | 27.07+4.91  |
| TIDNet           | 34.21+10.21    | 62.72+20.05    | 55.25413.12    | 30.21+7.45  | 35.40+2.77     | 75.10+4.45     | 63.41+1.64     | 21.68+3.54  |

## 2. The model ranks change completely!

<!-- image -->

<!-- image -->

<!-- image -->

Intra-Session

Inter-Subjects

<!-- image -->

## Take-Home insights

- 1) The way the evaluation is the devil in the details !
- 2) The variance dataset isn't necessarily the best for all. New models should be evaluated across different datasets, and statistical conclusions should be drawn.

of ranks between datasets is large; the best model for one

- 3) More data appears to improve deep learning models. It doesn't make sense to build deep learning models if we train one model per subject, traditional models will be better;
- 4) We need to optimize the models more, but we have a tradeof in computational costs, just here in an exploratory study we trained more than 10500 models.

Linkedin:

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Weibo20l4 rank

<!-- image -->

## Neuronal avalanches as potential features for Brain-Computer Interfaces

Marie-Constance Corsi, Pierpaolo Sorrentino, Denis P Schwartz, Nathalie George, Leonardo L. Gollo, Sylvain Chevallier, Ari E. Kahn, Sophie Dupont, Danielle S Bassett, Viktor Jirsa, et al.

## To cite this version:

Marie-Constance Corsi, Pierpaolo Sorrentino, Denis P Schwartz, Nathalie George, Leonardo L. Gollo, et al.. Neuronal avalanches as potential features for Brain-Computer Interfaces. Organization of Human Brain Mapping. OHBM 2024 - Organization for Human Brain Mapping, Jun 2024, Seoul, South Korea. Abstract Book 6: OHBM 2024 Annual Meeting https://doi.org/10.52294/001c.120596, 4 (1), 2024. ￿hal-04701010￿

## HAL Id: hal-04701010

## https://inria.hal.science/hal-04701010v1

Submitted on 18 Sep 2024

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

<!-- image -->

## Neuronal avalanches as potential features for Brain-Computer Interfaces

Marie-Constance Corsi *, 1

Denis Schwartz 1

Nathalie George 1

Ari E. Kahn 6

Sophie Dupont 1

Danielle S. Bassett 6

Viktor Jirsa

*, 2

Pierpaolo Sorrentino *, 2,3

Leonardo Gollo 4

Sylvain Chevallier 5

Fabrizio De Vico Fallani *, 1

1 Sorbonne Université, Institut du Cerveau - Paris Brain Institute, ICM, CNRS, Inria, Inserm, AP-HP,

Hôpital de la Pitié-Salpêtrière, Paris, France

2

3

Institut de Neurosciences des Systèmes, Marseille, France

University of Sassari, Department of Biomedical Sciences, Viale San Pietro, 07100, Sassari, Italy

4

5

Monash University, Melbourne, Victoria, Australia

Université Paris-Saclay, Saclay, France

6 University of Pennsylvania, Philadelphia, USA

Brain-Computer Interfaces (BCIs) constitute a promising tool for communication and control. However, mastering non-invasive BCI systems remains a learned skill difficult to develop for a non-negligible proportion of users. Even though similarities have been shown between MI-based BCI learning and motor sequence learning our understanding of the dynamical processes, and their reflection on brain signals during BCI performance is still incomplete. In particular, whole-brain functional imaging is dominated by a 'bursty' dynamics, "neuronal avalanches", with fast, fat-tailed distributed, aperiodic perturbations spreading across the whole brain. Neuronal avalanches evolve over a manifold during resting-state, generating a rich functional connectivity dynamics. In this work, we evaluated to which extent neuronal avalanches can be used as a tool to differentiate mental states in the context of BCI experiments.

## BCI experiment

The BCI task consisted of a standard 1D, two-target box task in which the subjects modulated their α and/or β activity. To hit the target-up, the subjects performed a sustained motor imagery of their righthand grasping and to hit the target-down, they remained at rest. Twenty subjects (aged 27.45 ± 4.01

<!-- image -->

years, 12 men), all right-handed participated in the study.

Magnetoencephalography (MEG) and electroencephalography (EEG) signals were simultaneously recorded. M/EEG data were preprocessed using the Independent Component Analysis method, followed by the source reconstruction on the epoched data [1].

## Neuronal avalanches computation

<!-- image -->

Neuronal avalanche

Transition Matrix

<!-- image -->

The signal was z -scored and thresholded, and set to 1 when above threshold , and to zero otherwise (threshold = |3|). An avalanche was defined as starting when at least one region is above threshold, and as finishing when no region is active. For each avalanche, we have estimated a transition matrix (TM) containing the probability that regions j would be active at time t+1, when region i was active at time t [2].

For each subject, we obtained an average transition matrix for the baseline condition (Rest), and an average transition matrix for the motor imagery task (MI).

We set out to test if neuronal avalanches would be suitable to track subjectspecific changes induced by a task in the large-scale brain dynamics. Our approach might capture part of the processes that were typically overlooked in a more oscillatory perspective. Our work paves the way to use aperiodic activities to improve classification performance and tailor BCI training programs.

Interested in this study? Scan the QR code to get access to the associated paper!

<!-- image -->

<!-- image -->

<!-- image -->

pierpaolo.sorrentino@univ-amu.fr

marie-constance.corsi@inria.fr

<!-- image -->

@PierpaSorre

@MConstanceCorsi

<!-- image -->

## Neuronal avalanches to inform mental states

After identifying at the individual level the edges and the nodes, that show a significant condition effect, we looked at the concordance of such differences across subjects. This way, we spotted the edges were engaged differently by perturbations in the two conditions in most subjects. The reliably different edges cluster on premotor regions bilaterally.

<!-- image -->

<!-- image -->

Then, we have summed the differences corresponding to all the edges which are incident upon each reliably different region obtaining an overall difference per each such node . In particular , the caudal middle frontal gyri bilaterally are the main regions upon which edges which differ between conditions cluster. These regions are involved in planning of motor actions, imagining of actions, as well as in executive attention [3] and in the selection between competing visual targets [4].

To assess to which extent neuronal avalanches can predict BCI scores, we focused on the differences in the transition probabilities in the edges which are incident upon the "taskrelated" regions. For each of these edges, we correlated the difference in probability in each subject with the performance in the MI task, as measured using the BCI score. Our findings suggest that perturbations spread more often between premotor/motor areas and parietal areas when the subject is engaged in the motor-imagery task as compared to resting -state condition .

## Neuronal avalanches to perform classification

<!-- image -->

To explore the performance of neuronal avalanches in the decoding of the tasks, we compared the ATMs to the common spatial patterns ( CSPs ) approach . Both were classified with a Support Vector Machine (SVM) .

<!-- image -->

At the group level, the classification performance was greater for ATM+SVM than CSP+SVM. We observed greater inter-subject variability in the case of CSP+SVM (std = 0.15) than with ATM+SVM (std = 0.10). At the invidual level, ATM+SVM yielded better classification accuracy than CSP+SVM for 12 subjects. In four subjects, CSPs yielded better accuracy than ATMs. In 5 subjects, there was not any statistically significant difference between the two approaches.

References

- [1] Corsi, et al (2020). Functional disconnection of associative cortical areas predicts performance during BCI training. NeuroImage [2] Sorrentino, et al (2021). The structural connectome constrains fast brain dynamics. Elife
- [4]
- [3] Andersson, et al (2009). Correlations between measures of executive attention and cortical thickness of left posterior middle frontal gyrus-A dichotic listening study. Behavioral and Brain Functions Germann, et al (2020). Area 8A within the Posterior Middle Frontal Gyrus Underlies Cognitive Selection between Competing Visual
- Germann, et al (2020). Area 8A within the Posterior Middle Frontal Gyrus Underlies Cognitive Selection between Competing Visual Targets. ENeuro

## Acknowledgements

The authors acknowledge support from European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 864729), from the program "Investissements d'avenir" ANR-10-IAIHU-06, from European Union's Horizon 2020 research and innovation programme under grant agreement No. 945539 (SGA3) Human Brain Project, and from VirtualBrainCloud No.826421.

<!-- image -->

<!-- image -->

<!-- image -->

## Neuronal avalanches for eeg-based motor imagery BCI

Camilla Mannino, Pierpaolo Sorrentino, Mario Chavez, Marie-Constance Corsi

## To cite this version:

Camilla Mannino, Pierpaolo Sorrentino, Mario Chavez, Marie-Constance Corsi. Neuronal avalanches for eeg-based motor imagery BCI. 9th Graz Brain-Computer Interface Conference 2024, Sep 2024, Graz, Austria. pp.98, ￿10.3217/978-3-99161-014-4-018￿. ￿hal-04698548￿

## HAL Id: hal-04698548

## https://hal.science/hal-04698548v1

Submitted on 16 Sep 2024

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

## NEURONALAVALANCHESFOREEG-BASEDMOTORIMAGERYBCI

C. Mannino $^{1}$, P. Sorrentino$^{2,3}$, M. Chavez$^{1}$, M.-C. Corsi$^{1}$

$^{1}$Sorbonne Université, Institut du Cerveau - Paris Brain Institute -ICM, CNRS, Inria, Inserm, AP-HP, Hôpital de la Pitié Salpêtrière, F-75013, Paris, France $^{2}$Institut de Neurosciences des Systèmes, Aix-Marseille Université, 13005 Marseille, France $^{3}$University of Sassari, Department of Biomedical Sciences, Viale San Pietro, 07100, Sassari, Italy

E-mail: marie-constance.corsi@inria.fr

## ABSTRACT:

Current features used in motor imagery-based BrainComputer Interfaces (BCI) rely on local measurements that miss the interactions among brain areas. Such interactions can manifest as bursts of activations, called neuronal avalanches. To track their spreading, we used the avalanche transition matrix (ATM), which contains the probability that an avalanche would consecutively recruit any two brain regions. Here, we proposed to use ATMs as a potential alternative feature. We compared the classification performance resulting from ATMs to a benchmark model based on Common Spatial Patterns. In both sensor-and source-spaces, our pipeline yielded an improvement of the classification performance associated with reduced inter-subject variability. A correspondence between the selected features with the elements of the ATMs that showed a significant condition effect led to higher classification performance, which speaks to the interpretability of our findings. In conclusion, working in the sensor space provides enough spatial resolution to classify. However the source space is crucial to precisely assess the involvement of individual regions.

## INTRODUCTION

Neuroscientists have been exploring and researching Brain-Computer Interface (BCI) since the70s as a way to restore communication and motor capabilities for severely disabled people., such as patients affected by amyotrophic lateral sclerosis, stroke, or spinal cord injury [1].

In non-invasive BCI, Event Related Desynchronization/Synchronization, Event Related Potentials, and Steady State Evoked Potentials are the most informative brain activity patterns for communication and control applications to design electroencephalography (EEG)-based BCI. One of the main drawbacks of the current systems lies in the high inter/intra-subject variability, notably in terms of performance. Indeed, multiple studies reported that 15%-30% of the subjects fail in controlling a BCI device. This is a phenomenon referred to as the "BCI inefficiency" [2]. Among the potential causes are the selected data features. Indeed, relying mostly on local measurements might not effectively capture brain functioning, as some information is encoded in the interactions between areas [3].

To overcome these limitations and to take advantage of the EEG time-resolution, in a recent work, we proposed

to use a metric that captures the dynamic nature (i.e. changing in space and time) of the brain activities: the neuronal avalanches. Neuronal avalanches are characterised by the propagation of cascading bursts of activity [4]. Previous studies show that their spreading preferentially across the white-matter bundles [5] and that neuronal cascades are a major determinant of spontaneous fluctuations in brain dynamics at rest [6]. Furthermore, in our previous work we showed that neuronal avalanches, estimated from sourcereconstructed data, spread differently according to the task performed by the user, demonstrating the potential relevance of neuronal avalanches as an alternative feature for detecting the subjects' intent [7].

Here, we investigated to which extent this framework would be compatible with a BCI experiment. For this purpose, instead of working in the source domain that requires additional data (e.g. individual magnetic resonance imaging) and computational resources, we tested the performance of neuronal avalanches directly in the sensor domain. Indeed, the methodological validity of sensor space measures is especially relevant for online studies in a clinical setting due to time and economic constraints. We hypothesised that despite a reduction of the spatial resolution, using the neuronal avalanches in the sensor space could help in classification performance, and that the selected features could be neurophysiologically interpretable and relevant.

## MATERIALS AND METHODS

## Participants

The research was conducted in accordance with the Declaration of Helsinki. A written informed consent was obtained from subjects after explanation of the study, which was approved by the ethical committee CPP-IDFVI of Paris. All participants received financial compensation at the end of their participation. Twenty healthy subjects (27.5 ± 4.0 years old, 12 men), with no medical or psychological disorder, were recruited.

## Experimental protocol

The dataset used in our study originates from Corsi et al. [8] and involves a BCI task structured around a twotarget box task. Participants were required to adjust their brain's alpha and/or beta activity levels to control a cursor's vertical movement, aiming to reach a vertical bar, referred to as a target displayed on the screen. Achieving the upper target necessitated the subjects to engage in continuous motor imagery (MI) of right-hand

grasping. Conversely, reaching the lower target required the subjects to remain in a resting state. Each session comprised 32 trials, evenly and randomly split between the up and down targets, correlating with the MI and Rest conditions, respectively. For a complete description of the protocol, the reader can refer to [8] .

## EEG data acquisition & pre-processing

EEG data were captured using a 74-channel EEG system equipped with Ag/AgCl passive sensors (Easycap, Germany), arranged according to the 10-10 standard montage. Reference electrodes were placed on the mastoids, with the ground electrode on the left scapula. Recordings took place in a magnetically shielded room, utilising a 0.01-300Hz bandwidth and sampling at 1kHz. Two channels (namely T9 and T10) were identified as bad and rejected based on the amplitude of the signals recorded, with a threshold of three standard deviations. For a complete description of the pre-processing steps, please refer to [8] .

## Neuronal Avalanches extraction

The neuronal avalanches analysis consists of identifying large signal excursions beyond a given threshold. The cascades are captured by clustering these discrete suprathreshold events based on temporal proximity, thus, defining neuronal avalanches as periods of collective spatio-temporal organization. Each signal was z-scored (over time), and set to 1 when above a threshold, and to 0 otherwise. An avalanche was defined as starting when at least one channel is above threshold (referred here as active channel), and as finishing when all channels were inactive [4,5,6]. For each avalanche, we estimated a transition matrix A , called Avalanche Transition Matrix (ATM), structured with channels in rows and columns, and the ij $^{th}$ element of matrix A defined as the probability that the electrode j would be active at time t+1, given the electrode i was active at time t. For each subject, we obtained a transition matrix over all avalanches for each condition (MI and Rest conditions).

## Classification Analysis

To explore the applicability of the ATM method in the context of a BCI training, we performed a subjectspecific analysis.

The classification step was done using a Support Vector Machine (SVM). To assess the extent to which the ATMs might be considered as an alternative feature for BCIs, we compared our approach (ATM+SVM) to a framework that relies on spatial filters, namely Common Spatial Patterns (CSP+SVM) [9, 10].

For each approach (namely ATM+SVM or CSP+SVM), we classified different tasks at the individual level. To evaluate the classification performance, we divided the dataset to include 80% of the trials in the train split and 20% of the trials in the test split. The classification scores for all pipelines were evaluated with an accuracy measurement using a random permutation crossvalidator. To assess the robustness of our framework, we also tested a different number of re-shuffling & splitting iterations (5/25/50/75).

For each subject, the CSP method decomposes signals using spatial filters, and then selects the n modes that capture the highest inter-class variance. Here, we selected eight spatial modes and returned the average power of each.

As for the ATMs, to consider the subjects' specificity, we optimised two parameters, namely: the threshold applied to the z-scored signals (ranging from 1.0 to 4.0), and the minimal duration of the considered avalanches (ranging from 2 to 8) [11]. Inside the ATM pipeline the choice of the best decoding parameters relied on a posteriori classification accuracy performance rate.

Finally, we individually compared the classification performance obtained with the CSP+SVM and with the ATM+SVM approaches, respectively. We run t-tests under the null hypothesis that, for a given subject, CSP+SVM and ATM+SVM would not yield statistically significant differences in classification. We repeated the comparison for all the subjects and corrected these statistical comparisons for multiple comparisons across subjects using the False Discovery Rate (FDR) [12]. Such an analysis has been performed across 25, 50, and 75 splits. However, given its poor statistical power, it is not possible to apply a statistical test over 5 splits classification. Therefore, to evaluate whether the difference between the two pipelines could be considered as significant, we c alculated the averaged classification performance across splits using both CSP+SVM and ATM+SVM and we determined the difference for each subject. Ultimately, we compared the absolute value of the difference with our predefined threshold, considering the classification performance not statistically different if the magnitude of the difference between the two methods was less than a threshold. W e established the threshold at an arbitrary value of 0.05. As a sanity check, we performed this analysis in the source space, as we did in [7]. For a complete description of the sourcereconstruction steps, the reader can refer to [8].

In this work, we used preprocessed signals that were bandpass filtered between 3 and 40Hz. To investigate the potential effect induced by the choice of the frequency band, we performed the same analysis in the μ band (8 13 Hz) and in the beta band (13 - 30 Hz) [not shown]. We performed a one way ANOVA (df = 2) among these three-frequency bands under the null hypothesis (H0) that these groups came from the same population. For both approaches (CSP+SVM and ATM+SVM), no frequency band effect on the classification performance was observed (p-value > 0.05). Therefore, in the next sections, we will report the results were obtained within the 3-40 Hz band.

## Decoding: Features importance analysis

To investigate the interpretability of the classification performance, we examined the relative importance of the features derived from the absolute values of the classification coefficients in the model. To better understand the features importance across subjects, we carried out a quantitative reliability analysis across the cohort to identify the repetition of the selected features in

Figure 1: Classification performance. (A/D) Effect of splits tested on ATM+SVM and CSP+SVM at group level in sensor-space (A) and source-space (D). (B/E) Individual level classification performance in sensor-space (B) and source-space (E) using 50 random splits. (D/F) Individual level classification performance in sensor-space (D) and source-space (F) using 5 random splits. Color coded: in salmon, ATM+SVM pipeline; in violet, CSP+SVM pipeline. Statistical difference between CSP + SVM & ATM + SVM: * pval < 0.05, ** pval < 0.01.

<!-- image -->

at least half of the subjects. To investigate features importance from a nodal point of view we set as a threshold the median value across channels and subjects, then we evaluated which nodes were over threshold in the majority of the We computed this analysis over the entire dataset (20 subjects) but also independently on two different subgroups: on the 10 most responsive subjects according to ATM classification performance and on the 10 least responsive subjects respectively. All these investigations were also performed in the source space.

## Encoding: Quantification and statistical analysis

To identify the edges (i.e. functional links) that are more likely to be recruited during a hand motor imagery task as compared to resting state, for each participant, we calculated the variance in the probability of perturbations traversing a specific edge between resting state and MI task. To assess the statistical significance, we randomized the labels of individual avalanches for each person. This shuffling was repeated 10,000 times to generate a distribution of differences for each edge under the null hypothesis that the transition matrices revealed no distinction between the two conditions. We then determined statistical significance for each edge against this null distribution, applying Benjamini-Hochberg correction for multiple comparisons across edges. This process yielded a matrix for each subject, highlighting S i,j values (here referred to as edges) with statistically significant differences between conditions. We assessed the consistency of these matrices across individuals, concentrating on edges consistently implicated in the task. Then, we performed a node-wise analysis to identify the nodes over which significant differences were clustered. These nodes were referred to as ''taskspecific'' areas.

RESULTS

Classification performance

Working on the entire dataset of 20 subjects, as a

standard configuration, we used 50 random splits.

At the group-level, the classification performance in the sensor space, between CSP+SVM and ATM+SVM is similar (t-test, pval > 0.05). Nevertheless, we observed a larger inter-subject variability with CSP+SVM (71%+/15%) as compared to ATM+SVM (71% +/- 9%). In the source-space, ATM+SVM (80%+/-8%) led to a statistical improvement of the classification performance as compared to CSP+SVM (75%+/-14%) (t-test, pval<0.05) such as a reduced inter-subject variability. At the individual level, in the sensor-space ATM+SVM yielded a statistically better classification accuracy than CSP+SVM for 9 subjects. In 8 subjects, CSPs yielded better accuracy than ATMs. In 3 subjects, there was not any statistically significant difference between the two approaches (Fig. 1B) . In the source-space, ATM+SVM yielded significantly higher classification accuracy than CSP+SVM for 13 subjects, while the opposite was true for 4 subjects. For the remaining 3 subjects, there was not any statistically significant difference between the decoding performances of the two approaches (Fig. 1E) .

To investigate the possibility to reduce the computational time to get closer to a configuration more compatible with the online requirements, we investigated the accuracy performance across different random splits configurations (5, 25, 50, 75) both at the individual and at the group level. As shown in Fig. 1A & D the performance was robust across splits for both CSP+SVM and ATM+SVM pipelines (one-way ANOVA p>0.05), and we observed a higher accuracy score for most of the subjects with 5 splits both in sensor and source space. Based on the observations made on the inter-subject variability, we validated the significant difference of the variance of these two populations via the F-test (pval<0.05). The statistical difference between the two pipelines was achieved both in the sensor and in the source space.

Figure 2: Features selection.

<!-- image -->

<!-- image -->

When considering 5 splits, at the group level, in the sensor space, no significant difference was observed between the two pipelines (t-test, p-value > 0.05) but CSP+SVM (72% +/- 15.55%) showed a larger intersubject as compared to ATM + SVM (73% +/- 9.14%). In the source space, ATM + SVM (81% +/- 7.5%) led to a statistical improvement of the performance (t-test, pval<0.01) as compared to CSP + SVM (75% +/- 14%) and a significant reduction of inter-subject variability. At the individual level: in the sensor-space (Fig.1C) , with 5 splits, ATM+SVM yielded a statistically better classification accuracy than CSP+SVM for 9 subjects. In 7 subjects, CSPs yielded better accuracy than ATMs and in four subjects, there was not any significant difference between the two approaches. However, CSP+SVM pipeline led to a larger number of subjects with a performance below the chance level, set to 58% here [13] (6 subjects) than with ATM+SVM (1 subject). In the source space, ATM+SVM showed an improved performance in 12 subjects, while the opposite was true for 3 subjects with CSP+SVM (Fig. 1F) .

From now on, unless specified otherwise, the chosen configuration will involve 5 splits to closely mimic a real-time setup, and the subsequent sections will deal with ATM data only.

Sensor and source space selected features

To investigate the interpretability of the decoding performance, we estimated the weights attributed to each feature. A preliminary probabilistic analysis showed that most of the selected features presented a lower feature importance and that only a few were notably higher, suggesting that only a reduced number of features were relevant . When considering the features selected in at

<!-- image -->

(A) Edges-wise, valid at group level in sensor-space

(B) Edges-wise, valid at group level in source-space

least half of the cohort, an edge invo l ving left central electrodes (C5) and occipital electrodes (O2) was obtained in 13 subjects (Fig. 2A) . We observed a predominant involvement of left central electrodes connected to occipital electrodes, between left and right central electrodes connected to parietal electrodes.

Similar observations were possible in the source-space. Looking for a recurrent path across most of the subjects, see Fig 2B , in 15 subjects, most of the connections involved left paracentral, rostral anterior cingulate cortex, caudal middle frontal gyrus and medial lateral orbito-frontal regions.

These interactions correspond to edge clusters that were taskdependent and consistent across subjects in encoding investigation shown in our previous paper [7]. Moreover, the features with higher weight often involved the left paracentral and the precentral areas. F i g u

To get a more synthetic vision of these results, we performed a similar analysis at the nodal level, confirming the results previously obtained. To increase the statistical validity of such observations, in this part, we worked with the 50-split configuration. In the sensorspace, the electrodes with highest features' importance were C5 and P8. Nevertheless, it is possible to observe a general activation in electrodes over the bilateral motor cortex, and the bilateral parietal lobe. In the source-space, the most frequently selected brain regions were the right paracentral area, the left frontal pole and the right rostral anterior cingulate. r e 2 : F e a t u r e s s

e

Encoding-Decoding Match in sensor-space l

To investigate the neurophysiological validity of the selected features, we compared them with the results obtained with an encoding framework. To achieve this, e c t i

<!-- image -->

o n . Figure 3: Encoding analysis in sensor-space.

( A (A) Encoding reliably different edges cluster at group level in sensor-space

) E (B) Encoding at nodal and group level in sensor-space

d g e s -w i s e

we examined differences between the two experimental conditions in the probabilities of perturbations propagating across two brain regions. Our results show that there is a set of links over CP and P electrodes (CP5, P1, P2 edges-wise and CP1, Pz, P2 at nodal level), in which the difference between two conditions (MI and rest) was consistently significant across most of the subjects (p < 0.0001, BH corrected) (Fig. 3A/B) .

Following the features' importance analysis described in the previous section, we performed a quantitative reliability analysis to consider only the edge-wise selected in at least half of the subjects (Fig. 2A) . The final goal of this analysis was the comparison between reliably different edges selected in the encoding phase and the features selected in most subjects with the larger attributed weight. Indeed, we observed a higher level of match score with the ten subjects with a highest classification performance (37%, see Fig. 4A) as compared to the ten subjects with the lowest classification performance (6%, see Fig. 4B) but also to the entire dataset (9%, see Fig. 4C) .

Figure. 4: Edges matches between encoding and decoding: (A) Results obtained from the ten subjects with the highest classification performance; (B) Results obtained from the ten subjects with the lowest classification performance; (C) Results obtained from the entire dataset.

<!-- image -->

## DISCUSSION

In the sensor space as well as in the source space, the classification of ATMs led to an improvement of the decoding performance with respect to the benchmark (namely the spatial filter-based approach) in most of the subjects robustly across the different number of tested random splits. Importantly, in both source and sensor domains, we observed a reduced intra and inter-subject variability with ATM+SVM as compared to CSP+SVM. These findings suggest that the use of our approach could be a tool to reduce the BCI inefficiency phenomenon.

Beyond the classification performance, we also investigated the interpretability of our findings through the study of the selected features. ATMs present a straightforward interpretability as opposed to CSPs, which operate on large-scale components of the signal that are not as readily interpretable. Indeed, it is possible

to study and to identify the selected features at the subject level but a quantitative analysis at the group level is not applicable because of the difficulty to identify a common precise pattern across different subjects and different selected features. At the individual level, the information captured by the two types of feature extraction (namely CSPs and ATMs) are complementary, as seen in Fig. 5. ATMs is based on edge-wise representations and focus on strong coherent interactions that intermittently occur on the large-scale whereas CSP features, that embedded pipelines based on techniques that assume stationarity, rely on local measurements (mostly frequency band power features and time-point features) disregarding the propagation of brain dynamics at consecutive time instants.

To further study the meaning of the features selected with ATMs, we adopted an encoding framework identified here as a set of functional connections (i.e., edges) that consistently exhibited a higher likelihood of dynamic recruitment during a hand motor imagery task as compared to the resting state at the group level. This straightforward approach, validated on the entire dataset, allowed us to reliably extract functional information specific to the task execution at the individual level, an observation not achievable through traditional functional metrics (namely power spectra and phase-locking value) [7]. Therefore, from a theoretical standpoint, our study establishes the foundation for exploring neuronal avalanche metrics as a novel functional connectivity measure for investigating changes during motor tasks based not on the functional activation between two brain areas at the same time but on consecutive activations.

In the sensor space, the electrodes that showed a higher feature importance, identified through the decoding framework, were located over the same brain areas defined as "reliably edges" in the encoding framework. Moreover, we noticed that an increased match between the selected features and the edges-clusters led to an improvement of the classification performance. This finding suggests a possible way to apply a dimensionality reduction in the features used in the decoding step, to improve the classification performance. An ongoing work consists in investigating the key-parameters of the neuronal avalanches to be tuned and the associated features selection approaches to assure that the most relevant information will be considered for the classification step. Considering such approaches will improve the performance, but they will also reduce the computational time as well; two key-aspects of the feasibility of our pipeline in real-world scenarios.

In our work, to emphasise this possible future development, we dealt with epochs of 5s from which 25ms and 27ms (respectively for ATM + SVM and CSP+SVM) were required to extract the features and to perform the classification. Such computational time estimations are in line with current real time settings that rely on similar time windows and propose an update of the provided feedback every 28 ms. Future work will consist in identifying strategies to extract neuronal

avalanches, and therefore ATMs, in shorter time windows to make our framework completely compatible with online settings.

To further investigate the physiological meaning of our findings we compared the results respectively obtained in the sensor and in the source space. The most frequently selected features involved central electrodes (C-CP) in the sensor-space, and the paracentral area in the sourcereconstructed data, implying the motor-area. Moreover, our results showed that other networks were involved in a motor-imagery task, through the selection of electrodes above parietal and occipital areas. The parietal lobe is structurally divided into inferior parietal lobe, superior parietal lobe, and precuneus [14]; its principal functions are the perception of the body, the integration of somatosensory information (e.g. touch, pain, pressure and temperature), visuospatial processing and coordination of movement. As such, the parietal activation is in line with our observations [7], because the subjects were instructed to perform a kinesthetic motor imagery task, that involves imagining movements as well as sensing the touch caused by the grasped object, a nd because coordinating hand, arm, and eye motions is required to perform our task. A similar role of precuneus in coordination of motor behaviour is achieved by anterior cingulate cortex [13] and its involvement has come to light in source-reconstructed data [7]. The occipital lobe [15] is primarily responsible for visual processing. Its recurrent activation and connection with central electrodes usually happens during a kinesthetic task, and when a visual stimulation is proposed as it was during our experiments.

Moreover, mainly in the source-space, we observed the involvement of the caudal portion of the middle frontal gyrus and of the medial-orbital frontal area. Within the caudal portion of the middle frontal gyrus, at the intersection with the precentral gyrus, is the frontal eye fields (Brodmann area 8). The frontal eye fields control saccadic eye movements, rapid, conjugate eye movements that allow the central vision to scan numerous details within a scene or image, same meaning of orbital regions involvement [16], instead medialorbital frontal region reflects the allocation of attentional resources, which are typically engaged in cognitive/motor tasks [7]. Such findings demonstrate the neurophysiological validity of the selected features.

<!-- image -->

## CONCLUSION

Our results suggest that the integration of periodic and aperiodic features would be a straightforward way to capture functionally relevant processes, in turn, to apply them to the design of BCIs and to improve task classification. The good performance of the ATMs on the EEG data in the sensor space is relevant to translate our methodology to real-world scenarios. Until now, we tested this new feature only during a hand motor imagery BCI task. Future work will consist in considering a wider range of BCI paradigms for communication and movements recovering applications.

## REFERENCES

- [1] Alwi Alkaff, et al. Applications of Brain Computer Interface in Present Healthcare Setting. in Artificial Intelligence vol. 0 (IntechOpen, 2024).
- [2] Allison, B. Z. & Neuper, C. Could Anyone Use a BCI? in Brain-Computer Interfaces (eds. Tan, D. S. & Nijholt, A.) 35-54 (2010).
- [3] Lotte, F. et al. A Review of Classification Algorithms for EEG-based Brain-Computer Interfaces: A 10-year Update. J. Neural Eng. (2018)
- [4] Arviv, O. et al. . Neuronal avalanches and timefrequency representations in stimulus-evoked activity. Sci. Rep. 9 , 13319 (2019).
- [5] Sorrentino, P. et al. The structural connectome constrains fast brain dynamics. eLife 10 , e67400 (2021). [6] Rabuffo, G., et al . Neuronal Cascades Shape WholeBrain Functional Dynamics at Rest. eNeuro 8 , ENEURO.0283-21.2021 (2021).
- [7] Corsi, M.-C. et al. Measuring brain critical dynamics to inform Brain-Computer Interfaces. iScience 27 , 108734 (2024)
- [8] Corsi, M.-C. et al. Functional disconnection of associative cortical areas predicts performance during BCI training. NeuroImage 209 , 116500 (2020).
- [9] Koles, Z.J., et al. Spatial patterns underlying population differences in the background EEG. Brain Topogr. 2 , 275-284 (1990).
- [10] Blankertz, B., et al. Optimizing Spatial filters for Robust EEG Single-Trial Analysis. IEEE Signal Process. Mag. 25 , 41-56 (2008).
- [11] Shriki, O., et al. Neuronal Avalanches in the Resting MEG of the Human Brain. Journal of Neuroscience, 33 (16) 7079-7090.
- [12] Benjamini, Y., and Hochberg, Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. J. R. Stat. Soc. Ser. B Methodol. 57 , 289-300 (1995).
- [13] Müller-Putz, et al. G. Better than random? A closer look on BCI results. International Journal of Bioelectromagnetism (2008).
- [14] Wenderoth, N., et al. The role of anterior cingulate cortex and precuneus in the coordination of motor behaviour. Eur. J. Neurosci. 22 , 235-246 (2005).
- [15] Kwon, S., et al. Neuropsychological Activations and Networks While Performing Visual and Kinesthetic Motor Imagery, Brain Sci. 2023 , 13 ,983.
- [16]Schall J.D.,Frontal Eye Fields, Encyclopedia of Neuroscience. Pages 367-374 (2009).