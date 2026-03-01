% ============================================================
%  NeuroFetal AI — IEEE Conference Paper (Overleaf Compatible)
%  Sections: Abstract, Problem Statement, Literature Survey, Dataset
%  Format  : IEEE two-column (IEEEtran class)
%  Authors : Krishna Sikheriya, Bodkhe Yash Sanjay, Lokesh Bawariya
%  Supervisor: Dr. Nikhilanand Arya, IIIT Allahabad
%  Date    : February 2026
% ============================================================

\documentclass[conference]{IEEEtran}

% ---- Required Packages ----
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{hyperref}
\usepackage{pifont} % for \ding check/cross symbols
\newcommand{\cmark}{\ding{51}}  % checkmark
\newcommand{\xmark}{\ding{55}}  % cross

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}

% ============================================================
\begin{document}

\title{NeuroFetal-AI: A Tri-Modal Architecture with Cross-Modal Attention for Cardiotocography Classification}

\author{
  \IEEEauthorblockN{Krishna Sikheriya}
  \IEEEauthorblockA{\textit{Department of Information Technology}\\
  \textit{Indian Institute of Information Technology Allahabad}\\
  Allahabad, India\\
  iit2023139@iiita.ac.in}
  \and
  \IEEEauthorblockN{Bodkhe Yash Sanjay}
  \IEEEauthorblockA{\textit{Department of Information Technology}\\
  \textit{Indian Institute of Information Technology Allahabad}\\
  Allahabad, India\\
  iit2023180@iiita.ac.in}
  \and
  \IEEEauthorblockN{Lokesh Bawariya}
  \IEEEauthorblockA{\textit{Department of Information Technology}\\
  \textit{Indian Institute of Information Technology Allahabad}\\
  Allahabad, India\\
  iit2023138@iiita.ac.in}
  \and
  \IEEEauthorblockN{Dr. Nikhilanand Arya}
  \IEEEauthorblockA{\textit{Department of Information Technology}\\
  \textit{Indian Institute of Information Technology Allahabad}\\
  Allahabad, India\\
  nikhilanand@iiita.ac.in}
}

\maketitle

% ============================================================
% SECTION I: ABSTRACT
% ============================================================

\begin{abstract}
Intrapartum fetal compromise—encompassing hypoxia and acidosis during
labor—remains a principal cause of perinatal mortality, stillbirth, and
long-term neonatal neurological injury worldwide, with disproportionate
impact in low-and-middle-income countries where trained obstetric personnel are
scarce. Cardiotocography (CTG), the clinical gold standard for intrapartum
monitoring, suffers from poor inter-observer agreement (60–70\%) and high
false-positive rates, motivating the need for reliable, automated decision
support. This paper presents \textbf{NeuroFetal AI v5.0}, a clinically
deployable, uncertainty-aware decision support system that performs tri-modal
fusion of Fetal Heart Rate (FHR) time-series, Uterine Contraction (UC)
signals, and 16-dimensional maternal clinical tabular features. The core
prediction engine is a \textbf{Stacking Ensemble} of three
architecturally diverse base learners: (i) \textit{AttentionFusionResNet}, a
six-block residual network augmented with Squeeze-and-Excitation (SE)
attention and a Cross-Modal Attention fusion layer; (ii)
\textit{1D-InceptionNet}, a multi-scale convolutional network with parallel
kernel branches; and (iii) \textit{XGBoost}, a gradient-boosted decision tree
trained on 35 hand-crafted tabular and Common Spatial Pattern (CSP) features.
A Logistic Regression meta-learner combines out-of-fold predictions via
rank-averaged stacking. To address the severe class imbalance (7.25\%
pathological), we introduce a \textit{TimeGAN}-based augmentation
strategy—a Wasserstein GAN with gradient penalty (WGAN-GP) trained exclusively
on pathological FHR+UC traces—generating 1,410 physiologically coherent
synthetic minority-class recordings that preserve temporal dynamics such as
late decelerations and contraction timing. Model calibration is performed via
Platt Scaling, and epistemic uncertainty is quantified through Monte Carlo
(MC) Dropout combined with information-theoretic metrics (Predictive Entropy
and Mutual Information). Evaluated on the public CTU-UHB Intrapartum
Cardiotocography Database (552 recordings, PhysioNet) under stratified 5-fold
cross-validation, NeuroFetal AI v5.0 achieves \textbf{AUC 0.8639, 96.34\%
Accuracy, 95.22\% F1-Score, Brier Score 0.046, and ECE 0.0543}, surpassing
the state-of-the-art baseline of Mendis et al.\ (AUC 0.84) which relied on an
unpublished private dataset of over 9,800 recordings. The system further
provides Grad-CAM signal-level explainability and is packaged as a 1.9 MB
TFLite Int8 model for offline inference on commodity mobile hardware, enabling
deployment in resource-constrained clinical settings.

\textbf{Index Terms}---Cardiotocography, Fetal Heart Rate, Fetal Compromise,
Deep Learning, Stacking Ensemble, Tri-Modal Fusion, TimeGAN, Uncertainty
Quantification, Explainable AI, Edge Deployment, Clinical Decision Support.
\end{abstract}

% ============================================================
% SECTION II: PROBLEM STATEMENT
% ============================================================

\section{Problem Statement}

\subsection{Clinical Context and Motivation}

Approximately \textbf{2.6 million stillbirths} occur globally each year, with
the vast majority concentrated in low-resource settings where access to trained
obstetric specialists is critically limited \cite{who2020stillbirth}. A
significant proportion of these adverse outcomes are attributable to
undetected or late-detected intrapartum fetal compromise—a condition
characterized by progressive fetal hypoxia and metabolic acidosis during labor,
resulting from insufficient uteroplacental oxygen delivery. Catching fetal compromise early and accurately is essential because it gives doctors time to intervene—whether through an emergency cesarean section or instrumental delivery. Such timely actions are often the only way to prevent fetal death or permanent neonatal brain damage.

\subsection{Cardiotocography: The Clinical Standard and Its Limitations}

Clinicians virtually everywhere rely on \textbf{Cardiotocography (CTG)} as the standard approach for keeping track of fetal health during labor. It simultaneously records two signals: the Fetal
Heart Rate (FHR), measured by an ultrasound Doppler transducer on the maternal
abdomen, and Uterine Contractions (UC), measured via tocodynamometry. To interpret these readings, doctors look for specific patterns in the FHR, such as baseline rate, short-term and long-term variability (STV and LTV), accelerations, and decelerations. They then analyze these patterns against the timeline of uterine contractions. Guidelines from
the International Federation of Gynecology and Obstetrics (FIGO) classify
CTG traces as Normal, Suspicious, or Pathological \cite{figo2015}.

However, visual CTG analysis is hindered by several limitations:

\begin{enumerate}
    \item \textbf{Poor inter-observer agreement}: Multiple studies demonstrate
    only 60–70\% agreement between experienced obstetric clinicians interpreting
    the same CTG trace, reflecting the inherent subjectivity of visual
    pattern recognition \cite{bernardes1997}.

    \item \textbf{Alert fatigue and high false-positive rates}: Existing
    automated CTG analysis systems generate excessive false alarms, leading
    clinicians to habitually override or dismiss system warnings—a phenomenon
    that undermines system utility and patient safety.

    \item \textbf{Single-signal dependency}: Current commercial and research
    automated systems predominantly analyze the FHR signal in isolation,
    disregarding the clinically essential temporal relationship between FHR
    patterns and uterine contraction timing. Late decelerations—FHR drops
    occurring after the contraction peak—are the cardinal marker of
    uteroplacental insufficiency and can only be identified through joint
    FHR-UC analysis.
\end{enumerate}

\subsection{Limitations of Existing Automated Approaches}

Prior deep learning approaches to automated CTG analysis exhibit several
critical deficiencies that limit their clinical applicability:

\begin{itemize}
    \item \textbf{Unimodal signal analysis}: Most published deep learning
    systems, including early 1D-CNN and ResNet-based approaches \cite{spilka2016,
    zhao2019}, process FHR in isolation, omitting the UC signal and all
    clinical metadata.

    \item \textbf{Absence of uncertainty quantification}: Existing models
    produce deterministic point predictions without confidence estimates.
    In a clinical context, a prediction delivered without a reliability
    measure is inherently dangerous, as it provides no mechanism for the
    clinician to identify ambiguous or out-of-distribution cases requiring
    human review.

    \item \textbf{Reliance on large private datasets}: State-of-the-art
    results, such as those reported by Mendis et al.\ \cite{mendis2023} (AUC
    0.84), were achieved using a private institutional dataset of 9,887
    recordings—far exceeding the size of any publicly available CTG database.
    This places their results beyond reproducibility and limits comparative
    benchmarking within the research community.

    \item \textbf{Deployment inaccessibility}: Published models are typically
    evaluated in laboratory settings and are not optimized for deployment on
    resource-constrained hardware. In low-income clinical settings, inference
    must be possible on commodity hardware without persistent internet
    connectivity.
\end{itemize}

\subsection{Research Problem Definition}

Given the foregoing limitations, this work addresses the following
research problem:

\textit{How can a multi-modal deep learning system integrate FHR time-series,
UC signals, and maternal clinical tabular features to accurately detect
intrapartum fetal compromise from a small public dataset, while providing
clinically meaningful uncertainty estimates, transparent decision explanations,
and remaining deployable on edge hardware—thereby enabling practical use in
resource-limited obstetric settings?}

The specific research objectives are:
\begin{enumerate}
    \item To design a tri-modal fusion architecture that exploits the
    complementary information in FHR signals, UC signals, and structured
    clinical tabular data for fetal compromise classification.
    \item To overcome documented class imbalance (7.25\% pathological) and
    limited dataset size (552 recordings) through a physiologically realistic
    time-series generative augmentation strategy (TimeGAN).
    \item To build a clinically trustworthy system via probabilistic uncertainty
    quantification (MC Dropout, Platt Scaling) and signal-level explainability
    (Grad-CAM).
    \item To achieve state-of-the-art performance on the publicly available
    CTU-UHB benchmark, outperforming results obtained with private datasets,
    thereby establishing a reproducible baseline for the community.
    \item To compress and quantize the trained model to under 2 MB for
    real-time offline inference on mobile devices.
\end{enumerate}

% ============================================================
% SECTION III: LITERATURE SURVEY
% ============================================================

\section{Literature Survey}

Historically, researchers trying to automate CTG analysis have gone through three major phases. Initially, they focused on manually designing features, then moved on to classical machine learning techniques, and most recently, they have embraced deep learning. In this section, we take a look at some of the most impactful studies to show exactly where NeuroFetal AI fits in.

\subsection{Classical Feature-Engineering and Machine Learning Approaches}

Early automated CTG analysis systems relied on expert-defined morphological
features derived from FHR traces. Surveys such as those by Georgoulas et
al.\ \cite{georgoulas2006} and Spilka et al.\ used predefined feature sets such as baseline FHR, STV, LTV, accelerations, and deceleration counts.
These features were then fed to classical classifiers:

\textbf{Support Vector Machines (SVM)} formed the backbone of early competitive
systems, achieving AUC scores in the range 0.72–0.79 on the CTU-UHB database
\cite{spilka2012}. Fergus et al.\ \cite{fergus2013} and Czabanski et
al.\ employed SVM with radial basis function kernels and feature selection to
identify pathological traces, demonstrating that clinical features such as
baseline rate and STV were the most discriminative.

\textbf{Random Forests and Ensemble Methods}: Gradient-boosting and
random forest classifiers applied to hand-crafted FHR features demonstrated
improvements over SVM, reaching AUC values near 0.83 \cite{krupa2011}, and
established that feature ensembling could compensate for the limitations of
individual engineered metrics. The real problem with these classical methods, however, was their limited scope. Because they mostly looked at aggregated statistics, they completely missed the subtle, moment-by-moment temporal structure of the FHR sequence. Even worse, they ignored the critical timing between FHR changes and uterine contractions.

\subsection{Deep Learning Approaches: Single-Modal Signal Analysis}

Convolutional neural networks (CNNs) have facilitated the extraction of temporal patterns directly from raw FHR sequences, reducing reliance on hand-crafted features.

\textbf{1D Convolutional Neural Networks}: Zhao et al.\ \cite{zhao2019}
proposed a 1D-CNN architecture processing raw FHR sequences, demonstrating
superior performance over classical baselines on the CTU-UHB dataset.
However, evaluation was limited to the FHR channel exclusively, ignoring UC.
Building on this, \textbf{Petrozziello et al.\ (2019)} \cite{petrozziello2019}
introduced an input-length invariant deep learning approach for FHR, allowing
rapid detection of fetal compromise without requiring fixed-length windows,
improving potential clinical utility.

Segment selection duration is an important factor in FHR analysis.
Recent studies, such as the cross-database evaluation by \textbf{Lopes et al.\ (2025)}
\cite{lopes2025}, have demonstrated that utilizing extended continuous FHR segments
(e.g., $\geq 30$ minutes immediately preceding delivery) and explicitly preserving signal
gaps yields stronger generalization performance across diverse clinical datasets
compared to aggressively gap-interpolated 10-minute snippets.

\textbf{Residual Networks (ResNet)}: Spilka et al. \cite{spilka2016} and
subsequent work demonstrated that residual connections substantially improved
gradient flow during training of deep 1D temporal networks. Mendis et
al.\ \cite{mendis2023} employed a three-block 1D-ResNet as the signal-analysis
branch of their Fusion ResNet, establishing AUC 0.80 for the FHR-only
configuration—a result they subsequently improved to AUC 0.84 by augmenting
with tabular clinical features.

\textbf{Multi-Scale Convolutional Architectures}: Inspired by
\textit{InceptionNet} \cite{szegedy2015}, several works applied parallel
convolutions with varying kernel sizes (capturing patterns at different
temporal scales) to FHR classification. Furthermore, attention-based models
such as the Multi-Scale CNN (MCNN) \cite{zhao2019} achieved AUC 0.81 on
CTU-UHB, representing the prior deep learning state-of-the-art before
multimodal fusion.

\textbf{Recurrent Architectures}: Long Short-Term Memory (LSTM) networks and
gated recurrent units (GRU) have been explored for FHR sequence modeling
\cite{xue2021}, leveraging their capacity for long-range temporal dependency
modeling. However, their sequential computation renders them significantly
slower than convolutional approaches for deployment, and they did not
consistently outperform well-regularized ResNets on the CTU-UHB benchmark.

\textbf{Emerging Architectures}: Recent advancements have seen the application of
\textbf{Foundation Models} to CTG. For instance, a 2024 study \cite{foundation2024}
proposed a foundation model approach pre-trained on massive unlabeled FHR corpora
and fine-tuned on CTU-UHB for fetal stress prediction during labor, demonstrating
strong zero-shot and few-shot capabilities. Additionally, research expanding beyond
just the second stage of labor has shown that deep learning classifiers can predict
fetal health across \textit{both} stages of labor \cite{bothstages2023}, though
these models often struggle generalizing across diverse hospital protocols.

\subsection{Multimodal and Fusion-Based Approaches}

Recognition that FHR alone is insufficient motivated the development of
multimodal systems.

\textbf{Fusion ResNet (Mendis et al., 2023)} \cite{mendis2023}: This work,
which serves as the direct baseline for NeuroFetal AI, presented the most
relevant prior architecture. It proposed a late-fusion design combining a
three-block 1D-ResNet (processing the last 60 minutes of FHR at 1 Hz) with a
two-layer Dense Network processing five tabular clinical features (Parity,
Maternal Age, Gestation, Baseline Intercept $\beta_0$, and Median Absolute
Deviation of detrended FHR). Fusion was performed via element-wise
multiplication, which the authors demonstrated outperformed concatenation and
addition as fusion operators. The model achieved \textbf{AUC 0.84} on the
public CTU-UHB dataset and was validated externally on their private MHW-pH
dataset (9,887 recordings). Critically, the uterine contraction channel was
discarded due to signal quality constraints, representing a data modality that
NeuroFetal AI explicitly incorporates. The authors provided SHAP-based
tabular feature attribution and Grad-CAM signal localization, establishing
dual-XAI as the interpretability standard for this domain.

\textbf{Christensen et al.} and related work explored clinical risk scoring
combined with CTG signal features, confirming that maternal clinical covariates
(Parity, Age, Gestation) add discriminative value beyond signal morphology alone.

\textbf{Sadeghi et al.\ (2024)} \cite{sadeghi2024} advanced the multimodal paradigm by
developing a deep learning algorithm trained exclusively on the CTU-UHB database
to detect specific morphological FIGO events (e.g., variable decelerations)
rather than just binary fetal outcomes. This approach indicates a shift toward event-specific detection rather than broad risk scoring.

\subsection{Time-Series Data Augmentation for Medical Imbalance}

Class imbalance is an endemic challenge in clinical CTG datasets, where
pathological cases comprise fewer than 10\% of recordings.

\textbf{SMOTE} \cite{chawla2002} applied to flattened FHR feature vectors was
a standard baseline augmentation, though it operates in feature space and
destroys temporal structure—generating synthetic sequences that may not
correspond to physiologically plausible FHR patterns.

\textbf{TimeGAN} (Yoon et al., 2019) \cite{yoon2019} introduced a framework
for realistic time-series synthesis using a GAN architecture with a supervised
loss component preserving stepwise temporal dynamics. Application of TimeGAN
to pathological CTG synthesis has not been widely explored in prior work,
which we investigate in this study. NeuroFetal AI
employs a WGAN-GP variant of TimeGAN that generates physiologically coherent
FHR+UC traces with preserved deceleration timing relative to contractions.

\subsection{Uncertainty Quantification in Clinical AI}

\textbf{MC Dropout (Gal \& Ghahramani, 2016)} \cite{gal2016}: Demonstrated
that Dropout applied at inference time provides a practical Bayesian
approximation, yielding predictive uncertainty through variance across
stochastic forward passes. This has been applied in medical imaging
\cite{kendall2017} but remains underexplored in time-series CTG classification.

\textbf{Calibration}: Platt Scaling \cite{platt1999} and isotonic regression
are established post-hoc calibration techniques that map model output scores to
well-calibrated probabilities. Their application to ensemble CTG classifiers
has not been previously reported in the literature.

\subsection{Explainable AI in Fetal Monitoring}

\textbf{Grad-CAM} \cite{selvaraju2017} has been adapted to 1D convolutional
networks for temporal signal attribution, identifying which time segments most
influenced the classification decision. Mendis et al.\ \cite{mendis2023}
applied Grad-CAM to their ResNet branch to validate that the model correctly
attended to late deceleration patterns. \textbf{SHAP} (SHapley Additive
exPlanations) \cite{lundberg2017} provides model-agnostic tabular feature
attribution and was used in \cite{mendis2023} to explain the contribution of
clinical risk factors such as nulliparity and maternal age.

Recent interpretable architectures, such as \textbf{DeepCTG 1.0} \cite{deepctg2022},
were specifically designed to provide clinicians with transparent feature
attributions for fetal hypoxia detection, reinforcing that high AUC alone is
insufficient for clinical adoption; models must provide physiologically
plausible rationales. Similarly, researchers have begun hybridizing deep learning
with classical signal processing, such as combining Instantaneous Frequency analysis
with \textbf{Common Spatial Patterns (CSP)} \cite{csp2023} for FHR classification,
a method we apply in this study for FHR-UC cross-modal extraction.

\subsection{Edge Deployment of Medical AI}

\textbf{TensorFlow Lite (TFLite)} quantization enables compression of deep
learning models to megabyte scale with minimal accuracy loss, suitable for
deployment on commodity mobile hardware. Its application to deployed
obstetric AI systems remains limited in published literature, with most
systems assuming continuous hospital network connectivity and GPU hardware.

Addressing this constraint, recent literature has explored the integration of
deep learning into mobile clinical interfaces. For example, the
\textbf{AI-based Mobile Partograph (2025)} \cite{eajhs2025} explicitly targets
resource-limited delivery rooms by embedding a predictive CTG framework
(validated on CTU-UHB) directly into a digital labor chart interface. However,
such mobile systems currently lack predictive uncertainty quantification, a
critical requirement for clinical trust.

\subsection{Research Gap and Positioning of NeuroFetal AI}

Table~\ref{tab:sota_compare} summarizes the key limitations of prior work and
the methods used in this study. Three primary gaps are addressed:

\begin{enumerate}
    \item \textit{UC signal omission}: All prior deep learning systems
    discarded the UC channel; NeuroFetal AI incorporates it via Cross-Modal
    Attention to leverage the FHR–contraction temporal relationship.
    \item \textit{No uncertainty quantification}: No prior clinical CTG
    deep learning system provided calibrated uncertainty estimates; NeuroFetal
    AI delivers MC Dropout variance, Predictive Entropy, and Mutual
    Information alongside every prediction.
    \item \textit{Public-only reproducibility}: Mendis et al.'s leading result
    (AUC 0.84) required a private 9,887-case dataset. NeuroFetal AI surpasses
    this on the public CTU-UHB dataset alone (AUC 0.8639), establishing a
    reproducible benchmark.
\end{enumerate}

\begin{table}[htbp]
\caption{Comparative Analysis of CTG Deep Learning Systems}
\label{tab:sota_compare}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{p{2.3cm} p{1.0cm} p{0.8cm} p{0.8cm} p{0.8cm} p{0.8cm}}
\toprule
\textbf{System} & \textbf{AUC} & \textbf{UC} & \textbf{Uncert.} & \textbf{XAI} & \textbf{Edge} \\
\midrule
Spilka et al.\ \cite{spilka2012} (SVM)     & 0.76 & \xmark & \xmark & \xmark & \xmark \\
Zhao et al.\ \cite{zhao2019} (MCNN) & 0.81 & \xmark & \xmark & \xmark & \xmark \\
Mendis et al.\ \cite{mendis2023}    & 0.84 & \xmark & \xmark & \cmark & \xmark \\
Sadeghi et al.\ \cite{sadeghi2024} & N/A* & \cmark & \xmark & \cmark & \xmark \\
\textbf{NeuroFetal AI v5.0 (Ours)} & \textbf{0.8639} & \textbf{\cmark} & \textbf{\cmark} & \textbf{\cmark} & \textbf{\cmark} \\
\bottomrule
\end{tabular}%
}
\end{table}

\textit{(*Note: \cite{sadeghi2024} detects specific events rather than global AUC.)}

\textit{(Note: \cmark\ = present, \xmark\ = absent. The symbols \cmark\ and
\xmark\ require the \texttt{pifont} package: add
\texttt{\textbackslash usepackage\{pifont\}} and define
\texttt{\textbackslash cmark}/\texttt{\textbackslash xmark} accordingly,
or replace with \checkmark and $\times$.)}

% ============================================================
% SECTION IV: DATASET
% ============================================================

\section{Dataset}

\subsection{Source: CTU-UHB Intrapartum Cardiotocography Database}

All experiments in this work were conducted exclusively on the
\textbf{CTU-UHB Intrapartum Cardiotocography Database} \cite{chudacek2014},
a publicly available, fully de-identified research dataset hosted on
\textbf{PhysioNet} \cite{physionet2000}. The database was jointly created by
the Czech Technical University (CTU) and the University Hospital Brno (UHB),
and is the largest publicly available labeled intrapartum CTG dataset with
objective biochemical outcome labels.

\subsubsection{Dataset Characteristics}

\begin{itemize}
    \item \textbf{Total Recordings}: 552 intrapartum CTG recordings from
    singleton pregnancies with a cephalic presentation and a gestational age
    of at least 36 weeks.
    \item \textbf{Signal Channels}: Two channels per recording—Channel 1
    (FHR, beats per minute, bpm) and Channel 2 (UC, arbitrary tocodynamometric
    units).
    \item \textbf{Native Sampling Rate}: 4 Hz (i.e., 4 samples per second,
    corresponding to 240 samples per minute).
    \item \textbf{Recording Duration}: Variable; clinically the last 60 minutes
    of the intrapartum period are most diagnostically relevant and are
    exclusively used in this study.
    \item \textbf{Ground Truth Label}: The binary classification target is
    derived from umbilical cord arterial blood pH measured immediately
    post-delivery:
    \begin{align}
        y =
        \begin{cases}
            1 & \text{(Compromised)}, \quad \text{pH} < 7.05 \\
            0 & \text{(Normal)},      \quad \text{pH} \geq 7.05
        \end{cases}
    \end{align}
    A pH of 7.05 is the established clinical threshold for significant fetal
    metabolic acidemia \cite{figo2015}.
    \item \textbf{Class Distribution}: 40 recordings (7.25\%) are labeled
    Compromised; 512 recordings (92.75\%) are labeled Normal. This constitutes
    a heavily imbalanced binary classification problem.
    \item \textbf{Clinical Metadata}: Each recording is accompanied by structured
    clinical metadata including: maternal age (years), gravidity, parity
    (nulliparous or multiparous), gestational age (weeks), delivery type, and
    Apgar scores at 1 and 5 minutes.
    \item \textbf{Dataset Availability}: Freely accessible at
    \url{https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/}.
    \item \textbf{Ethics}: The CTU-UHB dataset was collected under appropriate
    ethical approval at the originating institution and is made available under
    PhysioNet's data use agreement. No Protected Health Information (PHI) is
    present. No additional institutional review was required for secondary
    analysis.
\end{itemize}

\subsection{Preprocessing Pipeline}

Raw CTG recordings in PhysioNet WFDB format (\texttt{.dat}/\texttt{.hea})
were processed through a standardized, reproducible pipeline implemented in
\texttt{data\_ingestion.py}:

\subsubsection{Step 1: Signal Extraction}
FHR and UC signals were read using the \texttt{wfdb} Python library. The
final 60 minutes (14,400 samples at 4 Hz) of each recording were cropped to
focus processing on the clinically most informative intrapartum segment.

\subsubsection{Step 2: FHR Artifact Correction}
Signal dropout events (consecutive zero-valued samples representing sensor
disconnection) were handled as follows: gaps of fewer than 15 seconds
(i.e., $< 60$ samples at 4 Hz, $< 15$ samples at 1 Hz) were linearly
interpolated. Gaps of 15 seconds or longer were preserved as zero-padded
segments, as interpolation over extended dropouts risks introducing
physiologically implausible artifacts.

\subsubsection{Step 3: UC Signal Cleaning}
Tocodynamometer signals are notoriously susceptible to maternal movement
artefacts and baseline drift. A dedicated UC cleaning pipeline
(\texttt{uc\_cleaning.py}) applied median baseline subtraction and
amplitude normalization to improve contraction detection reliability.

\subsubsection{Step 4: Downsampling}
Both FHR and UC channels were resampled from their native 4 Hz to
\textbf{1 Hz} via linear interpolation, consistent with the approach in
\cite{mendis2023}. This reduces sequence length by 4$\times$ (60 min $\times$
60 s $\times$ 1 Hz $=$ 3,600 samples per recording), lowering computational
cost while preserving diagnostically relevant FHR morphological features.

\subsubsection{Step 5: MinMax Normalization}
FHR and UC signals were independently scaled to the range $[0, 1]$:
\begin{equation}
    x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\end{equation}
Normalization was computed per-recording to remove inter-patient amplitude
variation and was applied prior to windowing.

\subsubsection{Step 6: Sliding Window Segmentation}
Given the 60-minute (3,600 samples at 1 Hz) signal per recording, a sliding
window strategy was employed to expand the effective dataset size:
\begin{itemize}
    \item \textbf{Window Length}: 20 minutes (1,200 samples at 1 Hz).
    \item \textbf{Stride}: 10 minutes (600 samples), yielding a 50\% overlap
    between consecutive windows.
    \item \textbf{Label Inheritance}: Each window inherits the pH-derived binary
    label of its parent recording.
    \item \textbf{Result}: $\sim$5 windows per recording, expanding 552
    recordings to approximately \textbf{2,760 training samples}—a 5$\times$
    effective expansion of the dataset.
\end{itemize}
Window shapes are: FHR window $\in \mathbb{R}^{1200 \times 1}$,
UC window $\in \mathbb{R}^{1200 \times 1}$.

\subsubsection{Step 7: Tabular Feature Extraction}

For each 20-minute window, 16 structured clinical features were extracted
across two categories:

\textbf{Demographic Features (3)}:
\begin{enumerate}
    \item \textit{Maternal Age} (years, continuous) — Risk factor for
    uteroplacental insufficiency.
    \item \textit{Parity} (binary: 0 = Nulliparous, 1 = Multiparous) —
    Nulliparity is strongly associated with prolonged labor.
    \item \textit{Gestational Age} (weeks, continuous) — Determinant of
    fetal maturity and expected FHR baseline range.
\end{enumerate}

\textbf{Signal-Derived Features (13)}: Per-window statistics computed from
the FHR signal:
\begin{enumerate}
    \item \textit{Baseline FHR} (bpm) — Estimated mean FHR, excluding
    accelerations and decelerations.
    \item \textit{Short-Term Variability (STV)} — RMS of beat-to-beat
    differences; a critical marker of fetal neurological integrity.
    \item \textit{Long-Term Variability (LTV)} — Amplitude range of
    oscillatory cycles; reduced LTV is associated with fetal compromise.
    \item \textit{Acceleration Count} — Number of FHR rises $\geq 15$ bpm
    above baseline for $\geq 15$ seconds.
    \item \textit{Deceleration Count} — Number of FHR drops below baseline.
    \item \textit{Late Deceleration Flag} — Binary indicator of decelerations
    temporally lagging contraction peaks (key marker of uteroplacental
    insufficiency).
    \item \textit{Variable Deceleration Count} — Abrupt FHR drops, typically
    cord compression indicators.
    \item \textit{Approximate Entropy (ApEn)} — Non-linear regularity measure
    of FHR signal complexity.
    \item \textit{Sample Entropy (SampEn)} — Conditional probability of
    pattern continuation; complement to ApEn.
    \item \textit{UC Frequency} — Number of detectable contraction events per
    window.
    \item \textit{UC Amplitude} — Average peak-to-trough amplitude of
    contraction waveforms.
    \item \textit{FHR–UC Lag} — Cross-correlation lag at peak FHR-UC
    correlation, distinguishing early, variable, and late deceleration
    patterns.
    \item \textit{Signal Quality Index} — Percentage of non-artefact samples
    in the FHR window.
\end{enumerate}

Missing values in tabular features (arising from recordings with insufficient
signal quality for feature computation) were imputed using column-wise means
computed from the training fold.

\subsubsection{Step 8: Common Spatial Pattern (CSP) Feature Extraction}

A complementary feature representation was derived using \textbf{Common
Spatial Patterns (CSP)}, a technique originally developed for
electroencephalographic (EEG) brain-computer interface (BCI) analysis, here
applied as a novel modality for fetal monitoring. CSP identifies spatial
filters that maximize the ratio of the variance of one class's signal to the
other, applied to the two-channel (FHR, UC) input:

\begin{equation}
    W = \text{CSP}(X_\text{FHR}, X_\text{UC}) \in \mathbb{R}^{d}
\end{equation}

where $W$ comprises variance projections of FHR+UC interactions. This
produces a \textbf{19-dimensional CSP feature vector} per window, providing
a compact representation of the discriminative variance structure in the
FHR–UC interaction that is not captured by either signal alone.
CSP filters were fitted exclusively on the training fold within each
cross-validation iteration to prevent data leakage.

\subsection{Processed Dataset Summary}

\begin{table}[htbp]
\caption{Processed Dataset Specifications for NeuroFetal AI Training}
\label{tab:dataset_spec}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lp{1.6cm}p{3.5cm}}
\toprule
\textbf{Array File} & \textbf{Shape} & \textbf{Description} \\
\midrule
\texttt{X\_fhr.npy}     & $(N, 1200, 1)$ & FHR windows at 1 Hz, 20 min \\
\texttt{X\_uc.npy}      & $(N, 1200, 1)$ & UC windows at 1 Hz, 20 min \\
\texttt{X\_tabular.npy} & $(N, 16)$      & 3 demographic + 13 signal-derived \\
\texttt{X\_csp.npy}     & $(N, 19)$      & CSP variance features \\
\texttt{y.npy}          & $(N,)$         & Binary labels (0=Normal, 1=Comp.) \\
\midrule
\multicolumn{2}{l}{Approx. $N$ (before augmentation)} & $\sim$2,760 \\
\multicolumn{2}{l}{Pathological windows (before aug.)} & $\sim$200 (7.25\%) \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{TimeGAN Data Augmentation Strategy}

The extreme class imbalance (7.25\% pathological) necessitated a robust
augmentation strategy. NeuroFetal AI v4.0 replaced the previously employed
SMOTE with \textbf{TimeGAN}, a time-series generative adversarial network
\cite{yoon2019} implemented as a WGAN-GP architecture:

\begin{itemize}
    \item \textbf{Generator Architecture}: 1D Transposed Convolution network
    trained exclusively on pathological FHR+UC paired windows.
    \item \textbf{Training Signal}: Wasserstein distance with gradient penalty
    (WGAN-GP, $\lambda = 10$) for training stability.
    \item \textbf{Output}: \textbf{1,410 synthetic pathological FHR+UC traces}
    (3$\times$ the original pathological count per fold).
    \item \textbf{Temporal Fidelity}: Unlike SMOTE, TimeGAN preserves
    physiologically meaningful sequential dynamics in generated traces—
    including late deceleration morphology and the temporal delay between
    contraction peaks and FHR nadir.
    \item \textbf{Application Scope}: Augmentation was applied independently
    per training fold inside the cross-validation loop, ensuring no synthetic
    samples from validation or test windows contaminated evaluation.
\end{itemize}

Following TimeGAN augmentation, standard time-series jitter, scaling,
and time-warping transformations (2$\times$ expansion) were applied to the
combined real+synthetic training set for additional regularization.

\subsection{Cross-Validation Protocol}

To ensure robust and unbiased evaluation, all model training and evaluation
was performed under \textbf{Stratified 5-Fold Cross-Validation}:

\begin{itemize}
    \item \textbf{Stratification}: Each fold preserves the $\approx$7.25\%
    pathological class proportion.
    \item \textbf{Evaluation paradigm}: Out-of-Fold (OOF) evaluation—each
    sample is predicted exactly once by a model that did not observe it during
    training, producing an unbiased estimate of generalization performance.
    \item \textbf{Leakage prevention}: All data-dependent transformations
    (normalization statistics, CSP filter training, TimeGAN training, tabular
    imputation values) are computed exclusively from the training portion of
    each fold and applied to the held-out validation portion.
    \item \textbf{No independent held-out test set}: Given the limited dataset
    size (552 recordings), reserving a separate test partition would
    substantially reduce training data. The OOF paradigm provides an
    equivalent statistical guarantee without this sacrifice.
    \item \textbf{Reproducibility}: All stochastic operations were seeded with
    \texttt{random\_state=42}.
\end{itemize}

% ============================================================
% SECTION V: PROPOSED ARCHITECTURE
% ============================================================

\section{Proposed Architecture: AttentionFusionResNet}

NeuroFetal AI (v5.0) introduces a tri-modal deep learning architecture, designated as \textbf{AttentionFusionResNet}. Unlike conventional CTG classifiers that rely exclusively on the FHR continuous waveform, this architecture combines three distinct clinical representations: 
(1) Raw 1D Temporal FHR Signals, 
(2) Static Maternal Clinical Tabular Data, and 
(3) FHR-UC Spatial Pattern Features (CSP). 

The network is structurally composed of three independent feature extraction branches that converge at a \\textit{Cross-Modal Attention Fusion (CMAF)} module, followed by a probabilistic classification head with Monte Carlo (MC) Dropout for epistemic uncertainty quantification.

\subsection{FHR Temporal Encoder Branch}
The primary signal branch processes the 20-minute FHR sequence $\mathbf{X}_{\text{FHR}} \in \mathbb{R}^{1200 \times 1}$. The backbone is a 1D Residual Network (ResNet) enhanced with Squeeze-and-Excitation (SE) blocks \cite{hu2018} and Multi-Head Self-Attention. 

The encoder begins with a large receptive field convolution (kernel size 7, stride 2) and max pooling to rapidly downsample the uninformative high-frequency noise. This is followed by three stages of cascading residual blocks. Let $\mathbf{x}_l$ denote the input to the $l$-th residual block. The block operation is defined as:
\begin{equation}
    \mathbf{h}_l = \sigma(\text{BN}(\mathbf{W}_{l,2} * \sigma(\text{BN}(\mathbf{W}_{l,1} * \mathbf{x}_l))))
\end{equation}
\begin{equation}
    \mathbf{x}_{l+1} = \text{ReLU}(\mathbf{x}_l + \text{SE}(\mathbf{h}_l))
\end{equation}
where $*$ denotes 1D convolution, BN is Batch Normalization, and $\text{SE}(\cdot)$ represents the channel-wise attention recalibration. 

To mitigate overfitting on the small CTU-UHB dataset, we implement \textbf{Stochastic Depth} (Spatial Dropout 1D) within the residual paths, randomly dropping entire representation channels during training with a linearly increasing rate ($p \in [0.05, 0.15]$). Following the residual stages, Temporal Multi-Head Attention extracts long-range dependencies (e.g., relating a baseline shift at minute 2 to a late deceleration at minute 18). Finally, Global Average Pooling (GAP) reduces the temporal sequence to a dense vector $\mathbf{v}_{\text{FHR}} \in \mathbb{R}^{192}$.

\subsection{Clinical Context and CSP Branches}
Medical decision-making is heavily context-dependent; a moderate deceleration is interpreted differently in a 40-week nulliparous woman compared to a 36-week multiparous woman. 

The 16-dimensional tabular vector $\mathbf{X}_{\text{tab}}$ (comprising maternal demographics and extracted signal statistics such as Signal-to-Noise ratio and UC frequency) is processed through a two-layer Multi-Layer Perceptron (MLP) with ReLU activations and heavy dropout ($p=0.3$) to yield the clinical embedding $\mathbf{v}_{\text{tab}} \in \mathbb{R}^{128}$. 

Concurrently, the 19-dimensional Common Spatial Pattern vector $\mathbf{X}_{\text{CSP}}$, which encapsulates the discriminative variance between the FHR and UC contraction cycles, is processed through an identical, parallel MLP structure to yield $\mathbf{v}_{\text{CSP}} \in \mathbb{R}^{128}$.

\subsection{Cross-Modal Attention Fusion (CMAF)}
Existing late-fusion architectures often naively concatenate or multiply modal embeddings \cite{mendis2023}. We propose a dynamic \textbf{Cross-Modal Attention Fusion (CMAF)} layer where the FHR temporal embedding attends to the FHR-UC spatial patterns, explicitly gated by the maternal clinical context.

The CMAF mechanism projects the embeddings into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) spaces:
\begin{align}
    \mathbf{Q} &= \mathbf{W}_Q \mathbf{v}_{\text{FHR}} \\
    \mathbf{K} &= \mathbf{W}_K \mathbf{v}_{\text{CSP}} \\
    \mathbf{V} &= \mathbf{W}_V \mathbf{v}_{\text{CSP}}
\end{align}

Cross-modal attention scores are computed via scaled dot-product:
\begin{equation}
    \mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
\end{equation}

The attention output is gated by a learned, sigmoid-activated projection of the clinical tabular data:
\begin{equation}
    \mathbf{G}_{\text{clinical}} = \sigma(\mathbf{W}_G \mathbf{v}_{\text{tab}})
\end{equation}
\begin{equation}
    \mathbf{v}_{\text{fusion}} = \text{LayerNorm}(\mathbf{v}_{\text{FHR}} + \text{Dropout}(\mathbf{A} \odot \mathbf{G}_{\text{clinical}}))
\end{equation}
This gating mechanism enables the network to actively modulate how much weight it assigns to spatial contraction patterns based on the patient's specific clinical risk factors.

\subsection{Classification and Uncertainty Head}
The fused embedding $\mathbf{v}_{\text{fusion}}$ is passed to a classification head comprising two dense layers. To estimate epistemic (model) uncertainty, we apply \textbf{Monte Carlo (MC) Dropout} \cite{gal2016}. 

Unlike standard dropout which is disabled during inference, target dropout layers ($p=0.3$) remain active. For clinical prediction, the system performs $T=20$ stochastic forward passes for a single patient trace. The final predictive probability $\hat{p}$ is the mean of these passes, while the predictive variance $\sigma^2$ provides a measure of diagnostic uncertainty, flagging cases that require immediate human obstetrician review.

% ============================================================
% SECTION VI: EXPERIMENTAL SETUP \& RESULTS
% (This section has been moved to result.md for now)
% \input{result.md}
% ============================================================

% ============================================================
% REFERENCES  (to be compiled with BibTeX / replace with thebibliography)
% ============================================================

\begin{thebibliography}{99}

\bibitem{who2020stillbirth}
World Health Organization, ``Stillbirths,'' \textit{WHO Fact Sheets}, 2020.
[Online]. Available: \url{https://www.who.int/news-room/fact-sheets/detail/stillbirth}

\bibitem{figo2015}
A.~Ayres-de-Campos, C.~Spong, and C.~Chandraharan, ``FIGO consensus guidelines on
intrapartum fetal monitoring: Cardiotocography,''
\textit{Int. J. Gynaecol. Obstet.}, vol. 131, no. 1, pp. 13--24, 2015.

\bibitem{bernardes1997}
J.~Bernardes, A.~Costa-Pereira, D.~Ayres-de-Campos, H.~P. van Geijn, and
L.~Pereira-Leite, ``Evaluation of interobserver agreement of cardiotocograms,''
\textit{Int. J. Gynaecol. Obstet.}, vol. 57, no. 1, pp. 33--37, 1997.

\bibitem{mendis2023}
B.~Mendis, D.~A.~Medagoda, C.~L.~Mendis, W.~Perera, and J.~D.~Amarasinghe,
``Fusing tabular features and deep learning for fetal heart rate analysis:
A clinically interpretable model for fetal compromise detection,''
\textit{IEEE Access}, 2023, doi: 10.1109/ACCESS.2023.XXXXXXX.

\bibitem{chudacek2014}
V.~Chudáček, J.~Spilka, M.~Burša, P.~Janků, L.~Hruban, M.~Huptych,
and L.~Lhotská, ``Open access intrapartum CTG database,''
\textit{BMC Pregnancy Childbirth}, vol. 14, no. 1, p. 16, 2014,
doi: 10.1186/1471-2393-14-16.

\bibitem{physionet2000}
A.~L. Goldberger \textit{et al.}, ``PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals,''
\textit{Circulation}, vol. 101, no. 23, pp. e215--e220, 2000.

\bibitem{spilka2012}
J.~Spilka, V.~Chudáček, M.~Koucky, L.~Lhotská, M.~Huptych, P.~Janků,
G.~Georgoulas, and C.~Stylios, ``Using nonlinear features for fetal
distress classification,''
\textit{Biomed. Signal Process. Control}, vol. 7, no. 4, pp. 393--401, 2012.

\bibitem{spilka2016}
J.~Spilka \textit{et al.}, ``Intrapartum fetal heart rate classification:
A deep convolutional neural network approach,''
in \textit{Proc. IEEE EMBC}, 2016, pp. 4570--4573.

\bibitem{zhao2019}
Z.~Zhao, H.~Zhang, and R.~Fu, ``Multi-scale convolutional neural network for
fetal heart rate state classification,''
\textit{Comput. Methods Programs Biomed.}, vol. 176, pp. 251--262, 2019.

\bibitem{georgoulas2006}
G.~Georgoulas, D.~Gavrilis, I.~G. Tsoulos, C.~Stylios, J.~Bernardes,
and P.~Groumpos, ``Novel approach for fetal heart rate classification
introducing grammatical evolution,''
\textit{Biomed. Signal Process. Control}, vol. 1, no. 1, pp. 56--60, 2006.

\bibitem{fergus2013}
P.~Fergus, D.~Cheung, A.~Hussain, D.~Al-Jumeily, C.~Dobbins, and S.~Iram,
``Prediction of intrapartum hypoxia from cardiotocography data using
machine learning,''
in \textit{Proc. AISC}, 2013, pp. 369--376.

\bibitem{krupa2011}
B.~N. Krupa, M.~A. M. Ali, and E.~Zahedi,
``The application of empirical mode decomposition for the enhancement of
cardiotocograph signals,''
\textit{Physiol. Meas.}, vol. 32, no. 8, p. 1381, 2011.

\bibitem{szegedy2015}
C.~Szegedy \textit{et al.}, ``Going deeper with convolutions,''
in \textit{Proc. IEEE CVPR}, 2015, pp. 1--9.

\bibitem{xue2021}
M.~Xue, C.~Luo, and T.~Zhu, ``Fetal health state assessment using LSTM
and multiscale analysis,'' \textit{IEEE J. Biomed. Health Inform.},
vol. 25, no. 5, pp. 1607--1616, 2021.

\bibitem{yoon2019}
J.~Yoon, D.~Jarrett, and M.~van~der~Schaar,
``Time-series generative adversarial networks,''
in \textit{Proc. NeurIPS}, 2019, pp. 5508--5518.

\bibitem{chawla2002}
N.~V. Chawla, K.~W. Bowyer, L.~O. Hall, and W.~P. Kegelmeyer,
``SMOTE: Synthetic minority over-sampling technique,''
\textit{J. Artif. Intell. Res.}, vol. 16, pp. 321--357, 2002.

\bibitem{gal2016}
Y.~Gal and Z.~Ghahramani, ``Dropout as a Bayesian approximation:
Representing model uncertainty in deep learning,''
in \textit{Proc. ICML}, 2016, pp. 1050--1059.

\bibitem{kendall2017}
A.~Kendall and Y.~Gal, ``What uncertainties do we need in Bayesian deep
learning for computer vision?,''
in \textit{Proc. NeurIPS}, 2017, pp. 5574--5584.

\bibitem{platt1999}
J.~Platt, ``Probabilistic outputs for support vector machines and comparisons
to regularized likelihood methods,''
\textit{Adv. Large Margin Classifiers}, vol. 10, no. 3, pp. 61--74, 1999.

\bibitem{selvaraju2017}
R.~R. Selvaraju, M.~Cogswell, A.~Das, R.~Vedantam, D.~Parikh, and D.~Batra,
``Grad-CAM: Visual explanations from deep networks via gradient-based
localization,''
in \textit{Proc. IEEE ICCV}, 2017, pp. 618--626.

\bibitem{lundberg2017}
S.~M. Lundberg and S.-I. Lee,
``A unified approach to interpreting model predictions,''
in \textit{Proc. NeurIPS}, 2017, pp. 4765--4774.

\bibitem{lin2017}
T.-Y. Lin, P.~Goyal, R.~Girshick, K.~He, and P.~Dollár,
``Focal loss for dense object detection,''
in \textit{Proc. IEEE ICCV}, 2017, pp. 2980--2988.

\bibitem{hu2018}
J.~Hu, L.~Shen, and G.~Sun, ``Squeeze-and-excitation networks,''
in \textit{Proc. IEEE CVPR}, 2018, pp. 7132--7141.

\bibitem{lopes2025}
R.~Lopes \textit{et al.}, ``Cross-Database Evaluation of Deep Learning Methods
for Intrapartum Cardiotocography Classification,'' \textit{IEEE}, 2025.

\bibitem{sadeghi2024}
R.~Sadeghi \textit{et al.}, ``Multimodal Deep Learning-based Algorithm for
Specific Fetal Heart Rate Event Detection,'' \textit{ResearchGate}, 2024.

\bibitem{eajhs2025}
``The AI-based Mobile Partograph: A Deep Learning Approach for Automated
Fetal Distress Prediction,'' \textit{East African Journal of Health and
Science}, 2025.

\bibitem{petrozziello2019}
A.~Petrozziello \textit{et al.}, ``Rapid detection of fetal compromise using
input length invariant deep learning on fetal heart rate signals,''
\textit{IEEE}, 2019.

\bibitem{foundation2024}
``A Foundation Model Approach for Fetal Stress Prediction During Labor,''
\textit{arXiv preprint}, 2024.

\bibitem{deepctg2022}
``DeepCTG 1.0: an interpretable model to detect fetal hypoxia,''
\textit{Frontiers in Pediatrics}, 2023.

\bibitem{bothstages2023}
``Fetal Health Classification from Cardiotocograph for Both Stages of Labor,''
\textit{MDPI Diagnostics}, 2023.

\bibitem{csp2023}
``Fetal Hypoxia Classification from Cardiotocography Signals Using
Instantaneous Frequency and Common Spatial Pattern,''
\textit{MDPI Sensors}, 2023.

\end{thebibliography}

\end{document}

% ============================================================
% END OF FILE
% ============================================================
%
% COMPILATION NOTES FOR OVERLEAF:
% -------------------------------------------------------
% 1. Compiler  : pdfLaTeX (default on Overleaf)
% 2. Class     : IEEEtran (built-in on Overleaf — no extra install needed)
% 3. Bibliography: This file uses a manual thebibliography environment.
%    To switch to BibTeX, create a .bib file and replace the thebibliography
%    block with \bibliographystyle{IEEEtran} + \bibliography{references}
%
% 4. For final submission, remove the compilation notes at the end if desired.
%








