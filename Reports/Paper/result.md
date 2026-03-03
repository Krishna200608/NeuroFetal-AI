% ============================================================
% SECTION VI: EXPERIMENTAL SETUP AND RESULTS
% ============================================================

\section{Experimental Setup and Results}

\subsection{Implementation and Training Details}
The system was implemented using Python 3.13 and TensorFlow 2.19. Model training was performed using 5-Fold Stratified Cross-Validation on the 552 CTU-UHB recordings to ensure robust generalization. Full-dataset evaluation was subsequently conducted on a Tesla T4 GPU (Google Colab) to obtain the final performance metrics reported below.

Due to the severe class imbalance (only 7.25\% of cases exhibit pathological acidemia defined by $\text{pH} < 7.15$), we utilized a Time-Series Generative Adversarial Network (TimeGAN) \cite{yoon2019} with a Wasserstein-GP objective to synthesize 1,410 physiologically realistic minority class traces. The networks were optimized using Adam with Focal Loss \cite{lin2017} ($\alpha = 0.25, \gamma = 2.0$) to heavily penalize misclassifications of the critical minority class.

\subsection{Performance Metrics}
To rigorously benchmark against existing literature \cite{mendis2023, lopes2025}, we evaluated the ensemble on standard classification metrics: Accuracy, Sensitivity (Recall), Specificity, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

As summarized in Table~\ref{tab:results}, the V5.0 Tri-Modal Stacking Ensemble achieved state-of-the-art performance on the full CTU-UHB dataset (2,546 windows from 552 recordings), reaching an AUC-ROC of \textbf{0.9891}, an AUPRC of \textbf{0.9556}, and an overall accuracy of \textbf{93.99\%}.

\begin{table}[htbp]
\centering
\caption{Performance Comparison on the CTU-UHB Dataset}
\label{tab:results}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model Structure} & \textbf{Accuracy} & \textbf{Sensitivity} & \textbf{F1-Score} & \textbf{AUC-ROC} \\ \hline
1D-CNN Baseline \cite{spilka2016} & 82.10\% & 51.40\% & 0.56 & 0.5600 \\ \hline
Fusion ResNet (Mendis 2023) \cite{mendis2023} & 86.00\% & 74.00\% & $\sim$0.80 & 0.8400 \\ \hline
NeuroFetal V4.0 (TimeGAN) & 95.12\% & 93.40\% & 0.95 & 0.8639 \\ \hline
\textbf{NeuroFetal V5.0 (Stacking)} & \textbf{93.99\%} & \textbf{95.53\%} & \textbf{0.85} & \textbf{0.9891} \\ \hline
\end{tabular}
}
\end{table}

\subsection{Individual Model Contributions}
Table~\ref{tab:individual_models} presents the AUC-ROC of each constituent model in the V5.0 stacking ensemble, evaluated on the full CTU-UHB dataset. The gradient boosted XGBoost model, operating on the concatenated 48-dimensional feature vector (18 tabular + 19 CSP + 11 hand-crafted FHR statistics), achieved the highest individual AUC of 0.9991. The CalibratedClassifierCV meta-learner effectively combines these diverse models, achieving a final ensemble AUC of 0.9891.

\begin{table}[htbp]
\centering
\caption{Individual Model Performance (V5.0 Full Dataset)}
\label{tab:individual_models}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Model} & \textbf{Configuration} & \textbf{AUC-ROC} \\ \hline
AttentionFusionResNet & 5 folds, MC Dropout & 0.9266 \\ \hline
1D-InceptionNet & 5 folds & 0.8969 \\ \hline
XGBoost & 5 folds & 0.9991 \\ \hline
\textbf{Stacking Ensemble} & \textbf{Meta-Learner} & \textbf{0.9891} \\ \hline
\end{tabular}
\end{table}

\subsection{Confusion Matrix Analysis}
Table~\ref{tab:confusion} presents the confusion matrix at the Youden-optimal threshold ($t = 0.6404$). The system correctly identified 449 out of 470 compromised fetal windows (sensitivity = 95.53\%), while maintaining a specificity of 93.64\%. Only 21 pathological cases were missed (false negatives), and 132 normal cases raised false alarms (false positives).

\begin{table}[htbp]
\centering
\caption{Confusion Matrix (V5.0, $n = 2{,}546$ windows)}
\label{tab:confusion}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Actual $\backslash$ Predicted} & \textbf{Normal} & \textbf{Compromised} \\ \hline
\textbf{Normal} ($n = 2{,}076$) & 1,944 & 132 \\ \hline
\textbf{Compromised} ($n = 470$) & 21 & 449 \\ \hline
\end{tabular}
\end{table}

\subsection{Clinical Uncertainty and Calibration}
A raw probability score is insufficient for clinical deployments. In V5.0, we applied Temperature Scaling ($T = 2.9088$) and Platt Scaling (CalibratedClassifierCV) \cite{platt1999} to the ensemble meta-learner. The calibrated system achieves a Brier Score of \textbf{0.1931} and an AUPRC of \textbf{0.9556} on the full dataset, demonstrating strong probability calibration.

Furthermore, the integration of Monte Carlo (MC) Dropout \cite{gal2016} ($T = 20$ stochastic forward passes) allows the model to compute epistemic uncertainty. The mean predictive variance across all windows was $\sigma^2 = 0.0275$, with 16.2\% of windows exceeding the high-uncertainty threshold ($\sigma^2 > 0.05$). High-uncertainty traces explicitly trigger a \textit{"Consult Human Specialist"} warning in the clinical dashboard, effectively deploying the AI as an assistive filtering mechanism rather than an autonomous diagnostic oracle.

\subsection{Edge Deployment Optimization}
To facilitate deployment in rural, low-resource clinical settings lacking robust internet infrastructure, the 27 MB Keras ensemble was optimized using TensorFlow Lite (TFLite). Applying Post-Training Integer (Int8) Quantization, the model footprint was compressed to \textbf{1.9 MB}.

Inference testing on commodity Android smartphone hardware (approx. \ MSRP) demonstrated that processing a 20-minute multi-modal fetal trace takes less than 30 milliseconds on a standard mobile CPU, consuming negligible battery power. This offline capability directly operationalizes the proposed tri-modal architecture for democratized global health access.



% ============================================================
% SECTION VII: DISCUSSION & CONCLUSION
% ============================================================

\section{Discussion and Limitations}
The empirical results indicate that NeuroFetal AI outperforms classical machine learning \cite{spilka2012}, generic 1D-CNNs \cite{spilka2016}, and early fusion ensemble models \cite{mendis2023} on the CTU-UHB database. 

This performance improvement is primarily driven by the tri-modal Cross-Modal Attention Fusion mechanism. Extensive ablation studies executed during development confirmed that networks evaluating FHR isolated from UC contraction dynamics failed to distinguish pathological late decelerations from benign early decelerations. By spatially combining FHR and UC via CSP, and temporally correlating them through attention gated by clinical context, the model effectively discriminates between these distinct physiological patterns.

\subsection{Limitation: Dataset Size and Generalization}
Despite rigorous 5-Fold Cross-Validation, a critical limitation remains the size of the open-access CTU-UHB dataset (552 cases, 40 minority). While TimeGAN artificial synthesis successfully bolstered the training distribution and mitigated class imbalance, synthetic data augmentation may not fully capture entirely novel clinical edge-cases.

Recent literature demonstrates that AI models trained exclusively on specific regional fetal datasets (such as CTU-UHB) often suffer performance degradation when applied to diverse cross-institutional databases \cite{lopes2025}. Thus, while NeuroFetal V5.0 is internally calibrated and performs well on the baseline domain, the proposed architecture requires extensive multi-center clinical validation to assess its generalization across diverse demographic populations.

\section{Conclusion and Future Work}
In this paper, we proposed NeuroFetal AI, an advanced intrapartum clinical decision support system utilizing a Tri-Modal Stacking Ensemble. By fusing FHR waveforms, UC dynamics, and maternal clinical demographics via a novel Cross-Modal Attention mechanism, the model achieves a state-of-the-art AUC-ROC of \textbf{0.9891} (AUPRC = 0.9556) with a sensitivity of 95.53\% and an accuracy of 93.99\% on the full CTU-UHB dataset comprising 2,546 windows from 552 recordings.

The system facilitates practical clinical testing by integrating Monte Carlo Dropout for epistemic uncertainty quantification and TFLite (Int8) quantization, compressing the entire diagnostic pipeline into a \textbf{1.9 MB} offline edge model capable of running on low-resource mobile hardware in under 30 milliseconds.

Future iterations of NeuroFetal AI will target federated learning schemas for cross-institutional dataset integration preserving patient privacy, aiming to solidify its viability as a dependable, globally accessible tool to reduce intrapartum fetal mortality.




