# Deep echo path modeling for acoustic echo cancellation

## Announcement

This GitHub account was created to comply with INTERSPEECH's double-blind regulations, and it will be relocated to a different address once the acceptance results are published.

## PROPOSED METHOD

### Problem Formulation

<div align="center">
<img src="https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation/tree/main/pictures/LAEC_2.png" alt="LAEC" width="660" height="300">
</div>

The diagram of single channel Acoustic echo cancellation (AEC) system is illustrated in the picture above. The microphone signal $d(n)$ is a mixture of echo signal $y(n)$ and near-end signal $s(n)$. If the environmental noise is not considered, the microphone signal in the time domain can be formulated as follows:

$$
    d(n) = y(n) + s(n) \tag{1}
$$

The acoustic echo y(n) is the convolution of the source signal x(n) with the room impulse response (RIR) \cite{habets2006room} h(n). 
We reformulate Eq.(1) into the T-F domain by applying the short-time Fourier transform (STFT) as:

$$
    D[t, f] = S[t, f] + \sum_k^K H[k,f]X[t-k, f] \tag{2}
$$

where $S[t, f]$, $D[t, f]$ and $X[t, f]$ represent the near-end signal, microphone signal, and far-end signal at the frame $t$ and frequency $f$, respectively, and $H[k, f]$ is the echo path. Here, $K$ stands for the number of blocks.

### Details of the method

 We propose a new perspective for AEC. Specifically, we introduce a deep learning-based approach for modeling echo paths in the time-frequency (T-F) domain. Consequently, the neural network employed in our proposed method can accommodate larger inputs, outputs, and network sizes relative to those used in hybrid methods. This unique feature empowers our method to more effectively capitalize on the inherent advantages of neural networks.

 The key difference between the echo path predicted by the network and the finite impulse response filter (echo path) obtained from the PFDKF method is that the network predictions have an additional time dimension to capture the echo path for each frame, with shapes [**c**hannels, **t**ime, **f**requency, bloc**k**]. Channels are equal to 2 to indicate the real and imaginary parts. The strategy formula is as follows:

$$
    \hat{S}[t, f] = D[t, f] - \sum_k^K \hat{H}[k,t,f] * X[t-k, f] \tag{3}
$$

where $\hat{H}$ represents the predicted echo path, while \(*\) denotes the complex multiplication operation.

To obtain a more accurate estimate of the acoustic echo path, we first input the reference signal from the far-end speaker during periods of single-talk into a neural network, which generates an initial approximation of the true echo path impulse response. We then train the network under double-talk conditions, using the predicted echo path from the previous step as an additional regularization constraint. The simplified mathematical formulation for the echo path prediction is as follows:

$$
    \hat{S_1}, \hat{H_1} = \mathscr{F}_1(Y, X; \Phi_1)  \tag{4}
$$

$$
    \hat{S_2}, \hat{H_2} = \mathscr{F}_2(D, X; \Phi_2)  \tag{5}
$$

where $\hat{S}_1$ and $\hat{H}_1$ represent the predicted near-end signal and echo path, respectively, in the far-end single-talk scenario, while $\hat{S}_2$ and $\hat{H}_2$ are their counterparts in the double-talk scenario. The predictors $\mathscr{F_1}$ and $\mathscr{F_2}$ are based on the ICCRN model. Although $\mathscr{F_1}$ and $\mathscr{F_2}$ share the same network architecture, they have distinct parameters, denoted by $\Phi_1$ and $\Phi_2$, respectively. X, Y, and D represent the far-end signal, echo signal, and microphone signal in the T-F domain.

### ICCRN

<div align="center">
<img src="https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation/tree/main/pictures/ICCRN.png" alt="ICCRN">
</div>

The diagram of ICCRN is shown in the figure above. As an in-place model, the ICCRN does not involve any frequency downsampling or upsampling operations, reducing information loss and eliminating the introduction of irrelevant information. Consequently, the frequency dimension of the model remains unchanged at f = 160, with all convolution layers having the same number of output channels (c = 20). To further analyze the time-frequency (T-F) domain features, cepstrum analysis is applied. Cepstrum analysis involves performing a real-valued fast Fourier transform on each channel's T-F domain feature map for every frame. In the resulting cepstral space, the spectral envelope energy, containing most semantic information and timbre cues, concentrates in the low cepstral region, while harmonics appear as sparse peaks distributed periodically across higher cepstral bands \cite{skowronski2015cepstral}. The cepstrum analysis capability enables the ICCRN to examine the relationship between the far-end reference signal and the echo signal present in the microphone input from a harmonic structure viewpoint. This harmonic structural perspective significantly simplifies the process of echo path modeling.

### Loss function
For the predictor $\mathscr{F}_1$, which exclusively deals with far-end single-talk scenarios, the loss is computed using mean squared error (MSE) on the predicted signal, as follows:

$$
    \mathcal{L}_{\mathscr{F}_1}=\frac{1}{TF} \sum_t^T \sum_f^F |\hat{S}_1(t, f) - {S}_1(t, f)|^2 
$$


For the double-talk scenario of $\mathscr{F}_2$, the corresponding loss function consists of multiple items.
Stretched Scale-Invariant Signal-to-Noise Ratio (S-SISNR) is a modified version of the Scale-Invariant Signal-to-Noise Ratio (SISNR) loss function. S-SISNR is a time domain loss function that is obtained by doubling the period of SISNR. The simplified formula for S-SISNR is expressed as follows:

$$
 \mathcal{L}_{s-sisnr}=10 \log _{10} cot^2 (\frac{\alpha}{2}) =  10 \log _{10} \frac{1+cos(\alpha)}{1-cos(\alpha)}
$$

where $\alpha$ represents the angle between two vectors' ideal near-end signal $s$ and predicted near-end signal $\hat{s}$, since it is complicated to calculate the half angle, after the derivation of the trigonometric function, it can be represented by $cos(\alpha)$.

The “RI+Mag” loss criterion is adopted to recover the complex spectrum as follows:

$$
\mathcal{L}_{\mathrm{mag}}=\frac{1}{TF} \sum_t^T \sum_f^F |S(t, f)|^p-|\hat{S}(t, f)|^p|^2 
$$

$$
\mathcal{L}_{\mathrm{RI}}=\frac{1}{TF} \sum _t^T \sum _f^F |S(t, f)|^p e^{j \theta _{S(t, f)}}-|\hat{S}(t, f)|^p e^{j \theta _{\hat{S}(t, f)}}|^2
$$

where $p$ is a spectral compression factor (set to 0.5).
Operator $\theta$ calculates the phase of a complex number.

Additionally, the predicted echo path is directly constrained by calculating the MSE between $\hat{H}_1$ and $\hat{H}_2$, which we call the “rir” loss criterion, as follows:

$$
\mathcal{L}_{\mathrm{rir}}=\frac{1}{TFK} \sum_t^T \sum_f^F \sum_k^K |\hat{H}_1(t, f, k)|-|\hat{H}_2(t, f, k)||^2 
$$

$$
\mathcal{L}_{\mathscr{F}_2}=\mathcal{L} _{\mathrm{RI}}+\mathcal{L} _{ {mag }}+\mathcal{L} _{ {s-sisnr}} + \mathcal{L} _{ {rir}}
$$

where $\mathcal{L}_{\mathscr{F}_2}$ denotes the loss function of predictor ${\mathscr{F}_2}$.

## Result

|           |           |          |       |           |           |
|:---------:|:---------:|:--------:|:-----:|:---------:|:---------:|
|      near-end(speech), far-end(speech)     | ERLE      | PESQ     | SDR   | MACs      | Param     |
| mix       |        -- | 2.16     | 4.78  |        -- |        -- |
| PNLMS     | 11.91     | 2.38     | 5.62  |        -- |        -- |
| PFDKF     | 14.6      | 3.02     | 17.31 |        -- |        -- |
| NKF       | 22.39     | 3.16     | 16.42 | 0.28      | 5.3       |
| ICCRN-E2E | **33.4**      | **3.46**     | 18.34 | 0.84      | 121.36    |
| proposed  | 29.65     | 3.38     | **19.85** | 0.86      | 120.26    |

This section compares the performance of our proposed method against several baselines. As shown in table above, our method outperforms all other methods in terms of SDR. The superior SDR achieved by our proposed method demonstrates its effectiveness in preserving the desired speech signal while mitigating distortions.
Additionally, Our proposed method achieves the second-highest scores for ERLE and PESQ. The ERLE score reflects the method's capability to attenuate acoustic echoes, while the PESQ score measures the quality of the processed speech signal. The competitive performance in these metrics demonstrates our method's effectiveness in suppressing echoes while preserving speech quality, which is crucial for practical speech processing applications. Importantly, with its 121.3K parameters and 0.86G MACs, our system is aptly designed for real-time applications.

|                                  |           |          |       |
|:--------------------------------:|:---------:|:--------:|:-----:|
| near-end(music), far-end(speech) | ERLE      | PESQ     | SDR   |
| mix                              |        -- | 1.43     | 3.36  |
| PNLMS                            | 10.52     | 2.13     | 6.58  |
| PFDKF                            | 12.82     | 2.52     | 13.11 |
| NKF                              | 11.98     | 2.44     | 11.67 |
| ICCRN-E2E                        | **26.98**     | 2.76     | 12.41 |
| proposed                         | 26.96     | **2.85**     | **15.96** |

## Samples


The analysis of the AEC performance on unmatched test sets, as presented in table above, underscores the proposed method's superior generalization capabilities. The method achieves the highest scores in both SDR and PESQ, underscoring its effectiveness in preserving the integrity of the near-end music signal and ensuring a natural listening experience. The marginally lower ERLE score does not detract from the overall audio quality, suggesting a well-balanced approach to echo cancellation. These results indicate that the proposed AEC method is highly adaptable and robust, making it well-suited for real-world applications where acoustic conditions are varied and unpredictable. The method's performance highlights its potential for providing high-quality echo cancellation in diverse acoustic environments.

|              | near-end(speech), far-end(speech) |  near-end(speech), far-end(speech)  | near-end(music), far-end(speech)   |near-end(music), far-end(speech)   |
|:-------------:|:---------:|:--------:|:-----:|:-----:|
| mix           | <img src="https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation/tree/main/pictures/LAEC_2.png"> | 1.43     | 3.36  | 3.36  |
