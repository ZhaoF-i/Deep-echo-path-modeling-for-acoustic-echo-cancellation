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
<img src="https://github.com/ZhaoF-i/Deep-echo-path-modeling-for-acoustic-echo-cancellation/tree/main/pictures/ICCRN.png" alt="LAEC" width="660" height="300">
</div>

