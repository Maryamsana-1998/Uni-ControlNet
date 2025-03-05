
# Ultra Low Bandwidth Video Compression with Improved Realism 

Developed a novel video compression pipeline using controllable diffusion models that achieve compression at ultra-low bitrates while maintaining high video quality.

A controllable diffusion-based framework can provide precise adjustments to the compression-quality trade-off, enabling efficient, high-quality video streaming in bandwidth-constrained environments.

This work is based on:

Official implementation of Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models, which is accepted by NeurIPS 2023.

## Methadology

We Replace the conventional video decoder with a latent diffusion model to refine coarse decoded frames, achieving higher realism at ultra-low bitrates.

We have two stage conditioning for our video decoder: 
- **Global Adapter:** Extracts global semantics as described below:
 
  c_global = $g_ϕ$($y$, $x_{t-1}$)
 
  where `y` is the caption and $x_{t-1}$ is the previous decoded frame.

- **Local Adapter:** Ensures temporal consistency and motion coherence:

  c_local = $h_ψ$($f_{t-1→t}$, $x_{t-1}$)

  where $f_{t-1→t}$ is the optical flow from frame `t-1` to frame `t`.


- **Dataset Expansion:** We extend the Vimeo dataset by extracting forward optical flow between consecutive frames and LIC-encoded frames, enabling better temporal consistency and motion-aware training for our model.

- **Perceptual Enhancement with Combined Loss:** To ensure fidelity, perceptual quality, and efficient latent-space representation in reconstructed frames.
The multi-term objective function is defined as:

The multi-term objective function is defined as:

```math
L_{\text{total}} = \lambda_{\text{VLB}} L_{\text{VLB}} + \lambda_{\text{MSE}} L_{\text{MSE}} + \lambda_{\text{LPIPS}} L_{\text{LPIPS}},
```

where $L_{VLB}$ ensures efficient latent-space representation, $L_{MSE}$ preserves pixel-level accuracy, and $L_{LPIPS}$ enhances perceptual quality in a learned embedding space.


## Architecture


  <img background-color=white width="800" alt="image" src="./figs/Video Decoder.png">

## Results


<img background-color=white width="800" alt="image" src="./figs/Beauty_comparison.png">


## Acknowledgements 

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University))

