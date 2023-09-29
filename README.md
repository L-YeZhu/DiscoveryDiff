# UnseenDiffusion

This is the official Pytorch implementation for the paper **Image Synthesis from Unseen Domains with Diffusion Models**.


## 1. Project Overview

In this work, we seek to explore the synthesis ability of pre-trained Diffusion Models for generating images from completely unseen domains compared to the training stage.


## 2. Environment Setup


## 3. Analytical Experiments on Representation Ability

Our work is based on the key observation: a DDM trained even on a single domain small dataset, already has sufficient representation ability to accurately reconstruct arbitrary unseen images from the inverted latent encoding following a relatively deterministic denoising trajectory.

### 3.1 Unseen reconstruction

Given a pre-trained diffusion domain on a single domain dataset (e.g., dog faces on AFHQ-Dog-256), we aim to show that the pre-trained model can reconstruct an arbitraty image with deterministic inversion and denoising processes.



### 3.2 Bandwidth for unseen images

Next, we introduce the concept of bandwidth for unseen domain images given a pre-trained DM. 



