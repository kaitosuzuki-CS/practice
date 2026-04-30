# Denoising Process for MNIST

This document shows the denoising process of the diffusion model at different timesteps for the MNIST dataset. The model starts with random noise at the highest timestep and gradually denoises it to generate a clean image at timestep 0.

| Timestep |                   Denoised Image                   |
| :------: | :------------------------------------------------: |
| **999**  | ![Denoised image at timestep 999](imgs/x0_999.png) |
| **900**  | ![Denoised image at timestep 900](imgs/x0_900.png) |
| **800**  | ![Denoised image at timestep 800](imgs/x0_800.png) |
| **700**  | ![Denoised image at timestep 700](imgs/x0_700.png) |
| **600**  | ![Denoised image at timestep 600](imgs/x0_600.png) |
| **500**  | ![Denoised image at timestep 500](imgs/x0_500.png) |
| **400**  | ![Denoised image at timestep 400](imgs/x0_400.png) |
| **300**  | ![Denoised image at timestep 300](imgs/x0_300.png) |
| **200**  | ![Denoised image at timestep 200](imgs/x0_200.png) |
| **100**  | ![Denoised image at timestep 100](imgs/x0_100.png) |
|  **0**   |   ![Denoised image at timestep 0](imgs/x0_0.png)   |
