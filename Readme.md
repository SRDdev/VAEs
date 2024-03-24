# Variational Autoencoder (VAE)

Welcome to the Variational Autoencoder (VAE) repository! This repository is part of the "Paper Replication Series," where we delve into state-of-the-art machine learning architectures, replicate them from scratch, and experiment with them on smaller and simpler datasets.

## What is a Variational Autoencoder (VAE)?

Variational Autoencoder (VAE) is a type of artificial neural network used in unsupervised learning and generative modeling tasks. It belongs to the broader family of autoencoder neural networks. The key idea behind VAE is to learn a low-dimensional representation of input data in an unsupervised manner while simultaneously learning the data distribution.

![VAEs](ConvVAE/outputs/generated_images.gif)

## How does a VAE work?

At its core, a VAE consists of two main components: an encoder and a decoder.

### Encoder:
The encoder takes an input data point and maps it to a latent space, where each point represents a latent variable. In the case of VAE, instead of directly mapping the input to a point in the latent space, the encoder outputs parameters (mean and variance) of a probability distribution (usually Gaussian) representing the latent space.

### Decoder:
The decoder then takes a point from the latent space and reconstructs the original input data. Similar to the encoder, instead of directly generating the output, the decoder samples from the learned distribution in the latent space to generate diverse outputs.

### Training:
During training, VAE aims to minimize a loss function that consists of two components: a reconstruction loss, which measures the difference between the input and the output, and a regularization term, which ensures that the distribution in the latent space remains close to a prior distribution (usually a standard Gaussian distribution).

### Sampling:
Once trained, the VAE can generate new data points by sampling from the distribution in the latent space and passing them through the decoder.


## How to Use:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/VAE-replication.git
    cd VAE-replication
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data:**
    Place your dataset in the `data/` directory or use the provided datasets.

4. **Train the Model:**
    ```bash
    python train.py --epochs 50 --batch_size 64 --latent_dim 2
    ```

5. **Evaluate the Model:**
    ```bash
    python evaluate.py --num_samples 10
    ```

## References:

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

Feel free to explore, experiment, and modify the code to suit your needs! If you have any questions or suggestions, don't hesitate to reach out. Happy coding! 🚀