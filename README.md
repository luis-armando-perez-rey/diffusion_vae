# Diffusion Variational Autoencoder (∆VAE)
A  standard  Variational  Autoencoder,  with  a  Euclidean  latent  space,  is  structurally  incapable  of capturing topological properties of certain datasets. To remove the topological obstructions,  we introduce the Diffusion  Variational  Autoencoders  (∆VAE)  with arbitrary(closed) manifolds as a latent space.

### Implementation of Diffusion Variational Autoencoders
This repository contains the code for the [Diffusion Variational Autoencoders](https://arxiv.org/abs/1901.08991) paper [1]. It includes the necessary code for embedding image datasets into the [hyperspherical space](https://en.wikipedia.org/wiki/N-sphere) <img src="https://render.githubusercontent.com/render/math?math=S^d">, the [Clifford Torus](https://en.wikipedia.org/wiki/Clifford_torus) <img src="https://render.githubusercontent.com/render/math?math=S^1\times S^1">, the [torus](https://en.wikipedia.org/wiki/Torus) embedded in 3-dimensional Euclidean space <img src="https://render.githubusercontent.com/render/math?math=S^1\times S^1\subseteq \mathbb{R}^3">, the [orthogonal group in 3-dimensions](https://en.wikipedia.org/wiki/Orthogonal_group) O(3) , the [special orthogonal group in 3-dimensions](https://en.wikipedia.org/wiki/3D_rotation_group) SO(3), the d-dimensional [real projective space](https://en.wikipedia.org/wiki/Real_projective_space) <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}\mathbb{P}^d"> , and the standard Variational Autoencoder with d-dimensional Euclidean space <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^d"> [2]


## Dependencies

- python>=3.6
- tensorflow>=1.7
- imageio
- scipy




## Notebook Examples

- **binary_MNIST**: This notebook shows an example on how to train a ∆VAE with all possible manifolds using multi-layer perceptrons for the encoder and decoder network. 
![alt text](https://github.com/luis-armando-perez-rey/diffusion_vae_github/blob/master/images/manifolds.PNG "Embedding of Binary MNIST into different manifolds")

### Results after training

After training any of the example notebooks, the outcomes will be saved in the results folder within the notebooks directory. The following folders will be created:
- **images**: contains the images of the embedded data in latent space and its reconstructions if the plotting function is implemented for the given manifold.
- **tensorboard**: contains the log files monitoring the relevant components of the loss, metrics of interest and the computation graph.
- **weights_folder**: contains the trained weights that are saved once training is finished.
- **parameters**: contains the json files with the parameter values used for that given experiment. They include the encoder, decoder and diffusion variational autoencoders parameters. 



## Contact
For any questions regarding the code and the paper refer to [Luis Armando Pérez Rey](mailto:l.a.perez.rey@tue.nl)

## Citation 
[1] Perez Rey, L.A., Menkovski, V., Portegies, J.W. (2019). *Diffusion Variational Autoencoders*. 34th Conference on Uncertainty in Artificial Intelligence (UAI-18).

*BibTeX*
```
@article{deltavae19,
  title={Diffusion Variational Autoencoders},
  author={Perez Rey, L.A. and
          Menkovski, V. and
          Portegies, J.W.},
  journal={arXiv preprint},
  archivePrefix = {arXiv},
  eprint    = {1901.08991},
  year={2019}
}
```
## References
[2] Kingma, D.P.  and  Welling, M. (2014) *Auto-Encoding  Variational  Bayes*. In International Conference on Learning Representations (ICLR).

## License 
Apache License 2.0
