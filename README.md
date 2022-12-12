# NeRF
Neural Radience Field(Pytorch)

Implementation of NeRF(https://www.matthewtancik.com/nerf) as final project for Deep Learning for 3D course. The only distinction of this implementation with the original implementation is that it implements hierarcichal training in a two stage fashion.

First Stage trains the coarse model and the second stage trains the fine model with hierarchichal sampling. This was done to reduce the GPU memory.

There is also an option to init the second stage model with the coarse model of first stage, which I have used to get 29.287 psnr on a bottles dataset with 100,000 iterations in both phase.
