# dqtorch
## Introduction
dqtorch is a pytorch libarary for fast batched (dual) quaternion operations. The speed improvement over native pytorch implementation (e.g. [Pytorch3D](https://pytorch3d.org/)) is achieved through highly optimized CUDA extensions.
## Performance
- batch size = 4096*128 
(for the use case of training deformable NeRF, we typically have 4096 rays with 128 points sampled per ray).
- tested on a single RTX3080 GPU.

| | quaternion_mul | quaternion_conjugate |
| --- | -----| -----|
| native pytorch | 561.3 us | 66.5 us |
| dqtorch | 33.0 us | 22.5 us |

## Key Features
- fast CUDA implementation for quaternion operations.
- basic operations for SO(3) and SE3(3) transformation using (dual) quaternions.
- supports all ``torch.half``, ``torch.float``, ``torch.double`` tensors.
- supports gradient of a gradient.

## Get Started
### Requirments
tested in Pytorch 1.12, CUDA-11.1, gcc-6.3.0.

To do: check if it compiles with:
- [ ] Pytorch 2.0
- [ ] CUDA 11.7

### Install
```
python setup.py install
```
### Test
```
python examples.py
```

## Tutorial
### Basic quaternion operations
```python
import dqtorch
import torch

n_pts = 4096*128

# get a normalized quaternion from a batch of random axis angles
qr1 = dqtorch.axis_angle_to_quaternion(torch.randn(n_pts, 3).cuda()) 
qr2 = dqtorch.axis_angle_to_quaternion(torch.randn(n_pts, 3).cuda())

# quaternion multiplication
qr3 = dqtorch.quaternion_mul(qr1, qr2)
# if the number of channels is 3, we assume the quaternion real part is 0
qr4 = dqtorch.quaternion_mul(qr1, qr2[..., 1:])

# apply rotation to 3d points
p1 = torch.randn(n_pts, 3).cuda()
p2 = dqtorch.quaternion_apply(qr1, p1)

# quaternion conjugate
inv_qr1 = dqtorch.quaternion_conjugate(qr1)
# apply inverse transform to p2, and we should get back p1
p1_by_inv = dqtorch.quaternion_apply(inv_qr1, p2)
print((p1_by_inv-p1).abs().max()) # should be close to 0
```

### SE(3) transformation by quaternion + translation 
```python
# se3 representation by quaternion + translation
t1 = torch.randn(n_pts, 3).cuda() # create random translations
# apply se3 transformation to points
p2 = dqtorch.quaternion_translation_apply(qr1, t1, p1)
# inverse of se3 transformation
qr1_inv, t1_inv = dqtorch.quaternion_translation_inverse(qr1, t1)
# compose two se3 transformation
qr3, t3 = dqtorch.quaternion_translation_compose((qr1_inv, t1_inv), (qr1, t1))
print((qr3[..., 0]-1).abs().max(), qr3[..., 1:].abs().max(), t3.abs().max()) # should be close to 0
```

### SE(3) transformation by dual quaternion
```python
# se3 representation by dual quaternions, which is stored as a tuple of two tensors
dq1 = dqtorch.quaternion_translation_to_dual_quaternion(qr1, t1)
dq1_inv = dqtorch.quaternion_translation_to_dual_quaternion(qr1_inv, t1_inv)
# compose two se3 transformation
dq3 = dqtorch.dual_quaternion_mul(dq1_inv, dq1)
print((dq3[0][..., 0]-1).abs().max(), dq3[0][..., 1:].abs().max(), dq3[1].abs().max()) # should be close to 0
# apply se3 transformation to points
p2 = dqtorch.dual_quaternion_apply(dq1, p1)
# dual quaternion inverse
dq3 = dqtorch.dual_quaternion_inverse(dq1)
print((dq3[0]-dq1_inv[0]).abs().max(), (dq3[1]-dq1_inv[1]).abs().max()) # should be close to 0
# convert from dual quaternion to quaternion translation
qr3, t3 = dqtorch.dual_quaternion_to_quaternion_translation(dq1)
print((qr3-qr1).abs().max(), (t3-t1).abs().max()) # should be close to 0
```

## Related Projects

dqtorch has been used in our research of deformable NeRF:
- [RAC: Reconstructing Animatable Categories from Videos](https://gengshan-y.github.io/rac-www/)
- [Flow Supervison for Deformable NeRF](https://mightychaos.github.io/projects/fsdnerf/index.html)




