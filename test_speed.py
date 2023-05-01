import torch.utils.benchmark as benchmark
import dqtorch 
import torch

def _forward_and_backward(fun, args):
    for x in args:
        x.requires_grad = True
    out = fun(*args)
    loss = out.sum()
    loss.backward()
    for x in args:
        x.requires_grad = False


# create two quaternions
n_pts = 4096 * 128
# n_pts = 2048*128 # a typical workload for NeRF training
qr_1 = dqtorch.axis_angle_to_quaternion(torch.randn(n_pts, 3).cuda())
qr_2 = dqtorch.axis_angle_to_quaternion(torch.randn(n_pts, 3).cuda())

# 1. test the speed up of quaternino_mul over native pytorch implementaion 
print('------ quaternion_mul --------')
t = benchmark.Timer(
    stmt='_quaternion_mul_pytorch(qr_1, qr_2)',
    setup='from dqtorch import _quaternion_mul_pytorch',
    globals={'qr_1': qr_1, 'qr_2': qr_2})
print(t.timeit(200))

t = benchmark.Timer(
    stmt='quaternion_mul(qr_1, qr_2)',
    setup='from dqtorch import quaternion_mul',
    globals={'qr_1': qr_1, 'qr_2': qr_2})
print(t.timeit(200))

t = benchmark.Timer(
    stmt='_forward_and_backward(fun, (qr_1, qr_2))',
    setup='from __main__ import _forward_and_backward',
    globals={'fun':dqtorch._quaternion_mul_pytorch, 'qr_1': qr_1, 'qr_2': qr_2})
print(t.timeit(200))

t = benchmark.Timer(
    stmt='_forward_and_backward(fun, (qr_1, qr_2))',
    setup='from __main__ import _forward_and_backward',
    globals={'fun':dqtorch.quaternion_mul, 'qr_1': qr_1, 'qr_2': qr_2})
print('native pytorch: ', t.timeit(200))

print('------ quaternion_conjugate --------')

t = benchmark.Timer(
    stmt='quaternion_conjugate(qr_1)',
    setup='from dqtorch import quaternion_conjugate',
    globals={'qr_1': qr_1})
print(t.timeit(200))

t = benchmark.Timer(
    stmt='_quaternion_conjugate_pytorch(qr_1)',
    setup='from dqtorch import _quaternion_conjugate_pytorch',
    globals={'qr_1': qr_1})
print(t.timeit(200))




    