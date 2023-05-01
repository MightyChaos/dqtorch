#include <torch/extension.h>

#include "quaternion.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quaternion_mul_forward", &quaternion_mul_forward, "quaternion multiplication forward (CUDA)");
    m.def("quaternion_mul_backward", &quaternion_mul_backward, "quaternion multiplication backward (CUDA)");
    m.def("quaternion_mul_backward_backward", &quaternion_mul_backward_backward, "quaternion multiplication backward (CUDA)");
    m.def("quaternion_conjugate", &quaternion_conjugate, "quaternion_conjugate (CUDA)");
}