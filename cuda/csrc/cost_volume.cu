#include <torch/extension.h>
#include "anytime_kernels.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor cost_volume(torch::Tensor left,
                          torch::Tensor right, 
                          int max_disparity){
    CHECK_INPUT(left);
    CHECK_INPUT(right);

    int height = left.size(1);
    int width = left.size(2);
    int feat_size = left.size(0);

    torch::Tensor out;
    out = torch::zeros({max_disparity, height, width},
                       torch::dtype(left.dtype()).
                       device(torch::kCUDA).
                       requires_grad(false));
    
    k_100(left, right, out, height, width, max_disparity, feat_size);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cost_volume", &cost_volume, "Cost Volume (CUDA)");
}

