#ifndef ANYTIME_KERNELS 
#define ANYTIME_KERNELS 


void k_100(torch::Tensor left,
           torch::Tensor right,
           torch::Tensor cost_volume,
           const int height,
           const int width,
           const int max_disparity,
           const int feature_size);

#endif


