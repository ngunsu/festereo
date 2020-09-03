#include <cstdio>
#include <torch/extension.h>

constexpr int K_100_THREADS = 64;

//-----------------------------------------------------------------------
// K_100
//-----------------------------------------------------------------------
template <typename scalar_t>
__global__ void 
__launch_bounds__(1024) k_100_cuda(const scalar_t* __restrict__ left,
                                   const scalar_t* __restrict__ right,
                                   scalar_t* __restrict__ cost_volume,
                                   const int height,
                                   const int width,
                                   const int max_disparity,
                                   const int feature_size){
     
    const int pos_t = (blockIdx.x*blockDim.x) + threadIdx.x;
    const int h = pos_t/width;  
    const int w = pos_t-h*width; 
    int i = 0;
    int j = 0; 

    if (h < height && w < width){
        const int offset = h*width+w;

        for(i=0; i<max_disparity; i++){
            scalar_t maxmul_sum = 0;
            #pragma unroll 
            for(j=0; j<feature_size; j++){
                scalar_t sum = 0;
                if(i<w+1)
                    sum = left[offset+j*width*height] - right[offset-i+(width*height*j)];
                else
                    sum = left[offset+j*width*height];
                maxmul_sum += abs(sum);
            }
            cost_volume[offset+i*width*height] = maxmul_sum;
        }
    }
}

void k_100(torch::Tensor left,
           torch::Tensor right,
           torch::Tensor cost_volume,
           const int height,
           const int width,
           const int max_disparity,
           const int feature_size){

    int numel = float(height * width);

    dim3 grid(ceil(numel/K_100_THREADS)+1);
    dim3 block(K_100_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(left.type(), "k_100_cuda", ([&] {
                                        k_100_cuda<scalar_t><<<grid, block>>>(left.data<scalar_t>(),
                                                                              right.data<scalar_t>(),
                                                                              cost_volume.data<scalar_t>(),
                                                                              height,
                                                                              width,
                                                                              max_disparity,
                                                                              feature_size);

    }));
    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }
 
}

