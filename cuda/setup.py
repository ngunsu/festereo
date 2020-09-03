from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = ['csrc/cost_volume.cu',
           'csrc/kernels/anytime_kernels.cu']
include_dirs = ['include/']
nvcc = [
    '-gencode', 'arch=compute_53,code=sm_53',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_62,code=sm_62',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_72,code=sm_72',
    '-gencode', 'arch=compute_70,code=compute_70',
    '-lineinfo',
    '-Xptxas',
    '-dlcm=ca']
extra_compile_args = {'cxx': [], 'nvcc': nvcc}

setup(name='cost_volume',
      version='0.0.1',
      ext_modules=[CUDAExtension('cost_volume',
                                 sources,
                                 extra_compile_args=extra_compile_args,
                                 include_dirs=include_dirs)],
      cmdclass={'build_ext': BuildExtension})
