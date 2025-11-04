#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_

// GPU Runtime Abstraction Layer
// This header provides a unified interface for both CUDA and ROCm/HIP

#ifdef USE_ROCM
// ROCm/HIP includes
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Map CUDA types to HIP types
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaDeviceProp hipDeviceProp_t

// Map CUDA functions to HIP functions
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// Memory management
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocDefault hipHostAllocDefault

// Stream management
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

// Device management
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceFlags hipGetDeviceFlags
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset


#else
#include <cuda_runtime_api.h>
#endif


#endif // _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_GPU_RUNTIME_H_
