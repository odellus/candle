// HIP compatibility header for Candle
// This is NOT a hipified version of compatibility.cuh - it's a HIP-native header
// that provides only the functions HIP doesn't have natively.

#ifndef CANDLE_HIP_COMPATIBILITY_H
#define CANDLE_HIP_COMPATIBILITY_H

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "hip/hip_bfloat16.h"

// HIP provides __hmax_nan, __hmin_nan, atomicAdd for double natively
// We only need to provide atomicMaxf/atomicMinf for half/float/double

// Type alias for CUDA compatibility in kernel code
typedef hip_bfloat16 hip_bfloat16;

// __vsubss4: SIMD subtraction of four signed 8-bit values packed in 32-bit integers
// CUDA has this as a built-in, HIP needs a manual implementation
__device__ __forceinline__ int __vsubss4(int a, int b) {
    int result = 0;
    for (int i = 0; i < 4; i++) {
        int8_t va = (int8_t)((a >> (i * 8)) & 0xFF);
        int8_t vb = (int8_t)((b >> (i * 8)) & 0xFF);
        int diff = (int)va - (int)vb;
        // Clamp to [-128, 127] for signed saturation
        if (diff < -128) diff = -128;
        if (diff > 127) diff = 127;
        result |= ((diff & 0xFF) << (i * 8));
    }
    return result;
}

// atomicMaxf for half
__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmax_nan(val, __ushort_as_half(assumed))));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMaxf for float - uses bit manipulation for correct ordering with negative numbers
__device__ __forceinline__ float atomicMaxf(float* addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    }
}

// atomicMaxf for double
__device__ __forceinline__ double atomicMaxf(double* addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int*)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int*)addr, __double_as_longlong(value)));
    }
}

// atomicMinf for half
__device__ __forceinline__ __half atomicMinf(__half* address, __half val) {
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmin_nan(val, __ushort_as_half(assumed))));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMinf for float
__device__ __forceinline__ float atomicMinf(float* addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    }
}

// atomicMinf for double
__device__ __forceinline__ double atomicMinf(double* addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int*)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int*)addr, __double_as_longlong(value)));
    }
}

// atomicAdd for __half (FP16)
__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        __half sum = __hadd(__ushort_as_half(assumed), val);
        old = atomicCAS(casted_address, assumed, __half_as_ushort(sum));
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicAdd for hip_bfloat16 (BF16)
__device__ __forceinline__ hip_bfloat16 atomicAdd(hip_bfloat16* address, hip_bfloat16 val) {
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        // Convert to float using hip_bfloat16's operator float()
        hip_bfloat16 old_bf;
        old_bf.data = assumed;
        float old_f = float(old_bf);
        float val_f = float(val);
        // Convert back using hip_bfloat16's constructor
        hip_bfloat16 sum(old_f + val_f);
        old = atomicCAS(casted_address, assumed, sum.data);
    } while (assumed != old);
    hip_bfloat16 result;
    result.data = old;
    return result;
}

#endif // CANDLE_HIP_COMPATIBILITY_H
