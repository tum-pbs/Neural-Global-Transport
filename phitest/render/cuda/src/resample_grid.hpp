#pragma once

#ifndef _INCLUDE_RESAMPLE_GRID
#define _INCLUDE_RESAMPLE_GRID

//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "sampling_settings.hpp"

using GPUDevice = Eigen::GpuDevice;
const bool NORMALIZE_GRADIENTS = true;

template<typename Device, typename T, int32_t C>
struct SampleGridKernel{
	void operator()(const Device& d,
		const void* input,const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras,
		uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings, const bool globalSampling,
		void* output, const long long int* output_shape);
};

/*
#if GOOGLE_CUDA

template<typename T, int32_t C>
struct SampleGridKernel<GPUDevice, T, C>{
	void operator()(const GPUDevice& d,
		const void* input,const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras,
		uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings, const bool globalSampling,
		void* output, const long long int* output_shape);
};

#endif
*/

#endif //_INCLUDE_RESAMPLE_GRID