#pragma once

#ifndef _INCLUDE_ADVECT
#define _INCLUDE_ADVECT

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include"sampling_settings_v2.hpp"

using GPUDevice = Eigen::GpuDevice;

template<typename Device, typename T, int32_t C>
struct AdvectGridKernel{
	void operator()(const GPUDevice& d,
		const T* input, const long long int* input_shape,
		const float* velocity, T* tmp_fwd, T* tmp_min, T* tmp_max,
		const float timestep, const int32_t order, const Sampling::BoundaryMode boundaryMode,
		const bool revertExtrema, const int32_t numVelocities, const bool globalSampling,
		T* output, const long long int* output_shape);


};


#endif //_INCLUDE_ADVECT
