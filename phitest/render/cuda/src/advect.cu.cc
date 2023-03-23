
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/platform/errors.h"
//#include "tensorflow/core/framework/op_kernel.h"
#include <cuda_runtime.h>
#include "cuda-samples/Common/helper_cuda.h"
#include <iostream>
#include <string>
#include "render_errors.hpp"
//#define LOGGING

#ifdef LOGGING
#define PROFILING
#endif

#ifdef PROFILING
//#include <sys/time.h>
#include <chrono>
#endif


#include "vectormath.hpp"
#include "vector_io.hpp"

//kernel_setup params
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4
//define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y*BLOCK_SIZE_Z
//#define BLOCK_DIMS BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z
#include "kernel_setup.hpp"
#include "advect.hpp"
#include "sampling_v2.hpp"

//#include "sampling_v2.hpp"

inline __device__ int3 make_globalThreadIdx(){
	return make_int3(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
	//return make_int3(blockIdx*blockDim + threadIdx);
}


struct AdvectionConstants{
	int3 dimensionsInput;
	int3 dimensionsOutput;
	int32_t batch;
	int32_t channel;
	float timestep;
	bool revertExtrema;
};
__constant__ AdvectionConstants c_adv;


// --- SemiLangrange Advection ---

template<typename T, Sampling::BoundaryMode BM, bool EXTREMA>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kAdvectGrid3DSemiLangrange(const T* UG_PTR input, const float3* UG_PTR velocity, T* UG_PTR output, T* UG_PTR sampledMin, T* UG_PTR sampledMax){
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		float3 samplePos = vectorIO::readVectorType3D<float3,float3,int3>(globalIdx, c_adv.dimensionsOutput, velocity);
		samplePos *= c_adv.timestep;
		samplePos += make_float3(globalIdx);
		
		T data;
		if(EXTREMA){ //sample with extrema
			Sampling::DataWithExtrema<T> dataWithExtrema = Sampling::read3DInterpolatedWithExtrema<T, BM>(samplePos, input, c_adv.dimensionsInput, 0);
			data = dataWithExtrema.data;
			vectorIO::writeVectorType3D<T, T, int3>(data, globalIdx, c_adv.dimensionsOutput, sampledMin);
			vectorIO::writeVectorType3D<T, T, int3>(data, globalIdx, c_adv.dimensionsOutput, sampledMax);
		}else{ //sample without extrema
			data = Sampling::read3DInterpolated<T, BM>(samplePos, input, c_adv.dimensionsInput, 0);
		}
		vectorIO::writeVectorType3D<T, T, int3>(data, globalIdx, c_adv.dimensionsOutput, output);
	}
}

/*
 * input (grad): [depth, height, width, channel (1-4)]
 * lut (grad): [depth, height, width, channel (4)], channel: (abs_x, abs_y, abs_z, LoD)
https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/resampler/kernels/resampler_ops_gpu.cu.cc
*/
template<typename T, Sampling::BoundaryMode BM, bool GRADIENT_ADDITIVE>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kAdvectGrid3DSemiLangrangeGradients(const T* UG_PTR input, const T* UG_PTR output_grad, const float3* UG_PTR velocity, T* UG_PTR input_grad, float3* UG_PTR velocity_grad){
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		float3 samplePos = vectorIO::readVectorType3D<float3,float3,int3>(globalIdx, c_adv.dimensionsOutput, velocity);
		samplePos *= c_adv.timestep;
		samplePos += float3(globalIdx, 0.0);
		
		//
		const T out_grad = vectorIO::readVectorType3D<T, T, int3>(globalIdx, c_adv.dimensionsOutput, output_grad);
		
		//backprop gradients to velocity, including timestep
		Sampling::DataGrad3D<T> dataGrad = Sampling::read3DGrad<T,BM>(samplePos, input, c_adv.dimensionsInput);
		const float3 vel_grad = c_adv.timestep * make_float3(vmath::sum(out_grad*dataGrad.dx), vmath::sum(out_grad*dataGrad.dy), vmath::sum(out_grad*dataGrad.dz));
		if(GRADIENT_ADDITIVE){
			vectorIO::addVectorType3D<float3, float3, int3>(vel_grad, globalIdx, c_adv.dimensionsOutput, velocity_grad);
		}else{
			vectorIO::writeVectorType3D<float3, float3, int3>(vel_grad, globalIdx, c_adv.dimensionsOutput, velocity_grad);
		}
		
		Sampling::scatterGrad3DInterpolated<T,BM>(out_grad, samplePos, input_grad);
	}
}

// --- MacCormack correction step ---
// after a forward and a backward SemiLangrangian step
/*
template<typename T>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kCorrectGrid3DMacCormack(const T* input, const T* fwd, T* output, const T* sampledMin, const T* sampledMax, bool revertExtrema){
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		const T dataIn = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, input);
		const T dataFwd = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, fwd);
		const T dataBwd = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, output);
		
		T corrected = dataFwd + (dataIn - dataBwd) * 0.5f;
		
		if(revertExtrema){
			const T dataMin = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMin);
			const T dataMax = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMax);
			
			/*
			if (corrected<min && max<corrected) fwd; //out of bounds
			else corrected;
			* /
			corrected = lerp(corrected, dataFwd, (corrected<dataMin)*(dataMax<corrected));
		}
		
		vectorIO::writeVectorType3D<T, T, int3>(corrected, globalIdx, c_adv.dimensionsOutput, output);
	}
}

template<typename T>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kCorrectGrid3DMacCormackGradients(T* input_grads, T* fwd_grads, const T* output_grads, const T* sampledMin, const T* sampledMax, bool revertExtrema){
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		const T gradOut = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, output_grads);
		T gradFwd;
		T gradCorrected;
		
		if(revertExtrema){
			const T dataMin = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMin);
			const T dataMax = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMax);
			
			T t = (corrected<dataMin)*(dataMax<corrected)
			gradCorrected = gradOut * (1.0f - t);
			gradFwd = gradOut * t;
		}else{
			gradCorrected = gradOut;
			gradFwd = T(0);
		}
		
		gradFwd += gradCorrected;
		T gradIn = gradCorrected * 0.5f;
		T gradBwd = gradCorrected * 0.5f;
		
		
		vectorIO::writeVectorType3D<T, T, int3>(gradIn, globalIdx, c_adv.dimensionsOutput, input_grads);
		vectorIO::writeVectorType3D<T, T, int3>(gradFwd, globalIdx, c_adv.dimensionsOutput, fwd_grads);
		vectorIO::writeVectorType3D<T, T, int3>(gradBwd, globalIdx, c_adv.dimensionsOutput, bwd_grads);
	}
}
*/
// --- MacCormack Step (fused)---
// combines SemiLangrangian backward step and MacCormack correction, after a SL forward step

template<typename T, Sampling::BoundaryMode BM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kAdvectCorrectGrid3DMacCormack(const T* UG_PTR input, const T* UG_PTR fwd, const T* UG_PTR sampledMin, const T* UG_PTR sampledMax, const float3* UG_PTR velocity, T* UG_PTR output){
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		float3 samplePosBwd = vectorIO::readVectorType3D<float3,float3,int3>(globalIdx, c_adv.dimensionsOutput, velocity);
		const T dataIn = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsInput, input);
		const T dataFwd = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, fwd);
		samplePosBwd *= - c_adv.timestep;
		samplePosBwd += make_float3(globalIdx);
		const T dataBwd = Sampling::read3DInterpolated<T, BM>(samplePosBwd, fwd, c_adv.dimensionsOutput, 0);
		
		T corrected = dataFwd + (dataIn - dataBwd) * 0.5f;
		
		if(c_adv.revertExtrema){
			const T dataMin = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMin);
			const T dataMax = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMax);
			
			/*
			if (corrected<min && max<corrected) fwd; //out of bounds
			else corrected;
			*/
			corrected = vmath::lerp<T>(corrected, dataFwd, (corrected<dataMin)*(dataMax<corrected));
		}
		
		vectorIO::writeVectorType3D<T, T, int3>(corrected, globalIdx, c_adv.dimensionsOutput, output);
	}
}

template<typename T, Sampling::BoundaryMode BM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kAdvectCorrectGrid3DMacCormackGradients(const T* UG_PTR input, const T* UG_PTR output_grad, const T* UG_PTR fwd, const T* UG_PTR sampledMin, const T* UG_PTR sampledMax, const float3* UG_PTR velocity,
	T* UG_PTR input_grad, T* UG_PTR fwd_grad, float3* UG_PTR velocity_grad){
	
	int3 globalIdx = make_globalThreadIdx();
	if(isInDimensions<int3,int3>(globalIdx, c_adv.dimensionsOutput)){
		float3 samplePosBwd = vectorIO::readVectorType3D<float3,float3,int3>(globalIdx, c_adv.dimensionsOutput, velocity);
		samplePosBwd *= -c_adv.timestep;
		samplePosBwd += float3(globalIdx, 0.0);
		const T gradOut = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, output_grad);
		T gradFwd;
		T gradCorrected;
		
		if(c_adv.revertExtrema){
			const T dataMin = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMin);
			const T dataMax = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, sampledMax);
			
			const T dataIn = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, input);
			const T dataFwd = vectorIO::readVectorType3D<T,T,int3>(globalIdx, c_adv.dimensionsOutput, fwd);
			const T dataBwd = Sampling::read3DInterpolated<T, BM>(samplePosBwd, fwd, c_adv.dimensionsOutput, 0);
			const T corrected = dataFwd + (dataIn - dataBwd) * 0.5f;
			
			T t = (corrected<dataMin)*(dataMax<corrected);
			gradCorrected = gradOut * (1.0f - t);
			gradFwd = gradOut * t;
		}else{
			gradCorrected = gradOut;
			gradFwd = T(0);
		}
		
		gradFwd += gradCorrected;
		T gradIn = gradCorrected * 0.5f;
		T gradBwd = gradCorrected * 0.5f;
		
		//backprop gradients to velocity, including timestep
		Sampling::DataGrad3D<T> dataGrad = Sampling::read3DGrad<T,BM>(samplePosBwd, fwd, c_adv.dimensionsOutput);
		const float3 vel_grad = (-c_adv.timestep) * make_float3(vmath::sum(gradBwd*dataGrad.dx), vmath::sum(gradBwd*dataGrad.dy), vmath::sum(gradBwd*dataGrad.dz));
		vectorIO::writeVectorType3D<float3, float3, int3>(vel_grad, globalIdx, c_adv.dimensionsOutput, velocity_grad);
		
		vectorIO::writeVectorType3D<T, T, int3>(gradIn, globalIdx, c_adv.dimensionsOutput, input_grad);
		//vectorIO::writeVectorType3D<T, T, int3>(gradFwd, globalIdx, c_adv.dimensionsOutput, fwd_grads);
		//-> needs to be atomic to be compatible with bwd scatter
		vectorIO::atomicAddVectorType3D<T>(gradFwd, globalIdx, c_adv.dimensionsOutput, fwd_grad);
		//vectorIO::writeVectorType3D<T, T, int3>(gradBwd, globalIdx, c_adv.dimensionsOutput, bwd_grads);
		Sampling::scatterGrad3DInterpolated<T, BM>(gradBwd, samplePosBwd, fwd_grad);
	}
}

inline int3 dimensionsFromGridShape(const long long int* shape, uint32_t offset=1){
	return make_int3(shape[offset+2], shape[offset+1], shape[offset]); //default offset 1: NDHWC (zyx) -> WHD (xyz)
}


template<typename T>
void AdvectGridKernelLauncher(const GPUDevice& d,
		const T* input, const long long int* shape_input,
		const float3* velocity, T* tmp_fwd, T* tmp_min, T* tmp_max,
		const float timestep, const int32_t order, const Sampling::BoundaryMode boundaryMode,
		const bool revertExtrema, const int32_t numVelocities, const bool globalSampling,
		T* output, const long long int* shape_output
	){
	LOG("Begin AdvectGridKernelLauncher");
	
	//setup constants
	AdvectionConstants advectionConstants;
	{
		memset(&advectionConstants, 0, sizeof(AdvectionConstants));
		advectionConstants.dimensionsInput = dimensionsFromGridShape(shape_input,1);
		advectionConstants.dimensionsOutput = dimensionsFromGridShape(shape_output,2);
		advectionConstants.batch = shape_input[0];
		//advectionConstants.channel = shape_input[4];
		advectionConstants.timestep = timestep;
		advectionConstants.revertExtrema = revertExtrema;
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_adv, &advectionConstants, sizeof(AdvectionConstants)));
	}
	const size_t inputBatchSizeElements = vmath::prod(advectionConstants.dimensionsInput);
	const size_t velocityBatchSizeElements = vmath::prod(advectionConstants.dimensionsOutput);
	const size_t outputBatchSizeElements = velocityBatchSizeElements;
	
	
	const dim3 grid(GRID_DIMS(advectionConstants.dimensionsOutput));
	const dim3 block(BLOCK_DIMS);
	LOG("Advect " << batchSize << " grids with " << numVelocities << " velocities");
	
	for(size_t batch=0; batch<advectionConstants.batch; ++batch){
		const T* currInput = input + batch*inputBatchSizeElements;
		
		size_t velocityIdx = globalSampling? 0 : batch;
		const size_t endVelocityIdx = globalSampling? numVelocities : velocityIdx+1;
		for(; velocityIdx<endVelocityIdx; ++velocityIdx){
			const float3* currVel = velocity + velocityIdx*velocityBatchSizeElements;
			T* currOut;
			if(order==2){
				currOut = tmp_fwd;
			}else{
				currOut = output + (globalSampling? batch*numVelocities + velocityIdx : batch)*outputBatchSizeElements;
			}
			//SL fwd step
			if(order==2 && revertExtrema){
				if(boundaryMode==Sampling::BOUNDARY_BORDER)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_BORDER, true><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, tmp_min, tmp_max);
				else if(boundaryMode==Sampling::BOUNDARY_CLAMP)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_CLAMP, true><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, tmp_min, tmp_max);
				else if(boundaryMode==Sampling::BOUNDARY_WRAP)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_WRAP, true><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, tmp_min, tmp_max);
			}else{
				if(boundaryMode==Sampling::BOUNDARY_BORDER)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_BORDER, false><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, nullptr, nullptr);
				else if(boundaryMode==Sampling::BOUNDARY_CLAMP)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_CLAMP, false><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, nullptr, nullptr);
				else if(boundaryMode==Sampling::BOUNDARY_WRAP)
					kAdvectGrid3DSemiLangrange<T, Sampling::BOUNDARY_WRAP, false><<<grid, block, 0, d.stream()>>>(currInput, currVel, currOut, nullptr, nullptr);
			}
			
			if(order==2){
				currOut = output + (globalSampling? batch*numVelocities + velocityIdx : batch)*outputBatchSizeElements;
				//SL bwd step
				//kAdvectGrid3DSemiLangrange<, , false>(, );
				//kCorrectGrid3DMacCormack(, revertExtrema);
				if(boundaryMode==Sampling::BOUNDARY_BORDER)
					kAdvectCorrectGrid3DMacCormack<T, Sampling::BOUNDARY_BORDER><<<grid, block, 0, d.stream()>>>(currInput, tmp_fwd, tmp_min, tmp_max, currVel, currOut);
				else if(boundaryMode==Sampling::BOUNDARY_CLAMP)
					kAdvectCorrectGrid3DMacCormack<T, Sampling::BOUNDARY_CLAMP><<<grid, block, 0, d.stream()>>>(currInput, tmp_fwd, tmp_min, tmp_max, currVel, currOut);
				else if(boundaryMode==Sampling::BOUNDARY_WRAP)
					kAdvectCorrectGrid3DMacCormack<T, Sampling::BOUNDARY_WRAP><<<grid, block, 0, d.stream()>>>(currInput, tmp_fwd, tmp_min, tmp_max, currVel, currOut);
			}
		}
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("End AdvectGridKernelLauncher");
}

#define DEFINE_GPU_SPECS(T, C, VEC) \
	template<> \
	void AdvectGridKernel<GPUDevice, T, C>::operator()(const GPUDevice& d, \
		const T* input, const long long int* input_shape, \
		const float* velocity, T* tmp_fwd, T* tmp_min, T* tmp_max, \
		const float timestep, const int32_t order, const Sampling::BoundaryMode boundaryMode, \
		const bool revertExtrema, const int32_t numVelocities, const bool globalSampling, \
		T* output, const long long int* output_shape){ \
	AdvectGridKernelLauncher<VEC>(d, \
		reinterpret_cast<const VEC*>(input), input_shape, \
		reinterpret_cast<const float3*>(velocity), reinterpret_cast<VEC*>(tmp_fwd), reinterpret_cast<VEC*>(tmp_min), reinterpret_cast<VEC*>(tmp_max), \
		timestep, order, boundaryMode, \
		revertExtrema, numVelocities, globalSampling, \
		reinterpret_cast<VEC*>(output), output_shape); \
	} \
	template struct AdvectGridKernel<GPUDevice, T, C>;
DEFINE_GPU_SPECS(float, 1, float1);
DEFINE_GPU_SPECS(float, 2, float2);
DEFINE_GPU_SPECS(float, 4, float4);

/*
template<T>
AdvectGridGradsKernelLauncher(d,){
	
	//setup constants
	
	//zero gradient buffers
	
	//partially redo fwd step
	//SL fwd step for min/max
	if(order==2 && revertExtrema){
		kAdvectGrid3DSemiLangrange<, , true>(, );
	}
	
	if(order==2){
		
		kCorrectGrid3DMacCormackGradients(out_grads -> , revertExtrema);
		//SL bwd step
		kScatter3DGradients(bwd_grads -> fwd_grads);
	}
	
	
	kScatter3DGradients(fwd_grads -> input_grads);
	
}
*/

#endif
