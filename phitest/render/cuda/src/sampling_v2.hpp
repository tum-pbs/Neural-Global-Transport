#pragma once

#ifndef _INCLUDE_SAMPLING_2
#define _INCLUDE_SAMPLING_2

#include"sampling_settings_v2.hpp"

#include "vectormath.hpp"
#include "vector_io.hpp"
#include "bounds_checks.hpp"

namespace Sampling{

template<typename T>
inline __device__ T read3D(const bool valid, const int32_t positionX, const int32_t positionY, const int32_t positionZ, const T* buf, const int3 buf_dims, const float constantValue){
	if(valid){
		const T data = vectorIO::readVectorType3D<T, T, int32_t, int3>(positionX, positionY, positionZ, buf_dims, buf);
		return data;
	}else{
		const T data = vmath::make_cudaFloat<T>(constantValue);
		return data;
	}
	
}

__host__ __device__ inline bool isInBounds3D(const float3 pos, const int3 dims){
	return	0<=pos.x && pos.x<dims.x &&
			0<=pos.y && pos.y<dims.y &&
			0<=pos.z && pos.z<dims.z ;
}
__host__ __device__ inline bool isInBounds3D(const int3 pos, const int3 dims){
	return	0<=pos.x && pos.x<dims.x &&
			0<=pos.y && pos.y<dims.y &&
			0<=pos.z && pos.z<dims.z ;
}

template<typename T, BoundaryMode BM>
__device__ inline T read3DInterpolated(const float3 position, const T* buf, const int3 buf_dims, const float constantValue){
	
	//weights for ceil, floor weights are (1-weights)
	const float3 weights = fracf(position);
	
	//calculate corner indices
	int3  ceilIdx = make_int3(ceilf(position));
	int3 floorIdx = make_int3(floorf(position));
	T data = vmath::make_cudaFloat<T>(constantValue);
	
	if (BM == BoundaryMode::BOUNDARY_BORDER){// const value outside domain
		if(!isInBounds3D(position+0.5f, buf_dims)){
			return vmath::make_cudaFloat<T>(constantValue);
		}
		int3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		int3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
#		define READ_EXTREMA(X,Y,Z) read3D((X##Valid.x*Y##Valid.y*Z##Valid.z)>0.f, X##Idx.x, Y##Idx.y, Z##Idx.z, buf, buf_dims, constantValue)
		//read and interpolate along x
		const T v00 = lerp(
			READ_EXTREMA(floor, floor, floor),
			READ_EXTREMA( ceil, floor, floor),
			weights.x);
		const T v01 = lerp(
			READ_EXTREMA(floor,  ceil, floor),
			READ_EXTREMA( ceil,  ceil, floor),
			weights.x);
		//interpolate along y
		const T v0 = lerp(v00, v01, weights.y);
		
		
		const T v10 = lerp(
			READ_EXTREMA(floor, floor,  ceil),
			READ_EXTREMA( ceil, floor,  ceil),
			weights.x);
		const T v11 = lerp(
			READ_EXTREMA(floor,  ceil,  ceil),
			READ_EXTREMA( ceil,  ceil,  ceil),
			weights.x);
		//interpolate along y
		const T v1 = lerp(v10, v11, weights.y);
#		undef READ_EXTREMA
		
		//interpolate along z
		data = lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}
		else if (BM == BOUNDARY_WRAP){//periodic
			ceilIdx = vmath::positivemod<int3, int3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<int3, int3>(floorIdx, buf_dims);
		}
#		define READ_EXTREMA(X,Y,Z) read3D(true, X##Idx.x, Y##Idx.y, Z##Idx.z, buf, buf_dims, constantValue)
		//read and interpolate along x
		const T v00 = lerp(
			READ_EXTREMA(floor, floor, floor),
			READ_EXTREMA( ceil, floor, floor),
			weights.x);
		const T v01 = lerp(
			READ_EXTREMA(floor,  ceil, floor),
			READ_EXTREMA( ceil,  ceil, floor),
			weights.x);
		//interpolate along y
		const T v0 = lerp(v00, v01, weights.y);
		
		
		const T v10 = lerp(
			READ_EXTREMA(floor, floor,  ceil),
			READ_EXTREMA( ceil, floor,  ceil),
			weights.x);
		const T v11 = lerp(
			READ_EXTREMA(floor,  ceil,  ceil),
			READ_EXTREMA( ceil,  ceil,  ceil),
			weights.x);
		//interpolate along y
		const T v1 = lerp(v10, v11, weights.y);
#		undef READ_EXTREMA
		
		
		//interpolate along z
		data = lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
	}
	
	return data;
}
template<typename T, BoundaryMode BM>
__device__ inline T read3DNearest(const float3 position, const T* buf, const int3 buf_dims, const float constantValue){
	int3 idx = make_int3(position +0.5f);
	if (BM == BOUNDARY_BORDER){
		if(!isInBounds3D(idx, buf_dims)){
			return vmath::make_cudaFloat<T>(constantValue);
		}
	}
	else if(BM == BOUNDARY_CLAMP){
		idx = clamp(idx, make_int3(0), buf_dims - 1);
	}
	else if (BM == BOUNDARY_WRAP){//periodic
		idx = vmath::pmod(idx, buf_dims);
	}
	return vectorIO::readVectorType3D<T, T, int3>(idx, buf_dims, buf);
}

template<typename T>
struct DataWithExtrema{
	T data;
	T min;
	T max;
};

template<typename T>
inline __device__ T read3DWithExtrema(const bool valid, const int32_t positionX, const int32_t positionY, const int32_t positionZ, DataWithExtrema<T>& dataExtrema, const T* buf, const int3 buf_dims, const float constantValue){
	if(valid){
		const T data = vectorIO::readVectorType3D<T, T, int32_t, int3>(positionX, positionY, positionZ, buf_dims, buf);
		dataExtrema.min = vmath::lerp<T>(dataExtrema.min, data, data < dataExtrema.min);
		dataExtrema.max = vmath::lerp<T>(dataExtrema.max, data, dataExtrema.max < data);
		return data;
	}else{
		const T data = vmath::make_cudaFloat<T>(constantValue);
		return data;
	}
	
}

template<typename T, BoundaryMode BM>
__device__ inline DataWithExtrema<T> read3DInterpolatedWithExtrema(const float3 position, const T* buf, const int3 buf_dims, const float constantValue){
	
	//weights for ceil, floor weights are (1-weights)
	const float3 weights = fracf(position);
	
	//calculate corner indices
	int3  ceilIdx = make_int3(ceilf(position));
	int3 floorIdx = make_int3(floorf(position));
	DataWithExtrema<T> data;
	data.data = vmath::make_cudaFloat<T>(constantValue);
	data.min = vmath::make_cudaFloat<T, float>(FLOAT_MAX);
	data.max = vmath::make_cudaFloat<T, float>(-FLOAT_MAX);
	
	if (BM == BOUNDARY_BORDER){// const 0 outside domain
		if(!isInBounds3D(position+0.5f, buf_dims)){
			return data;
		}
		int3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		int3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
#		define READ_EXTREMA(X,Y,Z) read3DWithExtrema((X##Valid.x*Y##Valid.y*Z##Valid.z)>0.f, X##Idx.x, Y##Idx.y, Z##Idx.z, data, buf, buf_dims, constantValue)
		//read and interpolate along x
		const T v00 = lerp(
			READ_EXTREMA(floor, floor, floor),
			READ_EXTREMA( ceil, floor, floor),
			weights.x);
		const T v01 = lerp(
			READ_EXTREMA(floor,  ceil, floor),
			READ_EXTREMA( ceil,  ceil, floor),
			weights.x);
		//interpolate along y
		const T v0 = lerp(v00, v01, weights.y);
		
		
		const T v10 = lerp(
			READ_EXTREMA(floor, floor,  ceil),
			READ_EXTREMA( ceil, floor,  ceil),
			weights.x);
		const T v11 = lerp(
			READ_EXTREMA(floor,  ceil,  ceil),
			READ_EXTREMA( ceil,  ceil,  ceil),
			weights.x);
		//interpolate along y
		const T v1 = lerp(v10, v11, weights.y);
#		undef READ_EXTREMA
		
		//interpolate along z
		data.data = lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}
		else if (BM == BOUNDARY_WRAP){//periodic
			ceilIdx = vmath::positivemod<int3, int3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<int3, int3>(floorIdx, buf_dims);
		}
#		define READ_EXTREMA(X,Y,Z) read3DWithExtrema(true, X##Idx.x, Y##Idx.y, Z##Idx.z, data, buf, buf_dims, constantValue)
		//read and interpolate along x
		const T v00 = lerp(
			READ_EXTREMA(floor, floor, floor),
			READ_EXTREMA( ceil, floor, floor),
			weights.x);
		const T v01 = lerp(
			READ_EXTREMA(floor,  ceil, floor),
			READ_EXTREMA( ceil,  ceil, floor),
			weights.x);
		//interpolate along y
		const T v0 = lerp(v00, v01, weights.y);
		
		
		const T v10 = lerp(
			READ_EXTREMA(floor, floor,  ceil),
			READ_EXTREMA( ceil, floor,  ceil),
			weights.x);
		const T v11 = lerp(
			READ_EXTREMA(floor,  ceil,  ceil),
			READ_EXTREMA( ceil,  ceil,  ceil),
			weights.x);
		//interpolate along y
		const T v1 = lerp(v10, v11, weights.y);
#		undef READ_EXTREMA
		
		
		//interpolate along z
		data.data = lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
	}
	
	return data;
}

// spatial data gradients

template<typename T>
struct DataGrad3D{
	T dx,dz,dy;
};
template<typename T, BoundaryMode BM>
__device__ inline DataGrad3D<T> read3DGrad(const float3 position, const T* buf, const int3 buf_dims, const float constantValue){
	
	//weights for ceil, floor weights are (1-weights)
	const float3 weights = fracf(position);
	
	//calculate corner indices
	int3  ceilIdx = make_int3(ceilf(position));
	int3 floorIdx = make_int3(floorf(position));
	
	if (BM == BOUNDARY_BORDER){// constant value outside domain
		if(!isInBounds3D(position+0.5f, buf_dims)){
			DataGrad3D<T> zero = {0};// 0 grad due to constant value
			return zero;
		}
		int3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		int3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data = vmath::make_cudaFloat<T>(constantValue);
		//read
		const T fxfyfz = (floorValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxfyfz = ( ceilValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxcyfz = (floorValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxcyfz = ( ceilValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxfycz = (floorValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxfycz = ( ceilValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T fxcycz = (floorValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxcycz = ( ceilValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		//interpolate differences
		DataGrad3D<T> dataGrad;
		dataGrad.dx = lerp(
						lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
						lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
					weights.z);
		dataGrad.dy = lerp(
						lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
						lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
					weights.z);
		dataGrad.dz = lerp(
						lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
						lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
					weights.y);
		
		return dataGrad;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}
		else if (BM == BOUNDARY_WRAP){//periodic
			ceilIdx = vmath::positivemod<int3, int3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<int3, int3>(floorIdx, buf_dims);
		}
		//read
		const T fxfyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T cxfyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T fxcyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T cxcyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T fxfycz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxfycz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T fxcycz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxcycz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		//interpolate differences
		DataGrad3D<T> dataGrad;
		dataGrad.dx = lerp(
						lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
						lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
					weights.z);
		dataGrad.dy = lerp(
						lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
						lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
					weights.z);
		dataGrad.dz = lerp(
						lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
						lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
					weights.y);
		
		return dataGrad;
	}
}

template<typename T, FilterMode FM, BoundaryMode BM>
__device__ inline T sample3D(const float3 position, const T* buf, const int3 buf_dims, const float constantValue = 0.f){
	if(FM==FILTERMODE_LINEAR){
		return read3DInterpolated<T, BM>(position - 0.5f, buf, buf_dims, constantValue);
	}else if(FM==FILTERMODE_NEAREST){
		return read3DNearest<T, BM>(position - 0.5f, buf, buf_dims, constantValue);
	}else if(FM==FILTERMODE_MIN){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims, constantValue).min;
	}else if(FM==FILTERMODE_MAX){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims, constantValue).max;
	}else{
		T data{};
		return data;
	}
}
template<typename T, FilterMode FM, BoundaryMode BM>
__device__ inline DataGrad3D<T> sample3DGrad(const float3 position, const T* buf, const int3 buf_dims, const float constantValue = 0.f){
	if(FM==FILTERMODE_LINEAR){
		return read3DGrad<T, BM>(position - 0.5f, buf, buf_dims, constantValue);
	}/* else if(FM==FILTERMODE_NEAREST){
		return read3DNearest<T, BM>(position - 0.5f, buf, buf_dims);
	}else if(FM==FILTERMODE_MIN){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).min;
	}else if(FM==FILTERMODE_MAX){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).max;
	} */else{
		DataGrad3D<T> data{};
		return data;
	}
}

// gradient scattering

/*
// position is without offset (+0.0)
template<typename T, BoundaryMode BM>
__device__ inline void scatterGrad3DInterpolated(const T out_grad, const float3 position, T* input_grad, const int3 buf_dims){
	if(BM!=BOUNDARY_BORDER || (CHECK_BOUNDS_SV3V3(-1.f, <, position, <, buf_dims))){
		const float3 cw = fracf(position);
		const float3 fw = 1.f - cw;
		int3 ceilIdx = make_int3(ceilf(position));
		int3 floorIdx = make_int3(floorf(position));
		
		if(BM==BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}else if(BM==BOUNDARY_WRAP){
			ceilIdx = vmath::positivemod<int3, int3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<int3, int3>(floorIdx, buf_dims);
		}
		
		//accumulate weighted gradients
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && floorIdx.z>=0)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(fw.x*fw.y*fw.z), floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && floorIdx.y>=0 && floorIdx.z>=0)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(cw.x*fw.y*fw.z), ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && ceilIdx.y<buf_dims.y && floorIdx.z>=0)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(fw.x*cw.y*fw.z), floorIdx.x, ceilIdx.y, floorIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && ceilIdx.y<buf_dims.y && floorIdx.z>=0)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(cw.x*cw.y*fw.z), ceilIdx.x, ceilIdx.y, floorIdx.z, buf_dims, input_grad);
		}
		
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && ceilIdx.z<buf_dims.z)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(fw.x*fw.y*cw.z), floorIdx.x, floorIdx.y, ceilIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && floorIdx.y>=0 && ceilIdx.z<buf_dims.z)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(cw.x*fw.y*cw.z), ceilIdx.x, floorIdx.y, ceilIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && ceilIdx.y<buf_dims.y && ceilIdx.z<buf_dims.z)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(fw.x*cw.y*cw.z), floorIdx.x, ceilIdx.y, ceilIdx.z, buf_dims, input_grad);
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && ceilIdx.y<buf_dims.y && ceilIdx.z<buf_dims.z)){
			vectorIO::atomicAddVectorType3D<T>(out_grad*(cw.x*cw.y*cw.z), ceilIdx.x, ceilIdx.y, ceilIdx.z, buf_dims, input_grad);
		}
	}
}*/
// position is without offset (+0.0)
template<typename T, Sampling::BoundaryMode BM, typename C>
struct SingleSampleContribution{
	__device__ inline static void scatter(const T data, const int3 idx, const float w, T* buf, C* num_samples, const int3 buf_dims);
};
template<typename T, Sampling::BoundaryMode BM>
struct SingleSampleContribution<T, BM, void>{
	__device__ inline static void scatter(const T data, const int3 idx, const float w, T* buf, void* num_samples, const int3 buf_dims){
		if(BM!=BOUNDARY_BORDER || (CHECK_BOUNDS_SV3V3(0, <=, idx, <, buf_dims))){
			const size_t flatIdx3D = vectorIO::flatIdx3D(idx.x, idx.y, idx.z, buf_dims);
			vectorIO::atomicAddVectorTypeAbs<T>(data*w, flatIdx3D, buf);
		}
	}
};
template<typename T, Sampling::BoundaryMode BM>
struct SingleSampleContribution<T, BM, uint32_t>{
	__device__ inline static void scatter(const T data, const int3 idx, const float w, T* buf, uint32_t* num_samples, const int3 buf_dims){
		if(BM!=BOUNDARY_BORDER || (CHECK_BOUNDS_SV3V3(0, <=, idx, <, buf_dims))){
			const size_t flatIdx3D = vectorIO::flatIdx3D(idx.x, idx.y, idx.z, buf_dims);
			vectorIO::atomicAddVectorTypeAbs<T>(data*w, flatIdx3D, buf);
			atomicInc(num_samples + flatIdx3D, 0xffffffff);
		}
	}
};
template<typename T, Sampling::BoundaryMode BM>
struct SingleSampleContribution<T, BM, float>{
	__device__ inline static void scatter(const T data, const int3 idx, const float w, T* buf, float* num_samples, const int3 buf_dims){
		if(BM!=BOUNDARY_BORDER
				|| (CHECK_BOUNDS_SV3V3(0, <=, idx, <, buf_dims))){
			const size_t flatIdx3D = vectorIO::flatIdx3D(idx.x, idx.y, idx.z, buf_dims);
			vectorIO::atomicAddVectorTypeAbs<T>(data*w, flatIdx3D, buf);
			atomicAdd(num_samples + flatIdx3D, w);
		}
	}
};

template<typename T, Sampling::BoundaryMode BM, typename C>
__device__ inline void scatterGrad3DInterpolated(const T out_grad, const float3 position, T* input_grad, C* num_samples, const int3 buf_dims){
	if(BM!=BOUNDARY_BORDER || (CHECK_BOUNDS_SV3V3(-1.f, <, position, <, buf_dims))){
		const float3 cw = fracf(position);
		const float3 fw = 1.f - cw;
		int3 ceilIdx = make_int3(ceilf(position));
		int3 floorIdx = make_int3(floorf(position));
		
		if(BM==BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}else if(BM==BOUNDARY_WRAP){
			ceilIdx = vmath::pmod(ceilIdx, buf_dims);
			floorIdx = vmath::pmod(floorIdx, buf_dims);
		}
		
		//accumulate weighted gradients
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(floorIdx.x, floorIdx.y, floorIdx.z), (fw.x*fw.y*fw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(ceilIdx.x, floorIdx.y, floorIdx.z), (cw.x*fw.y*fw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(floorIdx.x, ceilIdx.y, floorIdx.z), (fw.x*cw.y*fw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(ceilIdx.x, ceilIdx.y, floorIdx.z), (cw.x*cw.y*fw.z), input_grad, num_samples, buf_dims);
		
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(floorIdx.x, floorIdx.y, ceilIdx.z), (fw.x*fw.y*cw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(ceilIdx.x, floorIdx.y, ceilIdx.z), (cw.x*fw.y*cw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(floorIdx.x, ceilIdx.y, ceilIdx.z), (fw.x*cw.y*cw.z), input_grad, num_samples, buf_dims);
		SingleSampleContribution<T, BM, C>::scatter(out_grad, make_int3(ceilIdx.x, ceilIdx.y, ceilIdx.z), (cw.x*cw.y*cw.z), input_grad, num_samples, buf_dims);
		/*
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && floorIdx.z>=0)){
			const float w = (fw.x*fw.y*fw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && floorIdx.y>=0 && floorIdx.z>=0)){
			const float w = (cw.x*fw.y*fw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && ceilIdx.y<buf_dims.y && floorIdx.z>=0)){
			const float w = (fw.x*cw.y*fw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(floorIdx.x, ceilIdx.y, floorIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, floorIdx.x, ceilIdx.y, floorIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && ceilIdx.y<buf_dims.y && floorIdx.z>=0)){
			const float w = (cw.x*cw.y*fw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(ceilIdx.x, ceilIdx.y, floorIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, ceilIdx.x, ceilIdx.y, floorIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && ceilIdx.z<buf_dims.z)){
			const float w = (fw.x*fw.y*cw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(floorIdx.x, floorIdx.y, ceilIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, floorIdx.x, floorIdx.y, ceilIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && floorIdx.y>=0 && ceilIdx.z<buf_dims.z)){
			const float w = (cw.x*fw.y*cw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(ceilIdx.x, floorIdx.y, ceilIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, ceilIdx.x, floorIdx.y, ceilIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (floorIdx.x>=0 && ceilIdx.y<buf_dims.y && ceilIdx.z<buf_dims.z)){
			const float w = (fw.x*cw.y*cw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(floorIdx.x, ceilIdx.y, ceilIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, floorIdx.x, ceilIdx.y, ceilIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		if(BM!=BOUNDARY_BORDER || (ceilIdx.x<buf_dims.x && ceilIdx.y<buf_dims.y && ceilIdx.z<buf_dims.z)){
			const float w = (cw.x*cw.y*cw.z);
			const size_t flatIdx3D = vectorIO::flatIdx3D(ceilIdx.x, ceilIdx.y, ceilIdx.z, buf_dims);
			vectorIO::atomicAddVectorType3D<T>(out_grad*w, ceilIdx.x, ceilIdx.y, ceilIdx.z, buf_dims, input_grad);
			if(CountSamples==1){ atomicInc(num_samples + flatIdx3D, 0xffffffff); }
			else if(CountSamples==2){ atomicAdd(num_samples + flatIdx3D, w); }
		}
		*/
	}
}

template<typename T, FilterMode FM, BoundaryMode BM, typename C>
__device__ inline void scatter3D(const T data, const float3 position, T* buf, const int3 buf_dims, C* num_samples){
	if(FM==FILTERMODE_LINEAR){
		scatterGrad3DInterpolated<T, BM, C>(data, position - 0.5f, buf, num_samples, buf_dims);
	}/*else if(FM==FILTERMODE_NEAREST){
		scatterGrad3DNearest<T, BM, CountSamples, C>(position - 0.5f, buf, buf_dims);
	}else if(FM==FILTERMODE_MIN){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).min;
	}else if(FM==FILTERMODE_MAX){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).max;
	}*/
}

// gradients of the data gradients

/*
* d lerp(a,b,t)
*   /d a: (1-t); /d b: t; /d t: b-a
* d lerp(lerp(a,b,t1),lerp(c,d,t1),t2)
*   /d a: [d lerp(...,t2)/d lerp(a,b,t1)]*[d lerp(a,b,t1)/d a] = (1-t2)*(1-t1)
*   /d t1: (1-t2)*(b-a) + t2*(d-c) = lerp(b-a,d-c,t2)
*   /d t2: lerp(c,d,t1) - lerp(a,b,t1) = lerp(c-a,d-b,t1)
*/

template<typename T, BoundaryMode BM>
__device__ inline void scatter3DGradDataGradInterpolated(const float3 position, const DataGrad3D<T> dataGradGrad, T* buf, const int3 buf_dims){
	// gradient of data gradient w.r.t. original data
	//weights for ceil, floor weights are (1-weights)
	const float3 weights = fracf(position);
	
	//calculate corner indices
	int3  ceilIdx = make_int3(ceilf(position));
	int3 floorIdx = make_int3(floorf(position));
	
	T fxfyfz = vmath::make_cudaFloat<T>(0.0f);
	T cxfyfz = vmath::make_cudaFloat<T>(0.0f);
	T fxcyfz = vmath::make_cudaFloat<T>(0.0f);
	T cxcyfz = vmath::make_cudaFloat<T>(0.0f);
	T fxfycz = vmath::make_cudaFloat<T>(0.0f);
	T cxfycz = vmath::make_cudaFloat<T>(0.0f);
	T fxcycz = vmath::make_cudaFloat<T>(0.0f);
	T cxcycz = vmath::make_cudaFloat<T>(0.0f);
	
	// for reference
	//DataGrad3D<T> dataGrad;
	/* dataGrad.dx = lerp(
					lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
					lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
				weights.z); */
	// a=(cxfyfz-fxfyfz), t1=weights.y, t2=weights.z
	
	fxfyfz = fxfyfz - (1-weights.y)*(1-weights.z)*dataGradGrad.dx;
	cxfyfz = cxfyfz + (1-weights.y)*(1-weights.z)*dataGradGrad.dx;
	fxcyfz = fxcyfz -    weights.y *(1-weights.z)*dataGradGrad.dx;
	cxcyfz = cxcyfz +    weights.y *(1-weights.z)*dataGradGrad.dx;
	fxfycz = fxfycz - (1-weights.y)*   weights.z *dataGradGrad.dx;
	cxfycz = cxfycz + (1-weights.y)*   weights.z *dataGradGrad.dx;
	fxcycz = fxcycz -    weights.y *   weights.z *dataGradGrad.dx;
	cxcycz = cxcycz +    weights.y *   weights.z *dataGradGrad.dx;
	// => x~sign, y~weight.y, z~weight.z
	
	/* dataGrad.dy = lerp(
					lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
					lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
				weights.z); */
	// => y~sign, x~weight.x, z~weight.z
	fxfyfz = fxfyfz - (1-weights.x)*(1-weights.z)*dataGradGrad.dy;
	cxfyfz = cxfyfz -    weights.x *(1-weights.z)*dataGradGrad.dy;
	fxcyfz = fxcyfz + (1-weights.x)*(1-weights.z)*dataGradGrad.dy;
	cxcyfz = cxcyfz +    weights.x *(1-weights.z)*dataGradGrad.dy;
	fxfycz = fxfycz - (1-weights.x)*   weights.z *dataGradGrad.dy;
	cxfycz = cxfycz -    weights.x *   weights.z *dataGradGrad.dy;
	fxcycz = fxcycz + (1-weights.x)*   weights.z *dataGradGrad.dy;
	cxcycz = cxcycz +    weights.x *   weights.z *dataGradGrad.dy;
	
	/* dataGrad.dz = lerp(
					lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
					lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
				weights.y); */
	// => z~sign, x~weight.x, y~weight.y
	fxfyfz = fxfyfz - (1-weights.x)*(1-weights.y)*dataGradGrad.dz;
	cxfyfz = cxfyfz -    weights.x *(1-weights.y)*dataGradGrad.dz;
	fxcyfz = fxcyfz - (1-weights.x)*   weights.y *dataGradGrad.dz;
	cxcyfz = cxcyfz -    weights.x *   weights.y *dataGradGrad.dz;
	fxfycz = fxfycz + (1-weights.x)*(1-weights.y)*dataGradGrad.dz;
	cxfycz = cxfycz +    weights.x *(1-weights.y)*dataGradGrad.dz;
	fxcycz = fxcycz + (1-weights.x)*   weights.y *dataGradGrad.dz;
	cxcycz = cxcycz +    weights.x *   weights.y *dataGradGrad.dz;
	
	// scatter without weighting
	SingleSampleContribution<T, BM, void>::scatter(fxfyfz, make_int3(floorIdx.x, floorIdx.y, floorIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(cxfyfz, make_int3( ceilIdx.x, floorIdx.y, floorIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(fxcyfz, make_int3(floorIdx.x,  ceilIdx.y, floorIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(cxcyfz, make_int3( ceilIdx.x,  ceilIdx.y, floorIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(fxfycz, make_int3(floorIdx.x, floorIdx.y,  ceilIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(cxfycz, make_int3( ceilIdx.x, floorIdx.y,  ceilIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(fxcycz, make_int3(floorIdx.x,  ceilIdx.y,  ceilIdx.z), 1.0f, buf, nullptr, buf_dims);
	SingleSampleContribution<T, BM, void>::scatter(cxcycz, make_int3( ceilIdx.x,  ceilIdx.y,  ceilIdx.z), 1.0f, buf, nullptr, buf_dims);
}
template<typename T, FilterMode FM, BoundaryMode BM>
__device__ inline void scatter3DGradDataGrad(const float3 position, const DataGrad3D<T> dataGradGrad, T* buf, const int3 buf_dims){
	if(FM==FILTERMODE_LINEAR){
		scatter3DGradDataGradInterpolated<T, BM>(position - 0.5f, dataGradGrad, buf, buf_dims);
	}/*else if(FM==FILTERMODE_NEAREST){
		scatterGrad3DNearest<T, BM, CountSamples, C>(position - 0.5f, buf, buf_dims);
	}else if(FM==FILTERMODE_MIN){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).min;
	}else if(FM==FILTERMODE_MAX){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).max;
	}*/
}

template<typename T, BoundaryMode BM>
__device__ inline float3 read3DGradPosGradInterpolated(const float3 position, const T* buf, const int3 buf_dims, const DataGrad3D<T> dataGradGrad, const float constantValue){
	// gradient of data gradient w.r.t. sampling position
	//weights for ceil, floor weights are (1-weights)
	const float3 weights = fracf(position);
	float3 weightsGrad = make_float3(0.0f);
	
	//calculate corner indices
	int3  ceilIdx = make_int3(ceilf(position));
	int3 floorIdx = make_int3(floorf(position));
	
	if (BM == BOUNDARY_BORDER){// const 0 outside domain
		if(!isInBounds3D(position+0.5f, buf_dims)){
			return weightsGrad;
		}
		int3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		int3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data = vmath::make_cudaFloat<T>(constantValue);
		//read
		const T fxfyfz = (floorValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxfyfz = ( ceilValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxcyfz = (floorValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxcyfz = ( ceilValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxfycz = (floorValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxfycz = ( ceilValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T fxcycz = (floorValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxcycz = ( ceilValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		//interpolation gradients w.r.t. position/weight
		
		// for reference:
		/* DataGrad3D<T> dataGrad;
		dataGrad.dx = lerp(
						lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
						lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
					weights.z); */
		// a=(cxfyfz-fxfyfz), b=..., t1=y, t2=z
		//weightsGrad.y += lerp((cxcyfz-fxcyfz)-(cxfyfz-fxfyfz),(cxcycz-fxcycz)-(cxfycz-fxfycz),weights.z)*vmath::sum(dataGradGrad.dx); // inner weight: back-front each
		//weightsGrad.z += lerp((cxfycz-fxfycz)-(cxfyfz-fxfyfz),(cxcycz-fxcycz)-(cxcyfz-fxcyfz),weights.y)*vmath::sum(dataGradGrad.dx); // outer weight: bottom-top each
		{
			//const float dataGradGradX = vmath::sum(dataGradGrad.dx);
			const T a = cxfyfz-fxfyfz;
			const T b = cxcyfz-fxcyfz;
			const T c = cxfycz-fxfycz;
			const T d = cxcycz-fxcycz;
			weightsGrad.y += vmath::sum(lerp(b-a,d-c,weights.z)*dataGradGrad.dx);
			weightsGrad.z += vmath::sum(lerp(c-a,d-b,weights.y)*dataGradGrad.dx);
		}
		/* dataGrad.dy = lerp(
						lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
						lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
					weights.z); */
		// a=(fxcyfz-fxfyfz), b=..., t1=x, t2=z
		// weightsGrad.x += lerp((cxcyfz-cxfyfz)-(fxcyfz-fxfyfz),(cxcycz-cxfycz)-(fxcycz-fxfycz),weights.z);
		// weightsGrad.z += lerp((fxcycz-fxfycz)-(fxcyfz-fxfyfz),(cxcycz-cxfycz)-(cxcyfz-cxfyfz),weights.x);
		{
			//const float dataGradGradY = vmath::sum(dataGradGrad.dy);
			const T a = fxcyfz-fxfyfz;
			const T b = cxcyfz-cxfyfz;
			const T c = fxcycz-fxfycz;
			const T d = cxcycz-cxfycz;
			weightsGrad.x += vmath::sum(lerp(b-a,d-c,weights.z)*dataGradGrad.dy);
			weightsGrad.z += vmath::sum(lerp(c-a,d-b,weights.x)*dataGradGrad.dy);
		}
		/* dataGrad.dz = lerp(
						lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
						lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
					weights.y); */
		// a=(fxfycz-fxfyfz), b=..., t1=x, t2=y
		// weightsGrad.x += lerp((cxfycz-cxfyfz)-(fxfycz-fxfyfz),(cxcycz-cxcyfz)-(fxcycz-fxcyfz),weights.y);
		// weightsGrad.y += lerp((fxcycz-fxcyfz)-(fxfycz-fxfyfz),(cxcycz-cxcyfz)-(cxfycz-cxfyfz),weights.x);
		{
			//const float dataGradGradZ = vmath::sum(dataGradGrad.dz);
			const T a = fxfycz-fxfyfz;
			const T b = cxfycz-cxfyfz;
			const T c = fxcycz-fxcyfz;
			const T d = cxcycz-cxcyfz;
			weightsGrad.x += vmath::sum(lerp(b-a,d-c,weights.y)*dataGradGrad.dz);
			weightsGrad.y += vmath::sum(lerp(c-a,d-b,weights.x)*dataGradGrad.dz);
		}
		
		return weightsGrad;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == BOUNDARY_CLAMP){
			ceilIdx = clamp(ceilIdx, make_int3(0), buf_dims -1);
			floorIdx = clamp(floorIdx, make_int3(0), buf_dims -1);
		}
		else if (BM == BOUNDARY_WRAP){//periodic
			ceilIdx = vmath::positivemod<int3, int3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<int3, int3>(floorIdx, buf_dims);
		}
		//read
		const T fxfyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T cxfyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T fxcyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T cxcyfz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T fxfycz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxfycz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T fxcycz = vectorIO::readVectorType3D<T, T, int32_t, int3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxcycz = vectorIO::readVectorType3D<T, T, int32_t, int3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		
		{
			const T a = cxfyfz-fxfyfz;
			const T b = cxcyfz-fxcyfz;
			const T c = cxfycz-fxfycz;
			const T d = cxcycz-fxcycz;
			weightsGrad.y += vmath::sum(lerp(b-a,d-c,weights.z)*dataGradGrad.dx);
			weightsGrad.z += vmath::sum(lerp(c-a,d-b,weights.y)*dataGradGrad.dx);
		}
		{
			const T a = fxcyfz-fxfyfz;
			const T b = cxcyfz-cxfyfz;
			const T c = fxcycz-fxfycz;
			const T d = cxcycz-cxfycz;
			weightsGrad.x += vmath::sum(lerp(b-a,d-c,weights.z)*dataGradGrad.dy);
			weightsGrad.z += vmath::sum(lerp(c-a,d-b,weights.x)*dataGradGrad.dy);
		}
		{
			const T a = fxfycz-fxfyfz;
			const T b = cxfycz-cxfyfz;
			const T c = fxcycz-fxcyfz;
			const T d = cxcycz-cxcyfz;
			weightsGrad.x += vmath::sum(lerp(b-a,d-c,weights.y)*dataGradGrad.dz);
			weightsGrad.y += vmath::sum(lerp(c-a,d-b,weights.x)*dataGradGrad.dz);
		}
		
		return weightsGrad;
	}
}
template<typename T, FilterMode FM, BoundaryMode BM>
__device__ inline float3 sample3DGradPosGrad(const float3 position, const T* buf, const int3 buf_dims, const DataGrad3D<T> dataGradGrad, const float constantValue = 0.f){
	if(FM==FILTERMODE_LINEAR){
		return read3DGradPosGradInterpolated<T, BM>(position - 0.5f, buf, buf_dims, dataGradGrad, constantValue);
	}/*else if(FM==FILTERMODE_NEAREST){
		scatterGrad3DNearest<T, BM, CountSamples, C>(position - 0.5f, buf, buf_dims);
	}else if(FM==FILTERMODE_MIN){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).min;
	}else if(FM==FILTERMODE_MAX){
		return read3DInterpolatedWithExtrema<T, BM>(position - 0.5f, buf, buf_dims).max;
	}*/
}


} //Sampling

#endif //_INCLUDE_SAMPLING_2