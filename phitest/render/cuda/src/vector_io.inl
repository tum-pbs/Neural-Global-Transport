
//TODO specify more as needed...

#define NUM_ELEMENTS(vector_type, element_type) (sizeof(vector_type)/sizeof(element_type))

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

namespace vectorIO{
	
	inline int32_t idxMod(const int32_t x, const int32_t m){
		//int32_t r = x%m;
		return x%m + (x<0 ? m : 0);
	}
	
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx4D(const C pos, const D dims){
		return dims.x*(dims.y*(dims.z*pos.w + pos.z) + pos.y) + pos.x; //dims.x*dims.y*pos.z + dims.x*pos.y + pos.x
	}
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx3D(const C pos, const D dims){
		return dims.x*(dims.y*pos.z + pos.y) + pos.x; //dims.x*dims.y*pos.z + dims.x*pos.y + pos.x
	}
	template<typename C, typename D, typename I>
	__device__ constexpr size_t flatIdx3DChannel(const C pos, const I channel, const D dims, const I channel_dim){
		return channel_dim*(dims.x*(dims.y*pos.z + pos.y) + pos.x) + channel;
	}
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx2D(const C pos, const D dims){
		return dims.x*pos.y + pos.x;
	}
	
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx3D(const C x, const C y, const C z, const D dims){
		return dims.x*(dims.y*z + y) + x;
	}
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx2D(const C x, const C y, const D dims){
		return dims.x*y + x;
	}
	
	template<typename C, typename D>
	__device__ constexpr C unflatIdx3D(const size_t flatIdx, const D dims){
		return C(flatIdx%dims.x, (flatIdx/dims.x)%dims.y, flatIdx/(dims.x*dims.y));
	}
	
	template<>
	__device__ inline glm::vec4 toVector<float4, glm::vec4>(const float4 v){
		return glm::vec4(v.x, v.y, v.z, v.w);
	}
	template<>
	__device__ inline glm::vec3 toVector<float4, glm::vec3>(const float4 v){
		return glm::vec3(v.x, v.y, v.z);
	}
	template<>
	__device__ inline glm::vec3 toVector<float3, glm::vec3>(const float3 v){
		return glm::vec3(v.x, v.y, v.z);
	}
	template<>
	__device__ inline float4 toVector<glm::vec4, float4>(const glm::vec4 v){
		return make_float4(v.x, v.y, v.z, v.w);
	}
	
	//read
	template<typename V, int VSIZE, typename T>
	__device__ inline V readVectorTypeAbs(const size_t idx, const T* buf){
		V v;
		#pragma unroll VSIZE
		for(int32_t i=0;i<VSIZE;++i){
			v[i] = buf[idx+i];
		}
		return v;
	}
	template<>
	__device__ inline float4 readVectorTypeAbs<float4, 1, float4>(const size_t idx, const float4* buf){
		return buf[idx];
	}
	template<>
	__device__ inline glm::vec4 readVectorTypeAbs<glm::vec4, 1, float4>(const size_t idx, const float4* buf){
		return toVector<float4,glm::vec4>(buf[idx]);
	}
	template<>
	__device__ inline float3 readVectorTypeAbs<float3, 1, float3>(const size_t idx, const float3* buf){
		return buf[idx];
	}
	template<>
	__device__ inline float2 readVectorTypeAbs<float2, 1, float2>(const size_t idx, const float2* buf){
		return buf[idx];
	}
	template<>
	__device__ inline float1 readVectorTypeAbs<float1, 1, float1>(const size_t idx, const float1* buf){
		return buf[idx];
	}
	template<>
	__device__ inline float readVectorTypeAbs<float, 1, float>(const size_t idx, const float* buf){
		return buf[idx];
	}
	
	template<typename V, typename T>
	__device__ inline V readVectorType(const size_t idx, const T* buf){
		return readVectorTypeAbs<V, NUM_ELEMENTS(V,T), T>(idx*NUM_ELEMENTS(V,T), buf);
	}
	template<typename V, typename T, typename I>
	__device__ inline V readVectorType3D(const I pos, const I dims, const T* buf){
		return readVectorType<V, T>(flatIdx3D<I, I>(pos, dims), buf);
	}
	template<typename V, typename T, typename C, typename I>
	__device__ inline V readVectorType3D(const C x, const C y, const C z, const I dims, const T* buf){
		return readVectorType<V, T>(flatIdx3D<C, I>(x,y,z, dims), buf);
	}
	//with bounds checking
	template<typename V, typename T, typename I>
	__device__ inline V readVectorType3DBounds(const I pos, const I dims, const T* buf){
		if(0<pos.x && pos.x<dims.x && 0<pos.y && pos.y<dims.y && 0<pos.z && pos.z<dims.z){
			return readVectorType3D<V,T,I>(pos,dims,buf);
		}else{
			return vmath::make_cudaFloat<T>(0.f);
		}
	}
	template<typename V, typename T, typename C, typename I>
	__device__ inline V readVectorType3DBounds(const C x, const C y, const C z, const I dims, const T* buf){
		if(0<x && x<dims.x && 0<y && y<dims.y && 0<z && z<dims.z){
			return readVectorType3D<V,T,I>(x,y,z,dims,buf);
		}else{
			return vmath::make_cudaFloat<T>(0.f);
		}
	}
	
	
	//write
	template<typename V, int32_t VSIZE, typename T>
	__device__ inline void writeVectorTypeAbs(const V v, const size_t idx, T* buf){
		#pragma unroll VSIZE
		for(int32_t i=0;i<VSIZE;++i){
			buf[idx+i] = v[i];
		}
	}
	//cuda vector types...
	template<>
	__device__ inline void writeVectorTypeAbs<float4, 1, float4>(const float4 v, const size_t idx, float4* buf){
		buf[idx] = v;
	}
	template<>
	__device__ inline void writeVectorTypeAbs<glm::vec4, 1, float4>(const glm::vec4 v, const size_t idx, float4* buf){
		buf[idx] = toVector<glm::vec4,float4>(v);
	}
	template<>
	__device__ inline void writeVectorTypeAbs<float4, 4, float>(const float4 v, const size_t idx, float* buf){
		buf[idx] = v.x;
		buf[idx+1] = v.y;
		buf[idx+2] = v.z;
		buf[idx+3] = v.w;
	}
	template<>
	__device__ inline void writeVectorTypeAbs<float2, 1, float2>(const float2 v, const size_t idx, float2* buf){
		buf[idx] = v;
	}
	template<>
	__device__ inline void writeVectorTypeAbs<float1, 1, float1>(const float1 v, const size_t idx, float1* buf){
		buf[idx] = v;
	}
	template<>
	__device__ inline void writeVectorTypeAbs<float1, 1, float>(const float1 v, const size_t idx, float* buf){
		buf[idx] = v.x;
	}
	template<>
	__device__ inline void writeVectorTypeAbs<float, 1, float>(const float v, const size_t idx, float* buf){
		buf[idx] = v;
	}
	
	template<typename V, typename T>
	__device__ inline void writeVectorType(const V v, const size_t idx, T* buf){
		writeVectorTypeAbs<V, NUM_ELEMENTS(V,T), T>(v, idx*NUM_ELEMENTS(V,T), buf);
	}
	template<typename V, typename T, typename I>
	__device__ inline void writeVectorType3D(const V v, const I pos, const I dims, T* buf){
		writeVectorType<V, T>(v, flatIdx3D<I, I>(pos, dims), buf);
	}
	
// add
	
	template<typename V, int32_t VSIZE, typename T>
	__device__ inline void addVectorTypeAbs(const V v, const size_t idx, T* buf){
		#pragma unroll VSIZE
		for(int32_t i=0;i<VSIZE;++i){
			buf[idx+i] += v[i];
		}
	}
	//cuda vector types...
	template<>
	__device__ inline void addVectorTypeAbs<float4, 1, float4>(const float4 v, const size_t idx, float4* buf){
		buf[idx] += v;
	}
	template<>
	__device__ inline void addVectorTypeAbs<glm::vec4, 1, float4>(const glm::vec4 v, const size_t idx, float4* buf){
		buf[idx] += toVector<glm::vec4,float4>(v);
	}
	template<>
	__device__ inline void addVectorTypeAbs<float4, 4, float>(const float4 v, const size_t idx, float* buf){
		buf[idx] += v.x;
		buf[idx+1] += v.y;
		buf[idx+2] += v.z;
		buf[idx+3] += v.w;
	}
	template<>
	__device__ inline void addVectorTypeAbs<float2, 1, float2>(const float2 v, const size_t idx, float2* buf){
		buf[idx] += v;
	}
	template<>
	__device__ inline void addVectorTypeAbs<float1, 1, float1>(const float1 v, const size_t idx, float1* buf){
		buf[idx] += v;
	}
	template<>
	__device__ inline void addVectorTypeAbs<float1, 1, float>(const float1 v, const size_t idx, float* buf){
		buf[idx] += v.x;
	}
	template<>
	__device__ inline void addVectorTypeAbs<float, 1, float>(const float v, const size_t idx, float* buf){
		buf[idx] += v;
	}
	template<typename V, typename T>
	__device__ inline void addVectorType(const V v, const size_t idx, T* buf){
		addVectorTypeAbs<V, NUM_ELEMENTS(V,T), T>(v, idx*NUM_ELEMENTS(V,T), buf);
	}
	template<typename V, typename T, typename I>
	__device__ inline void addVectorType3D(const V v, const I pos, const I dims, T* buf){
		addVectorType<V, T>(v, flatIdx3D<I, I>(pos, dims), buf);
	}
	
// atomic add
	
	template<typename T>
	__device__ inline void atomicAddVectorTypeAbs(const T v, const size_t idx, T* buf);
	template<>
	__device__ inline void atomicAddVectorTypeAbs(const float1 v, const size_t idx, float1* buf){
		float * buf_raw = reinterpret_cast<float*>(buf + idx);
		atomicAdd(buf_raw, v.x);
	}
	template<>
	__device__ inline void atomicAddVectorTypeAbs(const float2 v, const size_t idx, float2* buf){
		float * buf_raw = reinterpret_cast<float*>(buf + idx);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
	}
	template<>
	__device__ inline void atomicAddVectorTypeAbs(const float4 v, const size_t idx, float4* buf){
		float * buf_raw = reinterpret_cast<float*>(buf + idx);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
		atomicAdd(buf_raw +2, v.z);
		atomicAdd(buf_raw +3, v.w);
	}
	
	template<typename T>
	__device__ inline void atomicAddVectorType3D(const T v, const int3 pos, const int3 dims, T* buf);
	template<>
	__device__ inline void atomicAddVectorType3D(const float1 v, const int3 pos, const int3 dims, float1* buf){
		const size_t offset = flatIdx3D(pos, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
	}
	template<>
	__device__ inline void atomicAddVectorType3D(const float2 v, const int3 pos, const int3 dims, float2* buf){
		const size_t offset = flatIdx3D(pos, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
	}
	template<>
	__device__ inline void atomicAddVectorType3D(const float4 v, const int3 pos, const int3 dims, float4* buf){
		const size_t offset = flatIdx3D(pos, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
		atomicAdd(buf_raw +2, v.z);
		atomicAdd(buf_raw +3, v.w);
	}
	template<typename T>
	__device__ inline void atomicAddVectorType3D(const T v, const int32_t x, const int32_t y, const int32_t z, const int3 dims, T* buf);
	template<>
	__device__ inline void atomicAddVectorType3D(const float1 v, const int32_t x, const int32_t y, const int32_t z, const int3 dims, float1* buf){
		const size_t offset = flatIdx3D(x,y,z, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
	}
	template<>
	__device__ inline void atomicAddVectorType3D(const float2 v, const int32_t x, const int32_t y, const int32_t z, const int3 dims, float2* buf){
		const size_t offset = flatIdx3D(x,y,z, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
	}
	template<>
	__device__ inline void atomicAddVectorType3D(const float4 v, const int32_t x, const int32_t y, const int32_t z, const int3 dims, float4* buf){
		const size_t offset = flatIdx3D(x,y,z, dims);
		float * buf_raw = reinterpret_cast<float*>(buf + offset);
		atomicAdd(buf_raw, v.x);
		atomicAdd(buf_raw +1, v.y);
		atomicAdd(buf_raw +2, v.z);
		atomicAdd(buf_raw +3, v.w);
	}
	
	//performance test
	/*
	* pass globalIdx to avoid recomputation
	*/
	/* reading properly aligned 1,2,4,8,16 byte structs is automatically coalesced, so no need for this
	template<typename V, typename T, typename I>
	__device__ inline void writeVector4Type3DCoalescedX(const V data, const I globalIdx, const I dimensions, T* buf){
		__shared__ T s_vectorBuffer[BLOCK_SIZE_Z*BLOCK_SIZE_Y][BLOCK_SIZE_X*4+1]; //+1 for better bank access?
		const int32_t blockThreadIdxZY = BLOCK_SIZE_Y*threadIdx.z + threadIdx.y;
		const int32_t blockThreadIdxX = threadIdx.x*4;
		//#pragma unroll NUM_ELEMENTS(V,T)
		s_vectorBuffer[blockThreadIdxZY][blockThreadIdxX]   = data.x;
		s_vectorBuffer[blockThreadIdxZY][blockThreadIdxX+1] = data.y;
		s_vectorBuffer[blockThreadIdxZY][blockThreadIdxX+2] = data.z;
		s_vectorBuffer[blockThreadIdxZY][blockThreadIdxX+3] = data.w;
		//start offset in global row in vectors (V)
		const int32_t offsetX = blockIdx.x*BLOCK_SIZE_X;
		//start of row block in global memory in elements (T)
		const int32_t globalIdxZYx = 4* (offsetX + dimensions.x*(dimensions.y*globalIdx.z + globalIdx.y));
		//stride is BLOCK_SIZE_X, should be at least 8 when writing 4byte components
		__syncthreads();
		#pragma unroll
		for(int32_t i=0;i<4;++i){
			const int32_t currentThreadOffset = BLOCK_SIZE_X*i + threadIdx.x;
			//clamp bounds
			if(globalIdx.y<dimensions.y && globalIdx.z<dimensions.z && (offsetX+(currentThreadOffset)/4)<dimensions.x){
				buf[globalIdxZYx + currentThreadOffset] = s_vectorBuffer[blockThreadIdxZY][currentThreadOffset];
				//buf[globalIdxZYx + offsetX + BLOCK_SIZE_X*i + threadIdx.x] = s_vectorBuffer[BLOCK_SIZE_X*i + threadIdx.x];
			}
		}
	}
	template<typename V, typename T, typename I>
	__device__ inline void writeVector4Type3DCoalescedXShuffle(const V data, const I globalIdx, const I dimensions, T* buf){
		
		int32_t shuffleIdx=threadIdx.x%4;
		T shuffle_data = {data.x, data.y, data.z, data.w};
		//shuffle_data[shuffleIdx] = data[shuffle_idx];
		for(int32_t i=1;i<4;++i){
			shuffle_data[(shuffleIdx+i)%4] = __shfl_down_sync(0xffffffff, data[idxMod(shuffleIdx-i, 4)], i, 4);
		}
		//start offset in global row in vectors (V)
		const int32_t offsetX = blockIdx.x*BLOCK_SIZE_X;
		//start of row block in global memory in elements (T)
		const int32_t globalIdxZYx = 4* (offsetX + dimensions.x*(dimensions.y*globalIdx.z + globalIdx.y));
		//stride is BLOCK_SIZE_X, should be at least 8 when writing 4byte components
		#pragma unroll
		for(int32_t i=0;i<4;++i){
			const int32_t currentThreadOffset = BLOCK_SIZE_X*i + threadIdx.x;
			//clamp bounds
			if(globalIdx.y<dimensions.y && globalIdx.z<dimensions.z && (offsetX+(currentThreadOffset)/4)<dimensions.x){
				buf[globalIdxZYx + currentThreadOffset] = s_vectorBuffer[blockThreadIdxZY][currentThreadOffset];
				//buf[globalIdxZYx + offsetX + BLOCK_SIZE_X*i + threadIdx.x] = s_vectorBuffer[BLOCK_SIZE_X*i + threadIdx.x];
			}
		}
	}
	template<>
	__device__ inline void writeVector4Type3DCoalescedX<float1, float, glm::ivec3>(const float1 data, const glm::ivec3 globalIdx, const glm::ivec3 dimensions, float* buf){
		writeVectorType3D<float1, float, glm::ivec3>(data, globalIdx, dimensions, buf);
	}//*/
}