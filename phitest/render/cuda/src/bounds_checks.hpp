
#pragma once

#ifndef _INCLUDE_BOUNDSCHECKS
#define _INCLUDE_BOUNDSCHECKS

//for bounds checking
#define CHECK_BOUNDS_SV3S(l, c1, v, c2, u) l c1 v.x && v.x c2 u && l c1 v.y && v.y c2 u && l c1 v.z && v.z c2 u
#define CHECK_BOUNDS_SV3V3(l, c1, v, c2, u) l c1 v.x && v.x c2 u.x && l c1 v.y && v.y c2 u.y && l c1 v.z && v.z c2 u.z
#define CHECK_BOUNDS_V3V3V3(l, c1, v, c2, u) l.x c1 v.x && v.x c2 u.x && l.x c1 v.y && v.y c2 u.y && l.x c1 v.z && v.z c2 u.z
#define CHECK_BOUND_SV3(v1, c, v2) v1 c v2.x && v1 c v2.y && v1 c v2.z
#define CHECK_BOUND_V3S(v1, c, v2) v1.x c v2 && v1.y c v2 && v1.z c v2
#define CHECK_BOUND_V3V3(v1, c, v2) v1.x c v2.x && v1.y c v2.y && v1 c v2.z
/*
template<typename T, typename D>
__device__ inline bool isInDimensions(const T position, const D dimensions){
	return (position.x < dimensions.x && position.y < dimensions.y && position.z < dimensions.z);
}
template<typename T, typename D>
__device__ inline bool isInDimensions(const T x, const T y, const T z, const D dimensions){
	return (x < dimensions.x && y < dimensions.y && z < dimensions.z);
}
template<typename V3>
__device__ inline bool isNonNegative(const V3 position){
	//return (position.x >=0 && position.y >=0 && position.z >=0);
	return CHECK_BOUND_SV3(0, <=, position);
}
__device__ inline bool isNonNegative(const glm::vec3 position){
	//return (position.x >=0 && position.y >=0 && position.z >=0);
	return CHECK_BOUND_SV3(0.f, <=, position);
}
*/
/*
inline __device__ __host__ bool checkBounds3D(const int3 idx, const int3 dims){
	return CHECK_BOUNDS_SV3V3(0, <=, idx, <, dims);
}
inline __device__ __host__ bool checkBounds3D(const float3 idx, const int3 dims){
	return CHECK_BOUNDS_SV3V3(0, <=, idx, <, dims);
}
inline __device__ __host__ bool checkUpperBounds3D(const int3 idx, const int3 dims){
	return CHECK_BOUNDS_V3V3(idx, <, dims);
}
*/

#endif //_INCLUDE_BOUNDSCHECKS