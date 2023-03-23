#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/array_ops.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <string>
//#define LOGGING
#include "advect.hpp"
#include "render_errors.hpp"

#ifdef LOGGING
#define MYLOG(msg) std::cout << __FILE__ << "[" << __LINE__ << "]: " << msg << std::endl
#define LOG_PRINTF(msg) printf(msg)
#else
#define MYLOG(msg)
#define LOG_PRINTF(msg)
#endif
using namespace tensorflow;

REGISTER_OP("AdvectGridSemiLangrange")
	.Input("input: float") // NDHWC
	.Input("velocity_centered: float") //VDHWC, defines output shape
	.Attr("timestep: float = 0.0")
	//.Attr("interpolation: {'NEAREST', 'LINEAR', 'MIN', 'MAX'} = 'LINEAR'")
	.Attr("order: int >= 1 = 1")
	.Attr("clamp_extrema: bool = true")
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	//.Attr("mipmapping: {'NONE', 'NEAREST', 'LINEAR'} = 'NONE'")
	//.Attr("num_mipmaps: int = 0")
	//.Attr("mip_bias: float = 0.0")
	.Attr("separate_velocity_batch: bool = true")
	//.Attr("relative_coords: bool = true")
	//.Attr("normalized_coords: bool = false")
	.Output("output: float") // NVDHWC
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle channel;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 3, &channel));
		::tensorflow::shape_inference::ShapeHandle outShape;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 0, 3, &outShape));
		TF_RETURN_IF_ERROR(c->Concatenate(outShape, channel, &outShape));
		c->set_output(0, outShape);
		return Status::OK();
	});


// the gradient op
REGISTER_OP("AdvectGridSemiLangrangeGrad")
	.Input("input: float") // NDHWC
	.Input("output_grad: float") // NVDHWC
	.Input("velocity_centered: float") //VDHWC
	.Attr("timestep: float = 0.0")
	//.Attr("interpolation: {'NEAREST', 'LINEAR', 'MIN', 'MAX'} = 'LINEAR'")
	.Attr("order: int >= 1 = 1")
	.Attr("clamp_extrema: bool = true")
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	//.Attr("mipmapping: {'NONE'} = 'NONE'") //, 'NEAREST', 'LINEAR'. currently not supported
	//.Attr("num_mipmaps: int = 0")
	//.Attr("mip_bias: float = 0.0")
	.Attr("separate_velocity_batch: bool = true")
	//.Attr("relative_coords: bool = true")
	//.Attr("normalized_coords: bool = false")
	.Output("input_grad: float")
	.Output("velocity_grad: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		c->set_output(0, c->input(0));
		c->set_output(1, c->input(2));
		return Status::OK();
	});
	
#if GOOGLE_CUDA
#define DECLARE_GPU_SPEC(T, C) \
	template<> \
	void AdvectGridKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const T* input, const long long int* input_shape, \
		const float* velocity, T* tmp_fwd, T* tmp_min, T* tmp_max, \
		const float timestep, const int32_t order, const Sampling::BoundaryMode boundaryMode, \
		const bool revertExtrema, const int32_t numVelocities, const bool globalSampling, \
		T* output, const long long int* output_shape); \
	extern template struct AdvectGridKernel<GPUDevice, T, C>;
DECLARE_GPU_SPEC(float, 1)
DECLARE_GPU_SPEC(float, 2)
DECLARE_GPU_SPEC(float, 4)
#undef DECLARE_GPU_SPEC
#endif


template<typename Device, typename T>
class AdvectGridSemiLangrangeOp : public OpKernel{
public:
	explicit AdvectGridSemiLangrangeOp(OpKernelConstruction *context) : OpKernel(context){
		
		/*
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		*/
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_boundaryMode = Sampling::BOUNDARY_CLAMP;
		else if(s_boundary.compare("WRAP")==0) m_boundaryMode = Sampling::BOUNDARY_WRAP;
		else if(s_boundary.compare("BORDER")==0) m_boundaryMode = Sampling::BOUNDARY_BORDER;
		else throw errors::InvalidArgument("Invalid boundary argument.");
		
		OP_REQUIRES_OK(context, context->GetAttr("timestep", &m_timestep));
		OP_REQUIRES_OK(context, context->GetAttr("order", &m_order));
		OP_REQUIRES(context, m_order==1 || m_order==2,
			errors::InvalidArgument("Only first (SemiLangrange) and second (MacCormack) order advection supported."));
		OP_REQUIRES_OK(context, context->GetAttr("clamp_extrema", &m_clampExtrema));
		
		/*
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		*/
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_velocity_batch", &m_globalSampling));
		
		/*
		OP_REQUIRES_OK(context, context->GetAttr("relative_coords", &m_relativeCoords));
		OP_REQUIRES_OK(context, context->GetAttr("normalized_coords", &m_normalizedCoords));
		if(m_normalizedCoords){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset must be 0.5 when using normalized coordinates."));
		}
		*/
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Sample transform op kernel start");
		
		const Tensor& tensor_input = context->input(0);
		const Tensor& tensor_velocity = context->input(1);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = tensor_input.shape();
		OP_REQUIRES(context, tensor_input.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		MYLOG("Check velocity");
		TensorShape velocity_shape = tensor_velocity.shape();
		OP_REQUIRES(context, velocity_shape.dims()==5 && velocity_shape.dim_size(4)==3,
			errors::InvalidArgument("Invalid velocity shape (VDHWC) with C=3:", velocity_shape.DebugString()));
		int32_t numVelocities = velocity_shape.dim_size(0);
		if(m_order==2){
			OP_REQUIRES(context, 
				velocity_shape.dim_size(1)==input_shape.dim_size(1) &&
				velocity_shape.dim_size(2)==input_shape.dim_size(2) &&
				velocity_shape.dim_size(3)==input_shape.dim_size(3),
				errors::InvalidArgument("Spatial dimensions of input and velocity must match when using 2nd order advection:", input_shape.DebugString(), velocity_shape.DebugString()));
		}
		
		//create output shape
		TensorShape output_shape;
		{
			MYLOG("Create output shape");
			output_shape.AddDim(velocity_shape.dim_size(1));
			output_shape.AddDim(velocity_shape.dim_size(2));
			output_shape.AddDim(velocity_shape.dim_size(3));
			output_shape.InsertDim(0,batch);
			output_shape.AddDim(channel);
			MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
			
			if(!m_globalSampling){
				OP_REQUIRES(context, numVelocities==batch, errors::InvalidArgument("velocity batch must match data batch when not using global sampling."));
				output_shape.InsertDim(1,1);
			}else{
				output_shape.InsertDim(1,numVelocities);
			}
		}
		
		
		//allocate outout
		Tensor* tensor_output = nullptr;
		MYLOG("Allocate output");
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &tensor_output));
		MYLOG("Check allocated output\n");
		MYLOG("Allocated output size: " << tensor_output->flat<T>().size() << " - " << tensor_output->NumElements());
		
		//allocate temporary grids
		TensorShape tmp_shape;
		{
			tmp_shape.AddDim(1);
			tmp_shape.AddDim(velocity_shape.dim_size(1));
			tmp_shape.AddDim(velocity_shape.dim_size(2));
			tmp_shape.AddDim(velocity_shape.dim_size(3));
			tmp_shape.AddDim(channel);
		}
		Tensor tensor_fwd;
		T* p_tensor_fwd = nullptr;
		Tensor tensor_min;
		T* p_tensor_min = nullptr;
		Tensor tensor_max;
		T* p_tensor_max = nullptr;
		if(m_order==2){
			OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, tmp_shape, &tensor_fwd));
			p_tensor_fwd = tensor_fwd.flat<T>().data();
			if(m_clampExtrema){
				OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, tmp_shape, &tensor_min));
				p_tensor_min = tensor_min.flat<T>().data();
				OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, tmp_shape, &tensor_max));
				p_tensor_max = tensor_max.flat<T>().data();
			}
		}
		
		
		
		//TODO handle arbitrary amount of channel
		// - move channel dimension outwards (to batch) and handle only 1-channel case internally. would also handle batches
		//   this would benefit from NCHW layout, otherwise have to transpose
		//   or just require NCHW as input format (or NHWC with up to 4 channel, the rest has to be packed in N) and let tensorflow/user handle the conversion...
		// - split into up-to-4-channel partitions. might be faster? but is harder to handle
		
		MYLOG("Resample");
		switch(channel){
		case 1:
			AdvectGridKernel<GPUDevice, T, 1>()(tensor_input.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<T>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_boundaryMode, m_clampExtrema,
					numVelocities, m_globalSampling,
					tensor_output->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		case 2:
			AdvectGridKernel<GPUDevice, T, 2>()(tensor_input.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<T>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_boundaryMode, m_clampExtrema,
					numVelocities, m_globalSampling,
					tensor_output->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		case 4:
			AdvectGridKernel<GPUDevice, T, 4>()(tensor_input.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<T>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_boundaryMode, m_clampExtrema,
					numVelocities, m_globalSampling,
					tensor_output->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel data supported."));
		}
		
		MYLOG("Kernel done");
	}
private:
	bool m_globalSampling;
	float m_timestep;
	int32_t m_order;
	bool m_clampExtrema;
	Sampling::BoundaryMode m_boundaryMode;
	//bool m_relativeCoords;
	//bool m_normalizedCoords;
	
};

#if 0
	
#if GOOGLE_CUDA
#define DECLARE_GPU_SPEC(T, C) \
	template<> \
	void AdvectGridGradKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const void* input, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		uint8_t* mipAtlas, \
		const Sampling::CoordinateMode coordinateMode, \
		const Sampling::SamplerSettings, const bool globalSampling, \
		void* output, const long long int* output_shape \
		); \
	extern template struct AdvectGridGradKernel<GPUDevice, T, C>;
DECLARE_GPU_SPEC(float, 1)
DECLARE_GPU_SPEC(float, 2)
DECLARE_GPU_SPEC(float, 4)
#undef DECLARE_GPU_SPEC
#endif


template<typename Device, typename T>
class AdvectGridSemiLangrangeGradOp : public OpKernel{
public:
	explicit AdvectGridSemiLangrangeGradOp(OpKernelConstruction *context) : OpKernel(context){
		
		/*
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		*/
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::CLAMP;
		else if(s_boundary.compare("WRAP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::WRAP;
		else if(s_boundary.compare("MIRROR")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::MIRROR;
		else throw errors::InvalidArgument("Invalid boundary argument.");
		
		OP_REQUIRES_OK(context, context->GetAttr("timestep", &m_timestep));
		OP_REQUIRES_OK(context, context->GetAttr("order", &m_order));
		OP_REQUIRES(context, m_order==1 || m_order==2,
			errors::InvalidArgument("Only first (SemiLangrange) and second (MacCormack) order advection supported."));
		OP_REQUIRES_OK(context, context->GetAttr("clamp_extrema", &m_clampExtrema));
		
		/*
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		*/
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_velocity_batch", &m_globalSampling));
		
		/*
		OP_REQUIRES_OK(context, context->GetAttr("relative_coords", &m_relativeCoords));
		OP_REQUIRES_OK(context, context->GetAttr("normalized_coords", &m_normalizedCoords));
		if(m_normalizedCoords){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset must be 0.5 when using normalized coordinates."));
		}
		*/
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Sample transform op kernel start");
		
		const Tensor& tensor_input = context->input(0);
		const Tensor& tensor_grad_output = context->input(1);
		const Tensor& tensor_velocity = context->input(2);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = tensor_input.shape();
		OP_REQUIRES(context, tensor_input.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		MYLOG("Check velocity");
		TensorShape velocity_shape = tensor_velocity.shape();
		OP_REQUIRES(context, velocity_shape.dims()==5 && velocity_shape.dim_size(4)==3,
			errors::InvalidArgument("Invalid velocity shape (VDHWC) with C=3:", velocity_shape.DebugString()));
		int32_t numVelocities = velocity_shape.dim_size(0);
		if(m_order==2){
			OP_REQUIRES(context, 
				velocity_shape.dim_size(1)==input_shape.dim_size(1) &&
				velocity_shape.dim_size(2)==input_shape.dim_size(2) &&
				velocity_shape.dim_size(3)==input_shape.dim_size(3),
				errors::InvalidArgument("Spatial dimensions of input and velocity must match when using 2nd order advection:", input_shape.DebugString(), velocity_shape.DebugString()));
		}
		
		//check output shape
		TensorShape output_shape;
		{
			MYLOG("Create output shape");
			output_shape.AddDim(velocity_shape.dim_size(1));
			output_shape.AddDim(velocity_shape.dim_size(2));
			output_shape.AddDim(velocity_shape.dim_size(3));
			output_shape.InsertDim(0,batch);
			output_shape.AddDim(channel);
			MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
			
			if(!m_globalSampling){
				OP_REQUIRES(context, numVelocities==batch, errors::InvalidArgument("velocity batch must match data batch when not using global sampling."));
				output_shape.InsertDim(1,1);
			}else{
				output_shape.InsertDim(1,numVelocities);
			}
		}
		
		
		//allocate outout
		Tensor* tensor_output = nullptr;
		MYLOG("Allocate output");
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &tensor_output));
		MYLOG("Check allocated output\n");
		MYLOG("Allocated output size: " << tensor_output->flat<T>().size() << " - " << tensor_output->NumElements());
		
		//allocate temporary grids
		TensorShape tmp_shape;
		{
			tmp_shape.AddDim(1);
			tmp_shape.AddDim(velocity_shape.dim_size(1));
			tmp_shape.AddDim(velocity_shape.dim_size(2));
			tmp_shape.AddDim(velocity_shape.dim_size(3));
			tmp_shape.AddDim(channel);
		}
		Tensor tensor_fwd;
		T* p_tensor_fwd = nullptr;
		Tensor tensor_fwd_grad;
		T* p_tensor_fwd_grad = nullptr;
		Tensor tensor_min;
		T* p_tensor_min = nullptr;
		Tensor tensor_max;
		T* p_tensor_max = nullptr;
		if(m_order==2){
			OP_REQUIRES_OK(context, context->allocate_temp(T, tmp_shape, &tensor_fwd));
			p_tensor_fwd = tensor_fwd.flat<T>().data();
			OP_REQUIRES_OK(context, context->allocate_temp(T, tmp_shape, &tensor_fwd_grad));
			p_tensor_fwd_grad = tensor_fwd_grad.flat<T>().data();
			if(m_clampExtrema){
				OP_REQUIRES_OK(context, context->allocate_temp(T, tmp_shape, &tensor_min));
				p_tensor_min = tensor_min.flat<T>().data();
				OP_REQUIRES_OK(context, context->allocate_temp(T, tmp_shape, &tensor_max));
				p_tensor_max = tensor_max.flat<T>().data();
			}
		}
		
		
		
		//TODO handle arbitrary amount of channel
		// - move channel dimension outwards (to batch) and handle only 1-channel case internally. would also handle batches
		//   this would benefit from NCHW layout, otherwise have to transpose
		//   or just require NCHW as input format (or NHWC with up to 4 channel, the rest has to be packed in N) and let tensorflow/user handle the conversion...
		// - split into up-to-4-channel partitions. might be faster? but is harder to handle
		
		MYLOG("Resample");
		switch(channel){
		case 1:
			AdvectGridGradKernel<GPUDevice, T, 1>()(tensor_input.flat<float>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<float>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_clampExtrema
					numVelocities, m_globalSampling,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		case 2:
			AdvectGridGradKernel<GPUDevice, T, 2>()(tensor_input.flat<float>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<float>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_clampExtrema
					numVelocities, m_globalSampling,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		case 4:
			AdvectGridGradKernel<GPUDevice, T, 4>()(tensor_input.flat<float>().data(), input_shape.dim_sizes().data(),
					tensor_velocity.flat<float>().data(), p_tensor_fwd, p_tensor_min, p_tensor_max,
					m_timestep, m_order, m_clampExtrema
					numVelocities, m_globalSampling,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel data supported."));
		}
		
		MYLOG("Kernel done");
	}
private:
	bool m_globalSampling;
	float m_timestep;
	int32_t m_order;
	bool m_clampExtrema;
	//bool m_relativeCoords;
	//bool m_normalizedCoords;
	
};

#endif //0
