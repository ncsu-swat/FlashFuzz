#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/kernels/scatter_functor.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
        case 5:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    if (rank == 0) {
        return {};
    }

    std::vector<int64_t> shape;
    shape.reserve(rank);
    const auto sizeof_dim = sizeof(int64_t);

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof_dim <= total_size) {
            int64_t dim_val;
            std::memcpy(&dim_val, data + offset, sizeof_dim);
            offset += sizeof_dim;
            
            dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                    static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) %
                                        static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));

            shape.push_back(dim_val);
        } else {
             shape.push_back(1);
        }
    }

    return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
  auto flat = tensor.flat<T>();
  const size_t num_elements = flat.size();
  const size_t element_size = sizeof(T);

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset + element_size <= total_size) {
      T value;
      std::memcpy(&value, data + offset, element_size);
      offset += element_size;
      flat(i) = value;
    } else {
      flat(i) = T{};
    }
  }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case tensorflow::DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType ref_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t ref_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        
        bool use_locking = (data[offset++] % 2) == 1;

        std::vector<int64_t> ref_shape = parseShape(data, offset, size, ref_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);

        if (ref_shape.empty() || indices_shape.empty()) {
            return 0;
        }

        std::vector<int64_t> updates_shape = indices_shape;
        if (ref_shape.size() > 1) {
            for (size_t i = 1; i < ref_shape.size(); ++i) {
                updates_shape.push_back(ref_shape[i]);
            }
        }

        tensorflow::TensorShape ref_tensor_shape;
        tensorflow::TensorShape indices_tensor_shape;
        tensorflow::TensorShape updates_tensor_shape;
        
        if (!tensorflow::TensorShape::BuildTensorShape(ref_shape, &ref_tensor_shape).ok()) {
            return 0;
        }
        if (!tensorflow::TensorShape::BuildTensorShape(indices_shape, &indices_tensor_shape).ok()) {
            return 0;
        }
        if (!tensorflow::TensorShape::BuildTensorShape(updates_shape, &updates_tensor_shape).ok()) {
            return 0;
        }

        tensorflow::Tensor ref_tensor(ref_dtype, ref_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        tensorflow::Tensor updates_tensor(ref_dtype, updates_tensor_shape);

        fillTensorWithDataByType(ref_tensor, ref_dtype, data, offset, size);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        fillTensorWithDataByType(updates_tensor, ref_dtype, data, offset, size);

        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int32_t>(ref_shape[0]);
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % ref_shape[0];
            }
        }

        std::cout << "ref_tensor shape: ";
        for (int i = 0; i < ref_tensor.dims(); ++i) {
            std::cout << ref_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.dims(); ++i) {
            std::cout << indices_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "updates_tensor shape: ";
        for (int i = 0; i < updates_tensor.dims(); ++i) {
            std::cout << updates_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "use_locking: " << use_locking << std::endl;

        tensorflow::NodeDef node_def;
        tensorflow::NodeDefBuilder builder("scatter_max", "ScatterMax");
        builder.Input("ref", 0, ref_dtype)
               .Input("indices", 0, indices_dtype)
               .Input("updates", 0, ref_dtype)
               .Attr("T", ref_dtype)
               .Attr("Tindices", indices_dtype)
               .Attr("use_locking", use_locking);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            std::cout << "NodeDef build failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::Device> device;
        tensorflow::DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0", &device);
        
        if (!device) {
            std::cout << "Failed to create CPU device" << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::OpKernel> kernel;
        status = tensorflow::CreateOpKernel(device->device_type(), device.get(), device->GetAllocator(tensorflow::AllocatorAttributes()), node_def, TF_GRAPH_DEF_VERSION, &kernel);
        
        if (!status.ok()) {
            std::cout << "OpKernel creation failed: " << status.ToString() << std::endl;
            return 0;
        }

        tensorflow::OpKernelContext::Params params;
        params.device = device.get();
        params.frame_iter = tensorflow::FrameAndIter(0, 0);
        params.inputs = new tensorflow::TensorValue[3];
        params.inputs[0].tensor = &ref_tensor;
        params.inputs[1].tensor = &indices_tensor;
        params.inputs[2].tensor = &updates_tensor;
        params.op_kernel = kernel.get();
        
        tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> inputs;
        inputs.push_back(tensorflow::TensorValue(&ref_tensor));
        inputs.push_back(tensorflow::TensorValue(&indices_tensor));
        inputs.push_back(tensorflow::TensorValue(&updates_tensor));
        
        tensorflow::OpKernelContext context(&params);
        
        kernel->Compute(&context);
        
        if (!context.status().ok()) {
            std::cout << "Kernel compute failed: " << context.status().ToString() << std::endl;
        } else {
            std::cout << "ScatterMax operation completed successfully" << std::endl;
        }

        delete[] params.inputs;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}