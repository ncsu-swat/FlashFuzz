#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
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
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::vector<int64_t> lhs_shape = {2, 3};
        std::vector<int64_t> rhs_shape = {3, 4};
        
        tensorflow::Tensor lhs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(lhs_shape));
        fillTensorWithDataByType(lhs_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor rhs_tensor(tensorflow::DT_QINT8, tensorflow::TensorShape(rhs_shape));
        fillTensorWithDataByType(rhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        
        uint8_t scales_type = data[offset % size];
        offset++;
        bool per_channel = (scales_type % 2 == 0);
        
        tensorflow::Tensor rhs_scales_tensor;
        if (per_channel) {
            rhs_scales_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({rhs_shape[1]}));
        } else {
            rhs_scales_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        }
        fillTensorWithDataByType(rhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor rhs_zero_points_tensor;
        if (per_channel) {
            rhs_zero_points_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({rhs_shape[1]}));
        } else {
            rhs_zero_points_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        }
        fillTensorWithDataByType(rhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        auto lhs_input = tensorflow::ops::Const(root, lhs_tensor);
        auto rhs_input = tensorflow::ops::Const(root, rhs_tensor);
        auto rhs_scales_input = tensorflow::ops::Const(root, rhs_scales_tensor);
        auto rhs_zero_points_input = tensorflow::ops::Const(root, rhs_zero_points_tensor);
        
        int rhs_quantization_min_val = -128;
        int rhs_quantization_max_val = 127;
        int rhs_quantization_axis = per_channel ? 1 : -1;
        
        tensorflow::NodeDef node_def;
        node_def.set_op("UniformQuantizedDotHybrid");
        node_def.set_name("uniform_quantized_dot_hybrid");
        
        tensorflow::NodeDefBuilder builder("uniform_quantized_dot_hybrid", "UniformQuantizedDotHybrid");
        builder.Input(lhs_input.node()->name(), 0, tensorflow::DT_FLOAT)
               .Input(rhs_input.node()->name(), 0, tensorflow::DT_QINT8)
               .Input(rhs_scales_input.node()->name(), 0, tensorflow::DT_FLOAT)
               .Input(rhs_zero_points_input.node()->name(), 0, tensorflow::DT_INT32)
               .Attr("T", tensorflow::DT_FLOAT)
               .Attr("rhs_quantization_min_val", rhs_quantization_min_val)
               .Attr("rhs_quantization_max_val", rhs_quantization_max_val)
               .Attr("rhs_quantization_axis", rhs_quantization_axis);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Output result = root.AddNode(node_def);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
