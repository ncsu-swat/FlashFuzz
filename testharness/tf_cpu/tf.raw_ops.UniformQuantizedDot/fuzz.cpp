#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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
    case tensorflow::DT_QINT32:
      fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::vector<int64_t> lhs_shape = {2, 3};
        std::vector<int64_t> rhs_shape = {3, 4};
        
        tensorflow::Tensor lhs_tensor(tensorflow::DT_QINT8, tensorflow::TensorShape(lhs_shape));
        tensorflow::Tensor rhs_tensor(tensorflow::DT_QINT8, tensorflow::TensorShape(rhs_shape));
        
        fillTensorWithDataByType(lhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        fillTensorWithDataByType(rhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        
        tensorflow::Tensor lhs_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor lhs_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor rhs_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor rhs_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor output_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor output_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(lhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(lhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(rhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(rhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(output_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(output_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        auto lhs_const = tensorflow::ops::Const(root, lhs_tensor);
        auto rhs_const = tensorflow::ops::Const(root, rhs_tensor);
        auto lhs_scales_const = tensorflow::ops::Const(root, lhs_scales_tensor);
        auto lhs_zero_points_const = tensorflow::ops::Const(root, lhs_zero_points_tensor);
        auto rhs_scales_const = tensorflow::ops::Const(root, rhs_scales_tensor);
        auto rhs_zero_points_const = tensorflow::ops::Const(root, rhs_zero_points_tensor);
        auto output_scales_const = tensorflow::ops::Const(root, output_scales_tensor);
        auto output_zero_points_const = tensorflow::ops::Const(root, output_zero_points_tensor);
        
        tensorflow::Node* uniform_quantized_dot_node;
        tensorflow::NodeBuilder builder("uniform_quantized_dot", "UniformQuantizedDot");
        builder.Input(lhs_const.node())
               .Input(rhs_const.node())
               .Input(lhs_scales_const.node())
               .Input(lhs_zero_points_const.node())
               .Input(rhs_scales_const.node())
               .Input(rhs_zero_points_const.node())
               .Input(output_scales_const.node())
               .Input(output_zero_points_const.node())
               .Attr("Tout", tensorflow::DT_QINT32)
               .Attr("lhs_quantization_min_val", -128)
               .Attr("lhs_quantization_max_val", 127)
               .Attr("rhs_quantization_min_val", -128)
               .Attr("rhs_quantization_max_val", 127)
               .Attr("output_quantization_min_val", static_cast<int64_t>(-2147483648LL))
               .Attr("output_quantization_max_val", 2147483647)
               .Attr("lhs_quantization_axis", -1)
               .Attr("rhs_quantization_axis", -1)
               .Attr("output_quantization_axis", -1);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &uniform_quantized_dot_node);
        if (!build_status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Create a vector of Output objects
        std::vector<tensorflow::Output> fetch_outputs;
        fetch_outputs.push_back(tensorflow::Output(uniform_quantized_dot_node, 0));
        
        tensorflow::Status status = session.Run(fetch_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
