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
#include <limits>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
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
    case tensorflow::DT_QINT32:
      fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t lhs_rank = parseRank(data[offset++]);
        uint8_t rhs_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> lhs_shape = parseShape(data, offset, size, lhs_rank);
        std::vector<int64_t> rhs_shape = parseShape(data, offset, size, rhs_rank);
        
        tensorflow::TensorShape lhs_tensor_shape(lhs_shape);
        tensorflow::TensorShape rhs_tensor_shape(rhs_shape);
        
        tensorflow::Tensor lhs_tensor(tensorflow::DT_QINT32, lhs_tensor_shape);
        tensorflow::Tensor rhs_tensor(tensorflow::DT_QINT32, rhs_tensor_shape);
        
        fillTensorWithDataByType(lhs_tensor, tensorflow::DT_QINT32, data, offset, size);
        fillTensorWithDataByType(rhs_tensor, tensorflow::DT_QINT32, data, offset, size);
        
        if (offset >= size) return 0;
        
        uint8_t lhs_scales_rank = parseRank(data[offset++]);
        uint8_t rhs_scales_rank = parseRank(data[offset++]);
        uint8_t output_scales_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> lhs_scales_shape = parseShape(data, offset, size, lhs_scales_rank);
        std::vector<int64_t> rhs_scales_shape = parseShape(data, offset, size, rhs_scales_rank);
        std::vector<int64_t> output_scales_shape = parseShape(data, offset, size, output_scales_rank);
        
        tensorflow::TensorShape lhs_scales_tensor_shape(lhs_scales_shape);
        tensorflow::TensorShape rhs_scales_tensor_shape(rhs_scales_shape);
        tensorflow::TensorShape output_scales_tensor_shape(output_scales_shape);
        
        tensorflow::Tensor lhs_scales_tensor(tensorflow::DT_FLOAT, lhs_scales_tensor_shape);
        tensorflow::Tensor rhs_scales_tensor(tensorflow::DT_FLOAT, rhs_scales_tensor_shape);
        tensorflow::Tensor output_scales_tensor(tensorflow::DT_FLOAT, output_scales_tensor_shape);
        
        fillTensorWithDataByType(lhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(rhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(output_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor lhs_zero_points_tensor(tensorflow::DT_INT32, lhs_scales_tensor_shape);
        tensorflow::Tensor rhs_zero_points_tensor(tensorflow::DT_INT32, rhs_scales_tensor_shape);
        tensorflow::Tensor output_zero_points_tensor(tensorflow::DT_INT32, output_scales_tensor_shape);
        
        fillTensorWithDataByType(lhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(rhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(output_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        if (offset + 12 > size) return 0;
        
        int lhs_quantization_min_val, lhs_quantization_max_val;
        int rhs_quantization_min_val, rhs_quantization_max_val;
        int output_quantization_min_val, output_quantization_max_val;
        
        std::memcpy(&lhs_quantization_min_val, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&lhs_quantization_max_val, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&rhs_quantization_min_val, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&rhs_quantization_max_val, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&output_quantization_min_val, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&output_quantization_max_val, data + offset, sizeof(int));
        offset += sizeof(int);
        
        lhs_quantization_min_val = std::max<int>(-2147483647, std::min<int>(2147483647, lhs_quantization_min_val));
        lhs_quantization_max_val = std::max<int>(-2147483647, std::min<int>(2147483647, lhs_quantization_max_val));
        rhs_quantization_min_val = std::max<int>(-2147483647, std::min<int>(2147483647, rhs_quantization_min_val));
        rhs_quantization_max_val = std::max<int>(-2147483647, std::min<int>(2147483647, rhs_quantization_max_val));
        output_quantization_min_val = std::max<int>(-2147483647, std::min<int>(2147483647, output_quantization_min_val));
        output_quantization_max_val = std::max<int>(-2147483647, std::min<int>(2147483647, output_quantization_max_val));
        
        int lhs_quantization_axis = -1;
        int rhs_quantization_axis = -1;
        int output_quantization_axis = -1;
        
        auto lhs_input = tensorflow::ops::Const(root, lhs_tensor);
        auto rhs_input = tensorflow::ops::Const(root, rhs_tensor);
        auto lhs_scales_input = tensorflow::ops::Const(root, lhs_scales_tensor);
        auto lhs_zero_points_input = tensorflow::ops::Const(root, lhs_zero_points_tensor);
        auto rhs_scales_input = tensorflow::ops::Const(root, rhs_scales_tensor);
        auto rhs_zero_points_input = tensorflow::ops::Const(root, rhs_zero_points_tensor);
        auto output_scales_input = tensorflow::ops::Const(root, output_scales_tensor);
        auto output_zero_points_input = tensorflow::ops::Const(root, output_zero_points_tensor);
        
        tensorflow::NodeDef node_def;
        node_def.set_op("UniformQuantizedAdd");
        node_def.set_name("UniformQuantizedAdd");
        node_def.add_input(lhs_input.node()->name());
        node_def.add_input(rhs_input.node()->name());
        node_def.add_input(lhs_scales_input.node()->name());
        node_def.add_input(lhs_zero_points_input.node()->name());
        node_def.add_input(rhs_scales_input.node()->name());
        node_def.add_input(rhs_zero_points_input.node()->name());
        node_def.add_input(output_scales_input.node()->name());
        node_def.add_input(output_zero_points_input.node()->name());
        
        auto attr = node_def.mutable_attr();
        (*attr)["lhs_quantization_min_val"].set_i(lhs_quantization_min_val);
        (*attr)["lhs_quantization_max_val"].set_i(lhs_quantization_max_val);
        (*attr)["rhs_quantization_min_val"].set_i(rhs_quantization_min_val);
        (*attr)["rhs_quantization_max_val"].set_i(rhs_quantization_max_val);
        (*attr)["output_quantization_min_val"].set_i(output_quantization_min_val);
        (*attr)["output_quantization_max_val"].set_i(output_quantization_max_val);
        (*attr)["lhs_quantization_axis"].set_i(lhs_quantization_axis);
        (*attr)["rhs_quantization_axis"].set_i(rhs_quantization_axis);
        (*attr)["output_quantization_axis"].set_i(output_quantization_axis);
        
        tensorflow::Status status;
        auto uniform_quantized_add = tensorflow::ops::FromNodeDef(root, node_def, {
            lhs_input.output.data_type(),
            rhs_input.output.data_type(),
            lhs_scales_input.output.data_type(),
            lhs_zero_points_input.output.data_type(),
            rhs_scales_input.output.data_type(),
            rhs_zero_points_input.output.data_type(),
            output_scales_input.output.data_type(),
            output_zero_points_input.output.data_type()
        }, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        status = session.Run({uniform_quantized_add}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}