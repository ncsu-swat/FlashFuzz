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
    std::cerr << message << std::endl;
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t operand_rank = parseRank(data[offset++]);
        std::vector<int64_t> operand_shape = parseShape(data, offset, size, operand_rank);
        
        tensorflow::TensorShape operand_tensor_shape;
        for (int64_t dim : operand_shape) {
            operand_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor operand_tensor(tensorflow::DT_QINT32, operand_tensor_shape);
        fillTensorWithDataByType(operand_tensor, tensorflow::DT_QINT32, data, offset, size);
        
        if (offset >= size) return 0;
        
        int8_t quantization_axis = -1;
        if (offset < size) {
            quantization_axis = static_cast<int8_t>(data[offset++]);
            if (quantization_axis >= 0 && operand_rank > 0) {
                quantization_axis = quantization_axis % operand_rank;
            } else {
                quantization_axis = -1;
            }
        }
        
        tensorflow::TensorShape min_max_shape;
        tensorflow::TensorShape scales_zp_shape;
        
        if (quantization_axis == -1) {
            min_max_shape = tensorflow::TensorShape({});
            scales_zp_shape = tensorflow::TensorShape({});
        } else {
            int64_t axis_size = operand_shape[quantization_axis];
            min_max_shape = tensorflow::TensorShape({axis_size});
            scales_zp_shape = tensorflow::TensorShape({axis_size});
        }
        
        tensorflow::Tensor min_tensor(tensorflow::DT_QINT32, min_max_shape);
        fillTensorWithDataByType(min_tensor, tensorflow::DT_QINT32, data, offset, size);
        
        tensorflow::Tensor max_tensor(tensorflow::DT_QINT32, min_max_shape);
        fillTensorWithDataByType(max_tensor, tensorflow::DT_QINT32, data, offset, size);
        
        tensorflow::Tensor scales_tensor(tensorflow::DT_FLOAT, scales_zp_shape);
        fillTensorWithDataByType(scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor zero_points_tensor(tensorflow::DT_INT32, scales_zp_shape);
        fillTensorWithDataByType(zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        int32_t quantization_min_val = -2147483648;
        int32_t quantization_max_val = 2147483647;
        
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&quantization_min_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&quantization_max_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        
        auto operand_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_QINT32);
        auto min_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_QINT32);
        auto max_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_QINT32);
        auto scales_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto zero_points_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        tensorflow::NodeDef node_def;
        node_def.set_name("UniformQuantizedClipByValue");
        node_def.set_op("UniformQuantizedClipByValue");
        
        // Add inputs to the node def
        node_def.add_input(operand_placeholder.node()->name());
        node_def.add_input(min_placeholder.node()->name());
        node_def.add_input(max_placeholder.node()->name());
        node_def.add_input(scales_placeholder.node()->name());
        node_def.add_input(zero_points_placeholder.node()->name());
        
        // Set attributes
        auto attr_map = node_def.mutable_attr();
        (*attr_map)["T"].set_type(tensorflow::DT_QINT32);
        (*attr_map)["quantization_axis"].set_i(quantization_axis);
        (*attr_map)["quantization_min_val"].set_i(quantization_min_val);
        (*attr_map)["quantization_max_val"].set_i(quantization_max_val);
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run(
            {{operand_placeholder, operand_tensor},
             {min_placeholder, min_tensor},
             {max_placeholder, max_tensor},
             {scales_placeholder, scales_tensor},
             {zero_points_placeholder, zero_points_tensor}},
            {op},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
