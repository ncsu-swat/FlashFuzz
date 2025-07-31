#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset,
                                                total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset,
                                                 total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(indices_shape));
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT32, data, offset, size);
        auto indices = tensorflow::ops::Const(root, indices_tensor);

        uint8_t gradient_rank = parseRank(data[offset++]);
        std::vector<int64_t> gradient_shape = parseShape(data, offset, size, gradient_rank);
        tensorflow::Tensor gradient_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gradient_shape));
        fillTensorWithDataByType(gradient_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto gradient = tensorflow::ops::Const(root, gradient_tensor);

        uint8_t learning_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> learning_rate_shape = parseShape(data, offset, size, learning_rate_rank);
        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(learning_rate_shape));
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

        uint8_t accumulator_rank = parseRank(data[offset++]);
        std::vector<int64_t> accumulator_shape = parseShape(data, offset, size, accumulator_rank);
        tensorflow::Tensor accumulator_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(accumulator_shape));
        fillTensorWithDataByType(accumulator_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto accumulator = tensorflow::ops::Const(root, accumulator_tensor);

        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(embedding_table_shape));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        int feature_width = 1;
        if (offset < size) {
            feature_width = static_cast<int>(data[offset++] % 100 + 1);
        }

        tensorflow::NodeDef node_def;
        node_def.set_name("XlaSparseCoreAdagrad");
        node_def.set_op("XlaSparseCoreAdagrad");
        
        auto indices_node = indices.node();
        auto gradient_node = gradient.node();
        auto learning_rate_node = learning_rate.node();
        auto accumulator_node = accumulator.node();
        auto embedding_table_node = embedding_table.node();
        
        tensorflow::NodeDefBuilder builder("XlaSparseCoreAdagrad", "XlaSparseCoreAdagrad");
        builder.Input(indices_node->name(), 0, tensorflow::DT_INT32)
               .Input(gradient_node->name(), 0, tensorflow::DT_FLOAT)
               .Input(learning_rate_node->name(), 0, tensorflow::DT_FLOAT)
               .Input(accumulator_node->name(), 0, tensorflow::DT_FLOAT)
               .Input(embedding_table_node->name(), 0, tensorflow::DT_FLOAT)
               .Attr("feature_width", feature_width);
        
        tensorflow::Status s = builder.Finalize(&node_def);
        if (!s.ok()) {
            tf_fuzzer_utils::logError("Failed to build NodeDef: " + s.ToString(), data, size);
            return -1;
        }
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add node: " + status.ToString(), data, size);
            return -1;
        }
        
        auto updated_embedding_table = tensorflow::Output(op, 0);
        auto updated_accumulator = tensorflow::Output(op, 1);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({updated_embedding_table, updated_accumulator}, &outputs);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Session run failed: " + status.ToString(), data, size);
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}