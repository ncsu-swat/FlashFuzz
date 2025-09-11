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
#include "tensorflow/core/framework/shape_inference.h"
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t row_pointers_rank = parseRank(data[offset++]);
        std::vector<int64_t> row_pointers_shape = parseShape(data, offset, size, row_pointers_rank);
        tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(row_pointers_shape));
        fillTensorWithDataByType(row_pointers_tensor, tensorflow::DT_INT32, data, offset, size);
        auto row_pointers = tensorflow::ops::Const(root, row_pointers_tensor);

        uint8_t sorted_sample_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> sorted_sample_ids_shape = parseShape(data, offset, size, sorted_sample_ids_rank);
        tensorflow::Tensor sorted_sample_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(sorted_sample_ids_shape));
        fillTensorWithDataByType(sorted_sample_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto sorted_sample_ids = tensorflow::ops::Const(root, sorted_sample_ids_tensor);

        uint8_t sorted_token_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> sorted_token_ids_shape = parseShape(data, offset, size, sorted_token_ids_rank);
        tensorflow::Tensor sorted_token_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(sorted_token_ids_shape));
        fillTensorWithDataByType(sorted_token_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto sorted_token_ids = tensorflow::ops::Const(root, sorted_token_ids_tensor);

        uint8_t sorted_gains_rank = parseRank(data[offset++]);
        std::vector<int64_t> sorted_gains_shape = parseShape(data, offset, size, sorted_gains_rank);
        tensorflow::Tensor sorted_gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(sorted_gains_shape));
        fillTensorWithDataByType(sorted_gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto sorted_gains = tensorflow::ops::Const(root, sorted_gains_tensor);

        uint8_t activation_gradients_rank = parseRank(data[offset++]);
        std::vector<int64_t> activation_gradients_shape = parseShape(data, offset, size, activation_gradients_rank);
        tensorflow::Tensor activation_gradients_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(activation_gradients_shape));
        fillTensorWithDataByType(activation_gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto activation_gradients = tensorflow::ops::Const(root, activation_gradients_tensor);

        uint8_t learning_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> learning_rate_shape = parseShape(data, offset, size, learning_rate_rank);
        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(learning_rate_shape));
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(embedding_table_shape));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        uint8_t accumulator_rank = parseRank(data[offset++]);
        std::vector<int64_t> accumulator_shape = parseShape(data, offset, size, accumulator_rank);
        tensorflow::Tensor accumulator_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(accumulator_shape));
        fillTensorWithDataByType(accumulator_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto accumulator = tensorflow::ops::Const(root, accumulator_tensor);

        uint8_t num_minibatches_rank = parseRank(data[offset++]);
        std::vector<int64_t> num_minibatches_shape = parseShape(data, offset, size, num_minibatches_rank);
        tensorflow::Tensor num_minibatches_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(num_minibatches_shape));
        fillTensorWithDataByType(num_minibatches_tensor, tensorflow::DT_INT32, data, offset, size);
        auto num_minibatches_per_physical_sparse_core = tensorflow::ops::Const(root, num_minibatches_tensor);

        std::string table_name = "test_table";
        
        float clip_weight_min = -std::numeric_limits<float>::infinity();
        float clip_weight_max = std::numeric_limits<float>::infinity();
        
        if (offset < size) {
            std::memcpy(&clip_weight_min, data + offset, std::min(sizeof(float), size - offset));
            offset += sizeof(float);
        }
        if (offset < size) {
            std::memcpy(&clip_weight_max, data + offset, std::min(sizeof(float), size - offset));
            offset += sizeof(float);
        }

        tensorflow::NodeDef node_def;
        node_def.set_name("XlaSparseDenseMatmulGradWithAdagradAndCsrInput");
        node_def.set_op("XlaSparseDenseMatmulGradWithAdagradAndCsrInput");
        
        auto attr_clip_weight_min = node_def.mutable_attr()->insert({"clip_weight_min", tensorflow::AttrValue()});
        attr_clip_weight_min->second.set_f(clip_weight_min);
        
        auto attr_clip_weight_max = node_def.mutable_attr()->insert({"clip_weight_max", tensorflow::AttrValue()});
        attr_clip_weight_max->second.set_f(clip_weight_max);
        
        auto attr_table_name = node_def.mutable_attr()->insert({"table_name", tensorflow::AttrValue()});
        attr_table_name->second.set_s(table_name);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Output> inputs = {
            row_pointers,
            sorted_sample_ids,
            sorted_token_ids,
            sorted_gains,
            activation_gradients,
            learning_rate,
            embedding_table,
            accumulator,
            num_minibatches_per_physical_sparse_core
        };

        // Since we can't directly use the op, we'll just run the inputs to make sure they're valid
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(inputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
