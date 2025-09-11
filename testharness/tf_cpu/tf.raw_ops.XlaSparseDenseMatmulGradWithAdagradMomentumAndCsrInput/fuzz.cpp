#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/tensor_shape.h"
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

        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + sizeof(float) <= size) {
            float lr_val;
            std::memcpy(&lr_val, data + offset, sizeof(float));
            offset += sizeof(float);
            learning_rate_tensor.scalar<float>()() = lr_val;
        } else {
            learning_rate_tensor.scalar<float>()() = 0.01f;
        }
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

        uint8_t momenta_rank = parseRank(data[offset++]);
        std::vector<int64_t> momenta_shape = parseShape(data, offset, size, momenta_rank);
        tensorflow::Tensor momenta_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(momenta_shape));
        fillTensorWithDataByType(momenta_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto momenta = tensorflow::ops::Const(root, momenta_tensor);

        tensorflow::Tensor num_minibatches_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        if (offset + sizeof(int32_t) <= size) {
            int32_t num_mb;
            std::memcpy(&num_mb, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_minibatches_tensor.scalar<int32_t>()() = std::abs(num_mb) % 100 + 1;
        } else {
            num_minibatches_tensor.scalar<int32_t>()() = 1;
        }
        auto num_minibatches_per_physical_sparse_core = tensorflow::ops::Const(root, num_minibatches_tensor);

        bool use_nesterov = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        float exponent = 0.5f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&exponent, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(exponent) || std::isinf(exponent)) {
                exponent = 0.5f;
            }
        }

        float beta1 = 0.9f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta1, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(beta1) || std::isinf(beta1)) {
                beta1 = 0.9f;
            }
        }

        float beta2 = 0.999f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta2, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(beta2) || std::isinf(beta2)) {
                beta2 = 0.999f;
            }
        }

        float epsilon = 1e-8f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&epsilon, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(epsilon) || std::isinf(epsilon) || epsilon <= 0) {
                epsilon = 1e-8f;
            }
        }

        std::string table_name = "test_table";

        float clip_weight_min = -std::numeric_limits<float>::infinity();
        if (offset + sizeof(float) <= size) {
            std::memcpy(&clip_weight_min, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        float clip_weight_max = std::numeric_limits<float>::infinity();
        if (offset + sizeof(float) <= size) {
            std::memcpy(&clip_weight_max, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        // Use raw_ops directly since the op is not available in the C++ API
        tensorflow::NodeDef node_def;
        node_def.set_name("XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput");
        node_def.set_op("XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput");
        
        // Add inputs
        tensorflow::AddNodeInput("row_pointers", &node_def);
        tensorflow::AddNodeInput("sorted_sample_ids", &node_def);
        tensorflow::AddNodeInput("sorted_token_ids", &node_def);
        tensorflow::AddNodeInput("sorted_gains", &node_def);
        tensorflow::AddNodeInput("activation_gradients", &node_def);
        tensorflow::AddNodeInput("learning_rate", &node_def);
        tensorflow::AddNodeInput("embedding_table", &node_def);
        tensorflow::AddNodeInput("accumulator", &node_def);
        tensorflow::AddNodeInput("momenta", &node_def);
        tensorflow::AddNodeInput("num_minibatches_per_physical_sparse_core", &node_def);
        
        // Add attributes
        auto* attr_map = node_def.mutable_attr();
        (*attr_map)["use_nesterov"].set_b(use_nesterov);
        (*attr_map)["exponent"].set_f(exponent);
        (*attr_map)["beta1"].set_f(beta1);
        (*attr_map)["beta2"].set_f(beta2);
        (*attr_map)["epsilon"].set_f(epsilon);
        (*attr_map)["table_name"].set_s(table_name);
        (*attr_map)["clip_weight_min"].set_f(clip_weight_min);
        (*attr_map)["clip_weight_max"].set_f(clip_weight_max);
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddOperation(tensorflow::Operation(root.graph(), node_def, &status));
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create operation: " + status.ToString(), data, size);
            return -1;
        }
        
        // Create outputs
        auto updated_embedding_table = tensorflow::Output(op, 0);
        auto updated_accumulator = tensorflow::Output(op, 1);
        auto updated_momenta = tensorflow::Output(op, 2);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({updated_embedding_table, updated_accumulator, updated_momenta}, &outputs);
        
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
