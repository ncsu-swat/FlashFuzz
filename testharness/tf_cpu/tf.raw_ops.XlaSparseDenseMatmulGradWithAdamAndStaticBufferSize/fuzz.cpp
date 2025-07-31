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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 100) return 0;
    
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
            learning_rate_tensor.scalar<float>()() = 0.001f;
        }
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(embedding_table_shape));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        uint8_t momenta_rank = parseRank(data[offset++]);
        std::vector<int64_t> momenta_shape = parseShape(data, offset, size, momenta_rank);
        tensorflow::Tensor momenta_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(momenta_shape));
        fillTensorWithDataByType(momenta_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto momenta = tensorflow::ops::Const(root, momenta_tensor);

        uint8_t velocity_rank = parseRank(data[offset++]);
        std::vector<int64_t> velocity_shape = parseShape(data, offset, size, velocity_rank);
        tensorflow::Tensor velocity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(velocity_shape));
        fillTensorWithDataByType(velocity_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto velocity = tensorflow::ops::Const(root, velocity_tensor);

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

        bool use_sum_inside_sqrt = (offset < size) ? (data[offset++] % 2 == 0) : false;

        float beta1 = 0.9f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta1, data + offset, sizeof(float));
            offset += sizeof(float);
            beta1 = std::abs(beta1);
            if (beta1 > 1.0f) beta1 = 0.9f;
        }

        float beta2 = 0.999f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta2, data + offset, sizeof(float));
            offset += sizeof(float);
            beta2 = std::abs(beta2);
            if (beta2 > 1.0f) beta2 = 0.999f;
        }

        float epsilon = 1e-8f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&epsilon, data + offset, sizeof(float));
            offset += sizeof(float);
            epsilon = std::abs(epsilon);
            if (epsilon == 0.0f) epsilon = 1e-8f;
        }

        int max_ids_per_sparse_core = 1;
        if (offset + sizeof(int32_t) <= size) {
            int32_t max_ids;
            std::memcpy(&max_ids, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_ids_per_sparse_core = std::abs(max_ids) % 1000 + 1;
        }

        int max_unique_ids_per_sparse_core = 1;
        if (offset + sizeof(int32_t) <= size) {
            int32_t max_unique_ids;
            std::memcpy(&max_unique_ids, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_unique_ids_per_sparse_core = std::abs(max_unique_ids) % 1000 + 1;
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

        tensorflow::Node* op_node;
        tensorflow::NodeBuilder builder("XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize", "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize");
        builder.Input(row_pointers.node())
               .Input(sorted_sample_ids.node())
               .Input(sorted_token_ids.node())
               .Input(sorted_gains.node())
               .Input(activation_gradients.node())
               .Input(learning_rate.node())
               .Input(embedding_table.node())
               .Input(momenta.node())
               .Input(velocity.node())
               .Input(num_minibatches_per_physical_sparse_core.node())
               .Attr("use_sum_inside_sqrt", use_sum_inside_sqrt)
               .Attr("beta1", beta1)
               .Attr("beta2", beta2)
               .Attr("epsilon", epsilon)
               .Attr("max_ids_per_sparse_core", max_ids_per_sparse_core)
               .Attr("max_unique_ids_per_sparse_core", max_unique_ids_per_sparse_core)
               .Attr("table_name", table_name)
               .Attr("clip_weight_min", clip_weight_min)
               .Attr("clip_weight_max", clip_weight_max);

        tensorflow::Status build_status = builder.Finalize(root.graph(), &op_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({tensorflow::Output(op_node, 0), 
                                                 tensorflow::Output(op_node, 1), 
                                                 tensorflow::Output(op_node, 2)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}