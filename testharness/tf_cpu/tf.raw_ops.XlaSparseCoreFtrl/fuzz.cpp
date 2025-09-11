#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
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

        uint8_t linear_rank = parseRank(data[offset++]);
        std::vector<int64_t> linear_shape = parseShape(data, offset, size, linear_rank);
        tensorflow::Tensor linear_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(linear_shape));
        fillTensorWithDataByType(linear_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto linear = tensorflow::ops::Const(root, linear_tensor);

        uint8_t learning_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> learning_rate_shape = parseShape(data, offset, size, learning_rate_rank);
        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(learning_rate_shape));
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

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

        uint8_t beta_rank = parseRank(data[offset++]);
        std::vector<int64_t> beta_shape = parseShape(data, offset, size, beta_rank);
        tensorflow::Tensor beta_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(beta_shape));
        fillTensorWithDataByType(beta_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto beta = tensorflow::ops::Const(root, beta_tensor);

        uint8_t learning_rate_power_rank = parseRank(data[offset++]);
        std::vector<int64_t> learning_rate_power_shape = parseShape(data, offset, size, learning_rate_power_rank);
        tensorflow::Tensor learning_rate_power_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(learning_rate_power_shape));
        fillTensorWithDataByType(learning_rate_power_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate_power = tensorflow::ops::Const(root, learning_rate_power_tensor);

        uint8_t l2_regularization_strength_rank = parseRank(data[offset++]);
        std::vector<int64_t> l2_regularization_strength_shape = parseShape(data, offset, size, l2_regularization_strength_rank);
        tensorflow::Tensor l2_regularization_strength_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(l2_regularization_strength_shape));
        fillTensorWithDataByType(l2_regularization_strength_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto l2_regularization_strength = tensorflow::ops::Const(root, l2_regularization_strength_tensor);

        int feature_width = 1;
        if (offset < size) {
            feature_width = static_cast<int>(data[offset++]) % 100 + 1;
        }

        bool multiply_linear_by_learning_rate = false;
        if (offset < size) {
            multiply_linear_by_learning_rate = (data[offset++] % 2) == 1;
        }

        float l1_regularization_strength = 0.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l1_regularization_strength, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        tensorflow::Node* node;
        tensorflow::NodeBuilder builder("XlaSparseCoreFtrl", "XlaSparseCoreFtrl");
        builder.Input(embedding_table.node())
               .Input(accumulator.node())
               .Input(linear.node())
               .Input(learning_rate.node())
               .Input(indices.node())
               .Input(gradient.node())
               .Input(beta.node())
               .Input(learning_rate_power.node())
               .Input(l2_regularization_strength.node())
               .Attr("feature_width", feature_width)
               .Attr("multiply_linear_by_learning_rate", multiply_linear_by_learning_rate)
               .Attr("l1_regularization_strength", l1_regularization_strength);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({tensorflow::Output(node, 0), tensorflow::Output(node, 1), tensorflow::Output(node, 2)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
