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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank1 = parseRank(data[offset++]);
        std::vector<int64_t> shape1 = parseShape(data, offset, size, rank1);
        tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape1));
        fillTensorWithDataByType(row_pointers_tensor, tensorflow::DT_INT32, data, offset, size);
        auto row_pointers = tensorflow::ops::Const(root, row_pointers_tensor);

        uint8_t rank2 = parseRank(data[offset++]);
        std::vector<int64_t> shape2 = parseShape(data, offset, size, rank2);
        tensorflow::Tensor sorted_sample_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape2));
        fillTensorWithDataByType(sorted_sample_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto sorted_sample_ids = tensorflow::ops::Const(root, sorted_sample_ids_tensor);

        uint8_t rank3 = parseRank(data[offset++]);
        std::vector<int64_t> shape3 = parseShape(data, offset, size, rank3);
        tensorflow::Tensor sorted_token_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape3));
        fillTensorWithDataByType(sorted_token_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto sorted_token_ids = tensorflow::ops::Const(root, sorted_token_ids_tensor);

        uint8_t rank4 = parseRank(data[offset++]);
        std::vector<int64_t> shape4 = parseShape(data, offset, size, rank4);
        tensorflow::Tensor sorted_gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape4));
        fillTensorWithDataByType(sorted_gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto sorted_gains = tensorflow::ops::Const(root, sorted_gains_tensor);

        uint8_t rank5 = parseRank(data[offset++]);
        std::vector<int64_t> shape5 = parseShape(data, offset, size, rank5);
        tensorflow::Tensor activation_gradients_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape5));
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

        uint8_t rank6 = parseRank(data[offset++]);
        std::vector<int64_t> shape6 = parseShape(data, offset, size, rank6);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape6));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        uint8_t rank7 = parseRank(data[offset++]);
        std::vector<int64_t> shape7 = parseShape(data, offset, size, rank7);
        tensorflow::Tensor momenta_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape7));
        fillTensorWithDataByType(momenta_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto momenta = tensorflow::ops::Const(root, momenta_tensor);

        uint8_t rank8 = parseRank(data[offset++]);
        std::vector<int64_t> shape8 = parseShape(data, offset, size, rank8);
        tensorflow::Tensor velocity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape8));
        fillTensorWithDataByType(velocity_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto velocity = tensorflow::ops::Const(root, velocity_tensor);

        tensorflow::Tensor num_minibatches_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        if (offset + sizeof(int32_t) <= size) {
            int32_t num_val;
            std::memcpy(&num_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_minibatches_tensor.scalar<int32_t>()() = std::abs(num_val) % 100 + 1;
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

        // Use raw_ops instead of ops namespace
        auto op_attrs = tensorflow::AttrValue();
        op_attrs.set_b(use_sum_inside_sqrt);
        std::vector<std::pair<std::string, tensorflow::AttrValue>> attrs;
        attrs.push_back({"use_sum_inside_sqrt", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_f(beta1);
        attrs.push_back({"beta1", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_f(beta2);
        attrs.push_back({"beta2", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_f(epsilon);
        attrs.push_back({"epsilon", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_s(table_name);
        attrs.push_back({"table_name", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_f(clip_weight_min);
        attrs.push_back({"clip_weight_min", op_attrs});
        
        op_attrs = tensorflow::AttrValue();
        op_attrs.set_f(clip_weight_max);
        attrs.push_back({"clip_weight_max", op_attrs});

        // Create the operation using NodeBuilder
        tensorflow::NodeBuilder node_builder("XlaSparseDenseMatmulGradWithAdamAndCsrInput", "XlaSparseDenseMatmulGradWithAdamAndCsrInput");
        node_builder.Input(row_pointers.node());
        node_builder.Input(sorted_sample_ids.node());
        node_builder.Input(sorted_token_ids.node());
        node_builder.Input(sorted_gains.node());
        node_builder.Input(activation_gradients.node());
        node_builder.Input(learning_rate.node());
        node_builder.Input(embedding_table.node());
        node_builder.Input(momenta.node());
        node_builder.Input(velocity.node());
        node_builder.Input(num_minibatches_per_physical_sparse_core.node());
        
        for (const auto& attr : attrs) {
            node_builder.Attr(attr.first, attr.second);
        }

        tensorflow::Node* node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create node: " + status.ToString(), data, size);
            return -1;
        }

        // Create output operations
        tensorflow::ops::Identity updated_embedding_table(root, {tensorflow::Output(node, 0)});
        tensorflow::ops::Identity updated_momenta(root, {tensorflow::Output(node, 1)});
        tensorflow::ops::Identity updated_velocity(root, {tensorflow::Output(node, 2)});

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({updated_embedding_table, updated_momenta, updated_velocity}, &outputs);
        
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
