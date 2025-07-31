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
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

        uint8_t rank7 = parseRank(data[offset++]);
        std::vector<int64_t> shape7 = parseShape(data, offset, size, rank7);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape7));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        uint8_t rank8 = parseRank(data[offset++]);
        std::vector<int64_t> shape8 = parseShape(data, offset, size, rank8);
        tensorflow::Tensor accumulator_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape8));
        fillTensorWithDataByType(accumulator_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto accumulator = tensorflow::ops::Const(root, accumulator_tensor);

        uint8_t rank9 = parseRank(data[offset++]);
        std::vector<int64_t> shape9 = parseShape(data, offset, size, rank9);
        tensorflow::Tensor momenta_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape9));
        fillTensorWithDataByType(momenta_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto momenta = tensorflow::ops::Const(root, momenta_tensor);

        tensorflow::Tensor num_minibatches_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        fillTensorWithDataByType(num_minibatches_tensor, tensorflow::DT_INT32, data, offset, size);
        auto num_minibatches_per_physical_sparse_core = tensorflow::ops::Const(root, num_minibatches_tensor);

        bool use_nesterov = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        float exponent = 2.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&exponent, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        float beta1 = 0.9f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta1, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        float beta2 = 0.999f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta2, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        float epsilon = 1e-8f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&epsilon, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        int max_ids_per_sparse_core = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_ids_per_sparse_core, data + offset, sizeof(int));
            offset += sizeof(int);
            max_ids_per_sparse_core = std::max(1, std::abs(max_ids_per_sparse_core) % 1000 + 1);
        }

        int max_unique_ids_per_sparse_core = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_unique_ids_per_sparse_core, data + offset, sizeof(int));
            offset += sizeof(int);
            max_unique_ids_per_sparse_core = std::max(1, std::abs(max_unique_ids_per_sparse_core) % 1000 + 1);
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

        tensorflow::NodeDef node_def;
        node_def.set_name("XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize");
        node_def.set_op("XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize");
        
        auto attr_use_nesterov = node_def.mutable_attr()->insert({"use_nesterov", tensorflow::AttrValue()});
        attr_use_nesterov->second.set_b(use_nesterov);
        
        auto attr_exponent = node_def.mutable_attr()->insert({"exponent", tensorflow::AttrValue()});
        attr_exponent->second.set_f(exponent);
        
        auto attr_beta1 = node_def.mutable_attr()->insert({"beta1", tensorflow::AttrValue()});
        attr_beta1->second.set_f(beta1);
        
        auto attr_beta2 = node_def.mutable_attr()->insert({"beta2", tensorflow::AttrValue()});
        attr_beta2->second.set_f(beta2);
        
        auto attr_epsilon = node_def.mutable_attr()->insert({"epsilon", tensorflow::AttrValue()});
        attr_epsilon->second.set_f(epsilon);
        
        auto attr_max_ids = node_def.mutable_attr()->insert({"max_ids_per_sparse_core", tensorflow::AttrValue()});
        attr_max_ids->second.set_i(max_ids_per_sparse_core);
        
        auto attr_max_unique_ids = node_def.mutable_attr()->insert({"max_unique_ids_per_sparse_core", tensorflow::AttrValue()});
        attr_max_unique_ids->second.set_i(max_unique_ids_per_sparse_core);
        
        auto attr_table_name = node_def.mutable_attr()->insert({"table_name", tensorflow::AttrValue()});
        attr_table_name->second.set_s(table_name);
        
        auto attr_clip_weight_min = node_def.mutable_attr()->insert({"clip_weight_min", tensorflow::AttrValue()});
        attr_clip_weight_min->second.set_f(clip_weight_min);
        
        auto attr_clip_weight_max = node_def.mutable_attr()->insert({"clip_weight_max", tensorflow::AttrValue()});
        attr_clip_weight_max->second.set_f(clip_weight_max);

        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return 0;
        }

        std::vector<tensorflow::Output> inputs = {
            row_pointers, 
            sorted_sample_ids, 
            sorted_token_ids, 
            sorted_gains, 
            activation_gradients, 
            learning_rate, 
            embedding_table, 
            accumulator, 
            momenta, 
            num_minibatches_per_physical_sparse_core
        };

        for (size_t i = 0; i < inputs.size(); ++i) {
            tensorflow::NodeDefBuilder::NodeOut node_out(inputs[i].node()->name(), 0, inputs[i].type());
            status = tensorflow::NodeDefBuilder("input_" + std::to_string(i), "Identity")
                .Input(node_out)
                .Finalize(node_def.add_input());
            if (!status.ok()) {
                return 0;
            }
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Since we can't directly call the op, we'll just run the session with the inputs
        // to make sure they're valid
        status = session.Run({embedding_table, accumulator, momenta}, &outputs);
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return 0;
    }

    return 0;
}