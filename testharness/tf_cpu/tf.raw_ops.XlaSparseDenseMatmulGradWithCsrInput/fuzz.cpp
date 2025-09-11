#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/logging.h"
#include <iostream>
#include <vector>
#include <cstring>
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
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

        uint8_t num_tables = (data[offset++] % 3) + 1;
        std::vector<tensorflow::Output> tables;
        for (int i = 0; i < num_tables; ++i) {
            uint8_t rank_table = parseRank(data[offset++]);
            std::vector<int64_t> shape_table = parseShape(data, offset, size, rank_table);
            tensorflow::Tensor table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape_table));
            fillTensorWithDataByType(table_tensor, tensorflow::DT_FLOAT, data, offset, size);
            tables.push_back(tensorflow::ops::Const(root, table_tensor));
        }

        uint8_t num_hyperparams = (data[offset++] % 3) + 1;
        std::vector<tensorflow::Output> hyperparameters;
        for (int i = 0; i < num_hyperparams; ++i) {
            uint8_t rank_hyper = parseRank(data[offset++]);
            std::vector<int64_t> shape_hyper = parseShape(data, offset, size, rank_hyper);
            tensorflow::Tensor hyper_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape_hyper));
            fillTensorWithDataByType(hyper_tensor, tensorflow::DT_FLOAT, data, offset, size);
            hyperparameters.push_back(tensorflow::ops::Const(root, hyper_tensor));
        }

        uint8_t rank_minibatch = parseRank(data[offset++]);
        std::vector<int64_t> shape_minibatch = parseShape(data, offset, size, rank_minibatch);
        tensorflow::Tensor minibatch_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape_minibatch));
        fillTensorWithDataByType(minibatch_tensor, tensorflow::DT_INT32, data, offset, size);
        auto num_minibatches_per_physical_sparse_core = tensorflow::ops::Const(root, minibatch_tensor);

        std::string table_name = "test_table";

        tensorflow::NodeBuilder builder("XlaSparseDenseMatmulGradWithCsrInput", "XlaSparseDenseMatmulGradWithCsrInput");
        builder.Input(row_pointers.node())
               .Input(sorted_sample_ids.node())
               .Input(sorted_token_ids.node())
               .Input(sorted_gains.node())
               .Input(activation_gradients.node());
        
        for (const auto& table : tables) {
            builder.Input(table.node());
        }
        
        for (const auto& hyperparam : hyperparameters) {
            builder.Input(hyperparam.node());
        }
        
        builder.Input(num_minibatches_per_physical_sparse_core.node())
               .Attr("table_name", table_name);

        tensorflow::Node* op_node;
        tensorflow::Status build_status = builder.Finalize(root.graph(), &op_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
