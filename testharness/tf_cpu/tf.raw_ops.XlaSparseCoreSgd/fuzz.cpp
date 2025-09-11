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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT32, data, offset, size);
        
        uint8_t gradient_rank = parseRank(data[offset++]);
        std::vector<int64_t> gradient_shape = parseShape(data, offset, size, gradient_rank);
        tensorflow::TensorShape gradient_tensor_shape(gradient_shape);
        tensorflow::Tensor gradient_tensor(tensorflow::DT_FLOAT, gradient_tensor_shape);
        fillTensorWithDataByType(gradient_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::TensorShape learning_rate_tensor_shape({});
        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, learning_rate_tensor_shape);
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::TensorShape embedding_table_tensor_shape(embedding_table_shape);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, embedding_table_tensor_shape);
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int feature_width = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&feature_width, data + offset, sizeof(int));
            offset += sizeof(int);
            feature_width = std::abs(feature_width) % 100 + 1;
        }

        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto gradient_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto learning_rate_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto embedding_table_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

        tensorflow::Node* xla_sparse_core_sgd_node;
        tensorflow::NodeBuilder builder("XlaSparseCoreSgd", "XlaSparseCoreSgd");
        builder.Input(indices_placeholder.node())
               .Input(gradient_placeholder.node())
               .Input(learning_rate_placeholder.node())
               .Input(embedding_table_placeholder.node())
               .Attr("feature_width", feature_width);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &xla_sparse_core_sgd_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{indices_placeholder, indices_tensor},
             {gradient_placeholder, gradient_tensor},
             {learning_rate_placeholder, learning_rate_tensor},
             {embedding_table_placeholder, embedding_table_tensor}},
            {tensorflow::Output(xla_sparse_core_sgd_node)},
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
