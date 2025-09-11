#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t indices_rank = parseRank(data[offset++]);
        if (indices_rank < 2) indices_rank = 2;
        
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        uint8_t input_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape_shape = parseShape(data, offset, size, input_shape_rank);
        
        uint8_t new_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> new_shape_shape = parseShape(data, offset, size, new_shape_rank);

        tensorflow::TensorShape indices_tensor_shape;
        for (auto dim : indices_shape) {
            indices_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor input_indices(tensorflow::DT_INT64, indices_tensor_shape);
        fillTensorWithData<int64_t>(input_indices, data, offset, size);

        tensorflow::TensorShape input_shape_tensor_shape;
        for (auto dim : input_shape_shape) {
            input_shape_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor input_shape(tensorflow::DT_INT64, input_shape_tensor_shape);
        fillTensorWithData<int64_t>(input_shape, data, offset, size);

        tensorflow::TensorShape new_shape_tensor_shape;
        for (auto dim : new_shape_shape) {
            new_shape_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor new_shape(tensorflow::DT_INT64, new_shape_tensor_shape);
        fillTensorWithData<int64_t>(new_shape, data, offset, size);

        auto input_indices_op = tensorflow::ops::Const(root, input_indices);
        auto input_shape_op = tensorflow::ops::Const(root, input_shape);
        auto new_shape_op = tensorflow::ops::Const(root, new_shape);

        auto sparse_reshape = tensorflow::ops::SparseReshape(root, input_indices_op, input_shape_op, new_shape_op);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_reshape.output_indices, sparse_reshape.output_shape}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
