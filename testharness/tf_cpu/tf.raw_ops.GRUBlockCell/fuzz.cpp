#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    return tensorflow::DT_FLOAT;
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
    default:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t x_rank = parseRank(data[offset++]);
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        
        uint8_t h_prev_rank = parseRank(data[offset++]);
        std::vector<int64_t> h_prev_shape = parseShape(data, offset, size, h_prev_rank);
        
        uint8_t w_ru_rank = parseRank(data[offset++]);
        std::vector<int64_t> w_ru_shape = parseShape(data, offset, size, w_ru_rank);
        
        uint8_t w_c_rank = parseRank(data[offset++]);
        std::vector<int64_t> w_c_shape = parseShape(data, offset, size, w_c_rank);
        
        uint8_t b_ru_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_ru_shape = parseShape(data, offset, size, b_ru_rank);
        
        uint8_t b_c_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_c_shape = parseShape(data, offset, size, b_c_rank);

        tensorflow::TensorShape x_tensor_shape(x_shape);
        tensorflow::TensorShape h_prev_tensor_shape(h_prev_shape);
        tensorflow::TensorShape w_ru_tensor_shape(w_ru_shape);
        tensorflow::TensorShape w_c_tensor_shape(w_c_shape);
        tensorflow::TensorShape b_ru_tensor_shape(b_ru_shape);
        tensorflow::TensorShape b_c_tensor_shape(b_c_shape);

        tensorflow::Tensor x_tensor(dtype, x_tensor_shape);
        tensorflow::Tensor h_prev_tensor(dtype, h_prev_tensor_shape);
        tensorflow::Tensor w_ru_tensor(dtype, w_ru_tensor_shape);
        tensorflow::Tensor w_c_tensor(dtype, w_c_tensor_shape);
        tensorflow::Tensor b_ru_tensor(dtype, b_ru_tensor_shape);
        tensorflow::Tensor b_c_tensor(dtype, b_c_tensor_shape);

        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(h_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_ru_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_ru_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_c_tensor, dtype, data, offset, size);

        auto x = tensorflow::ops::Placeholder(root, dtype);
        auto h_prev = tensorflow::ops::Placeholder(root, dtype);
        auto w_ru = tensorflow::ops::Placeholder(root, dtype);
        auto w_c = tensorflow::ops::Placeholder(root, dtype);
        auto b_ru = tensorflow::ops::Placeholder(root, dtype);
        auto b_c = tensorflow::ops::Placeholder(root, dtype);

        // Use raw_ops namespace to access GRUBlockCell
        std::vector<tensorflow::Output> outputs;
        tensorflow::Status status;
        
        // Create the operation using NodeBuilder
        tensorflow::NodeBuilder node_builder("GRUBlockCell", "GRUBlockCell");
        node_builder.Input(x.node())
                   .Input(h_prev.node())
                   .Input(w_ru.node())
                   .Input(w_c.node())
                   .Input(b_ru.node())
                   .Input(b_c.node());
        
        tensorflow::Node* node;
        status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create GRUBlockCell node: " + status.ToString(), data, size);
            return -1;
        }
        
        // Get the outputs from the node
        for (int i = 0; i < 4; ++i) {
            outputs.push_back(tensorflow::Output(node, i));
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({{x, x_tensor},
                             {h_prev, h_prev_tensor},
                             {w_ru, w_ru_tensor},
                             {w_c, w_c_tensor},
                             {b_ru, b_ru_tensor},
                             {b_c, b_c_tensor}},
                           outputs,
                           &output_tensors);
        
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