#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype = tensorflow::DT_FLOAT;
    return dtype;
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_features = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::Output> float_values;
        std::vector<tensorflow::Tensor> feature_tensors;
        
        for (uint8_t i = 0; i < num_features; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = 1;
            std::vector<int64_t> shape = {static_cast<int64_t>((data[offset++] % 10) + 1)};
            
            tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(tensor, tensorflow::DT_FLOAT, data, offset, size);
            feature_tensors.push_back(tensor);
            
            auto placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
            float_values.push_back(placeholder);
        }
        
        if (float_values.empty()) {
            tensorflow::Tensor default_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
            default_tensor.flat<float>()(0) = 1.0f;
            feature_tensors.push_back(default_tensor);
            
            auto placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
            float_values.push_back(placeholder);
        }
        
        std::vector<int64_t> weights_shape = {static_cast<int64_t>(float_values.size())};
        tensorflow::Tensor example_weights_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(weights_shape));
        fillTensorWithDataByType(example_weights_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto example_weights = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        tensorflow::Tensor epsilon_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + sizeof(float) <= size) {
            float eps_val;
            std::memcpy(&eps_val, data + offset, sizeof(float));
            offset += sizeof(float);
            eps_val = std::abs(eps_val);
            if (eps_val == 0.0f || eps_val > 1.0f) eps_val = 0.1f;
            epsilon_tensor.scalar<float>()() = eps_val;
        } else {
            epsilon_tensor.scalar<float>()() = 0.1f;
        }
        auto epsilon = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Use raw_ops directly instead of the missing boosted_trees_ops.h
        auto result = tensorflow::ops::Operation(
            root.WithOpName("BoostedTreesMakeQuantileSummaries"),
            "BoostedTreesMakeQuantileSummaries",
            float_values,
            {example_weights},
            {epsilon}
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        
        for (size_t i = 0; i < float_values.size(); ++i) {
            feed_dict.push_back({float_values[i].node()->name(), feature_tensors[i]});
        }
        
        feed_dict.push_back({example_weights.node()->name(), example_weights_tensor});
        feed_dict.push_back({epsilon.node()->name(), epsilon_tensor});
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, {result.output(0)}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}