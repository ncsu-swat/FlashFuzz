#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/training_ops.h"
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
        std::cerr << message << std::endl;
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_weights_byte = data[offset++];
        int num_weights = (num_weights_byte % 5) + 1;
        
        std::vector<tensorflow::Output> weight_outputs;
        std::vector<tensorflow::Tensor> weight_tensors;
        
        for (int i = 0; i < num_weights; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (offset >= size) break;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor weight_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(weight_tensor, dtype, data, offset, size);
            weight_tensors.push_back(weight_tensor);
            
            auto weight_var = tensorflow::ops::Variable(root.WithOpName("weight_" + std::to_string(i)), 
                                                       tensor_shape, dtype);
            auto weight_assign = tensorflow::ops::Assign(root.WithOpName("weight_assign_" + std::to_string(i)), 
                                                        weight_var, weight_tensor);
            weight_outputs.push_back(weight_var);
        }
        
        if (weight_outputs.empty()) return 0;
        
        float l1_val = 0.1f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l1_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l1_val = std::abs(l1_val);
            if (l1_val > 10.0f) l1_val = 0.1f;
        }
        
        float l2_val = 0.1f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l2_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l2_val = std::abs(l2_val);
            if (l2_val > 10.0f || l2_val <= 0.0f) l2_val = 0.1f;
        }
        
        std::cout << "Number of weights: " << num_weights << std::endl;
        std::cout << "L1 regularization: " << l1_val << std::endl;
        std::cout << "L2 regularization: " << l2_val << std::endl;
        
        for (size_t i = 0; i < weight_tensors.size(); ++i) {
            std::cout << "Weight " << i << " shape: ";
            for (int j = 0; j < weight_tensors[i].shape().dims(); ++j) {
                std::cout << weight_tensors[i].shape().dim_size(j) << " ";
            }
            std::cout << std::endl;
        }
        
        // Use raw_ops namespace for SdcaShrinkL1
        auto sdca_shrink = tensorflow::ops::internal::SdcaShrinkL1(
            root.WithOpName("sdca_shrink"),
            weight_outputs,
            tensorflow::Input(l1_val),
            tensorflow::Input(l2_val));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({}, {}, {sdca_shrink.operation}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}