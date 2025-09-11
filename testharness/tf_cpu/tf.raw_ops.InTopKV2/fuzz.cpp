#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include <iostream>
#include <cstring>
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

tensorflow::DataType parseDataTypeForTargets(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t predictions_rank = parseRank(data[offset++]);
        if (predictions_rank < 2) predictions_rank = 2;
        
        std::vector<int64_t> predictions_shape = parseShape(data, offset, size, predictions_rank);
        if (predictions_shape.size() < 2) {
            predictions_shape = {2, 3};
        }
        
        int64_t batch_size = predictions_shape[0];
        
        tensorflow::Tensor predictions_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(predictions_shape));
        fillTensorWithDataByType(predictions_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        std::vector<int64_t> targets_shape = {batch_size};
        tensorflow::DataType targets_dtype = parseDataTypeForTargets(data[offset % size]);
        offset++;
        
        tensorflow::Tensor targets_tensor(targets_dtype, tensorflow::TensorShape(targets_shape));
        fillTensorWithDataByType(targets_tensor, targets_dtype, data, offset, size);
        
        tensorflow::DataType k_dtype = targets_dtype;
        tensorflow::Tensor k_tensor(k_dtype, tensorflow::TensorShape({}));
        
        if (k_dtype == tensorflow::DT_INT32) {
            int32_t k_val = 1;
            if (offset < size) {
                std::memcpy(&k_val, data + offset, std::min(sizeof(int32_t), size - offset));
                k_val = std::abs(k_val) % 10 + 1;
            }
            k_tensor.scalar<int32_t>()() = k_val;
        } else {
            int64_t k_val = 1;
            if (offset < size) {
                std::memcpy(&k_val, data + offset, std::min(sizeof(int64_t), size - offset));
                k_val = std::abs(k_val) % 10 + 1;
            }
            k_tensor.scalar<int64_t>()() = k_val;
        }

        auto predictions_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto targets_placeholder = tensorflow::ops::Placeholder(root, targets_dtype);
        auto k_placeholder = tensorflow::ops::Placeholder(root, k_dtype);

        auto in_top_k_op = tensorflow::ops::InTopKV2(root, predictions_placeholder, targets_placeholder, k_placeholder);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{predictions_placeholder, predictions_tensor},
                                                 {targets_placeholder, targets_tensor},
                                                 {k_placeholder, k_tensor}},
                                               {in_top_k_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
