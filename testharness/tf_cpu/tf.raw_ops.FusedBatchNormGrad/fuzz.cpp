#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t y_backprop_rank = 4;
        std::vector<int64_t> y_backprop_shape = parseShape(data, offset, size, y_backprop_rank);
        
        uint8_t x_rank = 4;
        std::vector<int64_t> x_shape = y_backprop_shape;
        
        int64_t channel_dim = (data[offset % size] % 2 == 0) ? y_backprop_shape[3] : y_backprop_shape[1];
        offset++;
        
        std::vector<int64_t> scale_shape = {channel_dim};
        std::vector<int64_t> reserve_space_1_shape = {channel_dim};
        std::vector<int64_t> reserve_space_2_shape = {channel_dim};
        
        tensorflow::Tensor y_backprop_tensor(dtype, tensorflow::TensorShape(y_backprop_shape));
        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape(x_shape));
        tensorflow::Tensor scale_tensor(dtype, tensorflow::TensorShape(scale_shape));
        tensorflow::Tensor reserve_space_1_tensor(dtype, tensorflow::TensorShape(reserve_space_1_shape));
        tensorflow::Tensor reserve_space_2_tensor(dtype, tensorflow::TensorShape(reserve_space_2_shape));
        
        fillTensorWithDataByType(y_backprop_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(scale_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(reserve_space_1_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(reserve_space_2_tensor, dtype, data, offset, size);
        
        auto y_backprop_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto x_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto scale_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto reserve_space_1_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto reserve_space_2_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        float epsilon = 0.0001f;
        if (offset < size) {
            float eps_val;
            std::memcpy(&eps_val, data + offset, sizeof(float));
            offset += sizeof(float);
            epsilon = std::abs(eps_val);
            if (epsilon == 0.0f) epsilon = 0.0001f;
        }
        
        std::string data_format = "NHWC";
        if (offset < size && data[offset] % 2 == 1) {
            data_format = "NCHW";
        }
        offset++;
        
        bool is_training = true;
        if (offset < size) {
            is_training = (data[offset] % 2 == 0);
        }
        offset++;
        
        auto fused_batch_norm_grad = tensorflow::ops::FusedBatchNormGrad(
            root,
            y_backprop_placeholder,
            x_placeholder,
            scale_placeholder,
            reserve_space_1_placeholder,
            reserve_space_2_placeholder,
            tensorflow::ops::FusedBatchNormGrad::Attrs()
                .Epsilon(epsilon)
                .DataFormat(data_format)
                .IsTraining(is_training)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{y_backprop_placeholder, y_backprop_tensor},
             {x_placeholder, x_tensor},
             {scale_placeholder, scale_tensor},
             {reserve_space_1_placeholder, reserve_space_1_tensor},
             {reserve_space_2_placeholder, reserve_space_2_tensor}},
            {fused_batch_norm_grad.x_backprop,
             fused_batch_norm_grad.scale_backprop,
             fused_batch_norm_grad.offset_backprop,
             fused_batch_norm_grad.reserve_space_3,
             fused_batch_norm_grad.reserve_space_4},
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