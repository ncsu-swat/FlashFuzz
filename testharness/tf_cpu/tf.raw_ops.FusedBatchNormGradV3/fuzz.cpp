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
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType y_backprop_dtype = parseDataType(data[offset++]);
        
        uint8_t y_backprop_rank = parseRank(data[offset++]);
        if (y_backprop_rank != 4) y_backprop_rank = 4;
        
        std::vector<int64_t> y_backprop_shape = parseShape(data, offset, size, y_backprop_rank);
        
        tensorflow::Tensor y_backprop_tensor(y_backprop_dtype, tensorflow::TensorShape(y_backprop_shape));
        fillTensorWithDataByType(y_backprop_tensor, y_backprop_dtype, data, offset, size);
        
        tensorflow::Tensor x_tensor(y_backprop_dtype, tensorflow::TensorShape(y_backprop_shape));
        fillTensorWithDataByType(x_tensor, y_backprop_dtype, data, offset, size);
        
        int64_t channel_dim = 1;
        if (y_backprop_shape.size() >= 4) {
            channel_dim = y_backprop_shape[3];
        }
        
        std::vector<int64_t> scale_shape = {channel_dim};
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        fillTensorWithData<float>(scale_tensor, data, offset, size);
        
        tensorflow::Tensor reserve_space_1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        fillTensorWithData<float>(reserve_space_1_tensor, data, offset, size);
        
        tensorflow::Tensor reserve_space_2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        fillTensorWithData<float>(reserve_space_2_tensor, data, offset, size);
        
        tensorflow::Tensor reserve_space_3_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        fillTensorWithData<float>(reserve_space_3_tensor, data, offset, size);
        
        float epsilon = 0.0001f;
        if (offset < size) {
            std::memcpy(&epsilon, data + offset, std::min(sizeof(float), size - offset));
            offset += sizeof(float);
            epsilon = std::abs(epsilon);
            if (epsilon == 0.0f) epsilon = 0.0001f;
        }
        
        std::string data_format = "NHWC";
        if (offset < size && data[offset] % 2 == 1) {
            data_format = "NCHW";
        }
        offset++;
        
        bool is_training = true;
        if (offset < size && data[offset] % 2 == 0) {
            is_training = false;
        }
        offset++;

        auto y_backprop_op = tensorflow::ops::Const(root, y_backprop_tensor);
        auto x_op = tensorflow::ops::Const(root, x_tensor);
        auto scale_op = tensorflow::ops::Const(root, scale_tensor);
        auto reserve_space_1_op = tensorflow::ops::Const(root, reserve_space_1_tensor);
        auto reserve_space_2_op = tensorflow::ops::Const(root, reserve_space_2_tensor);
        auto reserve_space_3_op = tensorflow::ops::Const(root, reserve_space_3_tensor);

        auto fused_batch_norm_grad = tensorflow::ops::FusedBatchNormGradV3(
            root,
            y_backprop_op,
            x_op,
            scale_op,
            reserve_space_1_op,
            reserve_space_2_op,
            reserve_space_3_op,
            tensorflow::ops::FusedBatchNormGradV3::Epsilon(epsilon)
                .DataFormat(data_format)
                .IsTraining(is_training)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            fused_batch_norm_grad.x_backprop,
            fused_batch_norm_grad.scale_backprop,
            fused_batch_norm_grad.offset_backprop,
            fused_batch_norm_grad.reserve_space_4,
            fused_batch_norm_grad.reserve_space_5
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}