#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
      return;
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 3) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 2:
      dtype = tensorflow::DT_HALF;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        tensorflow::DataType y_backprop_dtype = parseDataType(data[offset++]);
        tensorflow::DataType x_dtype = y_backprop_dtype;
        
        uint8_t y_backprop_rank = parseRank(data[offset++]);
        if (y_backprop_rank != 4) return 0;
        
        std::vector<int64_t> y_backprop_shape = parseShape(data, offset, size, y_backprop_rank);
        std::vector<int64_t> x_shape = y_backprop_shape;
        
        int64_t channels = 1;
        if (offset < size) {
            uint8_t data_format_selector = data[offset++];
            if (data_format_selector % 2 == 0) {
                channels = y_backprop_shape[3];
            } else {
                channels = y_backprop_shape[1];
            }
        }
        
        std::vector<int64_t> scale_shape = {channels};
        std::vector<int64_t> reserve_space_shape = {channels};
        
        tensorflow::TensorShape y_backprop_tensor_shape(y_backprop_shape);
        tensorflow::TensorShape x_tensor_shape(x_shape);
        tensorflow::TensorShape scale_tensor_shape(scale_shape);
        tensorflow::TensorShape reserve_space_tensor_shape(reserve_space_shape);
        
        tensorflow::Tensor y_backprop_tensor(y_backprop_dtype, y_backprop_tensor_shape);
        tensorflow::Tensor x_tensor(x_dtype, x_tensor_shape);
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, scale_tensor_shape);
        tensorflow::Tensor reserve_space_1_tensor(tensorflow::DT_FLOAT, reserve_space_tensor_shape);
        tensorflow::Tensor reserve_space_2_tensor(tensorflow::DT_FLOAT, reserve_space_tensor_shape);
        tensorflow::Tensor reserve_space_3_tensor(tensorflow::DT_FLOAT, reserve_space_tensor_shape);
        
        fillTensorWithDataByType(y_backprop_tensor, y_backprop_dtype, data, offset, size);
        fillTensorWithDataByType(x_tensor, x_dtype, data, offset, size);
        fillTensorWithData<float>(scale_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_1_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_2_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_3_tensor, data, offset, size);
        
        float epsilon = 0.0001f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&epsilon, data + offset, sizeof(float));
            offset += sizeof(float);
            epsilon = std::abs(epsilon);
            if (epsilon < 1e-8f) epsilon = 0.0001f;
        }
        
        std::string data_format = "NHWC";
        if (offset < size) {
            uint8_t format_selector = data[offset++];
            if (format_selector % 2 == 1) {
                data_format = "NCHW";
            }
        }
        
        bool is_training = true;
        if (offset < size) {
            is_training = (data[offset++] % 2) == 1;
        }
        
        std::cout << "y_backprop shape: ";
        for (auto dim : y_backprop_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "x shape: ";
        for (auto dim : x_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "scale shape: ";
        for (auto dim : scale_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "epsilon: " << epsilon << std::endl;
        std::cout << "data_format: " << data_format << std::endl;
        std::cout << "is_training: " << is_training << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto y_backprop_placeholder = tensorflow::ops::Placeholder(root, y_backprop_dtype);
        auto x_placeholder = tensorflow::ops::Placeholder(root, x_dtype);
        auto scale_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_1_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_2_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_3_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto fused_batch_norm_grad = tensorflow::ops::FusedBatchNormGradV3(
            root,
            y_backprop_placeholder,
            x_placeholder,
            scale_placeholder,
            reserve_space_1_placeholder,
            reserve_space_2_placeholder,
            reserve_space_3_placeholder,
            tensorflow::ops::FusedBatchNormGradV3::Epsilon(epsilon)
                .DataFormat(data_format)
                .IsTraining(is_training)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {y_backprop_placeholder, y_backprop_tensor},
            {x_placeholder, x_tensor},
            {scale_placeholder, scale_tensor},
            {reserve_space_1_placeholder, reserve_space_1_tensor},
            {reserve_space_2_placeholder, reserve_space_2_tensor},
            {reserve_space_3_placeholder, reserve_space_3_tensor}
        }, {
            fused_batch_norm_grad.x_backprop,
            fused_batch_norm_grad.scale_backprop,
            fused_batch_norm_grad.offset_backprop,
            fused_batch_norm_grad.reserve_space_4,
            fused_batch_norm_grad.reserve_space_5
        }, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::cout << "Operation completed successfully" << std::endl;
        std::cout << "Number of outputs: " << outputs.size() << std::endl;
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "Output " << i << " shape: ";
            for (int j = 0; j < outputs[i].dims(); ++j) {
                std::cout << outputs[i].dim_size(j) << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}