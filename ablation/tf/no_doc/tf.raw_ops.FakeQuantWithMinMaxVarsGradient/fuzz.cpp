#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
            break;
        default:
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
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> gradients_shape = parseShape(data, offset, size, rank);
        std::vector<int64_t> inputs_shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape gradients_tensor_shape(gradients_shape);
        tensorflow::TensorShape inputs_tensor_shape(inputs_shape);
        tensorflow::TensorShape min_shape({});
        tensorflow::TensorShape max_shape({});
        
        tensorflow::Tensor gradients_tensor(dtype, gradients_tensor_shape);
        tensorflow::Tensor inputs_tensor(dtype, inputs_tensor_shape);
        tensorflow::Tensor min_tensor(dtype, min_shape);
        tensorflow::Tensor max_tensor(dtype, max_shape);
        
        fillTensorWithDataByType(gradients_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(inputs_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(min_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(max_tensor, dtype, data, offset, size);
        
        if (offset + 1 <= size) {
            uint8_t num_bits_byte = data[offset++];
            int num_bits = (num_bits_byte % 16) + 1;
            
            bool narrow_range = false;
            if (offset < size) {
                narrow_range = (data[offset++] % 2) == 1;
            }
            
            std::cout << "Gradients tensor shape: ";
            for (int i = 0; i < gradients_tensor.dims(); ++i) {
                std::cout << gradients_tensor.dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Inputs tensor shape: ";
            for (int i = 0; i < inputs_tensor.dims(); ++i) {
                std::cout << inputs_tensor.dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Min tensor shape: ";
            for (int i = 0; i < min_tensor.dims(); ++i) {
                std::cout << min_tensor.dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Max tensor shape: ";
            for (int i = 0; i < max_tensor.dims(); ++i) {
                std::cout << max_tensor.dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "num_bits: " << num_bits << std::endl;
            std::cout << "narrow_range: " << narrow_range << std::endl;
            std::cout << "dtype: " << tensorflow::DataTypeString(dtype) << std::endl;
            
            tensorflow::Scope root = tensorflow::Scope::NewRootScope();
            
            auto gradients_placeholder = tensorflow::ops::Placeholder(root, dtype);
            auto inputs_placeholder = tensorflow::ops::Placeholder(root, dtype);
            auto min_placeholder = tensorflow::ops::Placeholder(root, dtype);
            auto max_placeholder = tensorflow::ops::Placeholder(root, dtype);
            
            auto fake_quant_grad = tensorflow::ops::FakeQuantWithMinMaxVarsGradient(
                root, gradients_placeholder, inputs_placeholder, min_placeholder, max_placeholder,
                tensorflow::ops::FakeQuantWithMinMaxVarsGradient::Attrs().NumBits(num_bits).NarrowRange(narrow_range)
            );
            
            tensorflow::ClientSession session(root);
            
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session.Run(
                {{gradients_placeholder, gradients_tensor},
                 {inputs_placeholder, inputs_tensor},
                 {min_placeholder, min_tensor},
                 {max_placeholder, max_tensor}},
                {fake_quant_grad.backprops_wrt_input, fake_quant_grad.backprop_wrt_min, fake_quant_grad.backprop_wrt_max},
                &outputs
            );
            
            if (status.ok()) {
                std::cout << "Operation executed successfully" << std::endl;
                std::cout << "Number of outputs: " << outputs.size() << std::endl;
                for (size_t i = 0; i < outputs.size(); ++i) {
                    std::cout << "Output " << i << " shape: ";
                    for (int j = 0; j < outputs[i].dims(); ++j) {
                        std::cout << outputs[i].dim_size(j) << " ";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cout << "Operation failed: " << status.ToString() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}