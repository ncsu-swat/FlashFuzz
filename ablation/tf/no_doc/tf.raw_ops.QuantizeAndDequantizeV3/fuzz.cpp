#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/lib/bfloat16/bfloat16.h>
#include <unsupported/Eigen/CXX11/Tensor>

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
            dtype = tensorflow::DT_BFLOAT16;
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        int32_t num_bits = 8;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_bits, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_bits = std::abs(num_bits) % 16 + 1;
        }
        
        bool range_given = false;
        if (offset < size) {
            range_given = (data[offset++] % 2) == 1;
        }
        
        float input_min = -6.0f;
        float input_max = 6.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&input_min, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&input_max, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (input_min > input_max) {
            std::swap(input_min, input_max);
        }
        
        bool narrow_range = false;
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }
        
        int32_t axis = -1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&axis, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            if (input_rank > 0) {
                axis = axis % static_cast<int32_t>(input_rank);
            }
        }
        
        std::cout << "Input tensor shape: ";
        for (int64_t dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Input dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "num_bits: " << num_bits << std::endl;
        std::cout << "range_given: " << range_given << std::endl;
        std::cout << "input_min: " << input_min << std::endl;
        std::cout << "input_max: " << input_max << std::endl;
        std::cout << "narrow_range: " << narrow_range << std::endl;
        std::cout << "axis: " << axis << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto input_min_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto input_max_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_tensor.scalar<float>()() = input_min;
        
        tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_tensor.scalar<float>()() = input_max;
        
        auto quantize_op = tensorflow::ops::QuantizeAndDequantizeV3(
            root,
            input_placeholder,
            input_min_placeholder,
            input_max_placeholder,
            tensorflow::ops::QuantizeAndDequantizeV3::NumBits(num_bits)
                .RangeGiven(range_given)
                .NarrowRange(narrow_range)
                .Axis(axis)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {input_min_placeholder, min_tensor},
             {input_max_placeholder, max_tensor}},
            {quantize_op},
            &outputs
        );
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}