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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        tensorflow::DataType filter_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_sizes_shape = {4};
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(input_sizes_shape));
        fillTensorWithDataByType(input_sizes_tensor, tensorflow::DT_INT32, data, offset, size);
        
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, 4);
        tensorflow::Tensor filter_tensor(filter_dtype, tensorflow::TensorShape(filter_shape));
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        
        std::vector<int64_t> out_backprop_shape = parseShape(data, offset, size, 4);
        tensorflow::Tensor out_backprop_tensor(filter_dtype, tensorflow::TensorShape(out_backprop_shape));
        fillTensorWithDataByType(out_backprop_tensor, filter_dtype, data, offset, size);
        
        std::vector<int> strides = {1, 1, 1, 1};
        if (offset + 16 <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t stride_val;
                std::memcpy(&stride_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                strides[i] = std::abs(stride_val) % 5 + 1;
            }
        }
        
        std::string padding = "VALID";
        if (offset < size) {
            uint8_t padding_selector = data[offset++];
            switch (padding_selector % 3) {
                case 0: padding = "VALID"; break;
                case 1: padding = "SAME"; break;
                case 2: padding = "EXPLICIT"; break;
            }
        }
        
        std::vector<int> explicit_paddings = {};
        if (padding == "EXPLICIT") {
            explicit_paddings = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < 8 && offset + sizeof(int32_t) <= size; ++i) {
                int32_t pad_val;
                std::memcpy(&pad_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                explicit_paddings[i] = std::abs(pad_val) % 10;
            }
        }
        
        std::string data_format = "NHWC";
        if (offset < size) {
            uint8_t format_selector = data[offset++];
            data_format = (format_selector % 2 == 0) ? "NHWC" : "NCHW";
        }
        
        std::vector<int> dilations = {1, 1, 1, 1};
        if (offset + 16 <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t dilation_val;
                std::memcpy(&dilation_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dilations[i] = std::abs(dilation_val) % 5 + 1;
            }
        }
        
        std::cout << "Input sizes shape: ";
        for (auto dim : input_sizes_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Filter shape: ";
        for (auto dim : filter_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Out backprop shape: ";
        for (auto dim : out_backprop_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Strides: ";
        for (auto s : strides) std::cout << s << " ";
        std::cout << std::endl;
        
        std::cout << "Padding: " << padding << std::endl;
        std::cout << "Data format: " << data_format << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_sizes_op = tensorflow::ops::Const(root, input_sizes_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop_tensor);
        
        tensorflow::ops::DepthwiseConv2dNativeBackpropInput::Attrs attrs;
        attrs = attrs.Padding(padding);
        attrs = attrs.DataFormat(data_format);
        attrs = attrs.Dilations(dilations);
        if (!explicit_paddings.empty()) {
            attrs = attrs.ExplicitPaddings(explicit_paddings);
        }
        
        auto result = tensorflow::ops::DepthwiseConv2dNativeBackpropInput(
            root, input_sizes_op, filter_op, out_backprop_op, strides, attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation completed successfully" << std::endl;
            std::cout << "Output shape: ";
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