#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
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
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset,
                                                total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset,
                                                 total_size);
      break;
    default:
      break;
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 6) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 3:
      dtype = tensorflow::DT_HALF;
      break;
    case 4:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 5:
      dtype = tensorflow::DT_COMPLEX128;
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
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_sizes_shape = {4};
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(input_sizes_shape));
        fillTensorWithData<int32_t>(input_sizes_tensor, data, offset, size);
        
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, 4);
        if (filter_shape.size() != 4) {
            filter_shape = {1, 3, 3, 1};
        }
        tensorflow::Tensor filter_tensor(dtype, tensorflow::TensorShape(filter_shape));
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        
        std::vector<int64_t> out_backprop_shape = parseShape(data, offset, size, 4);
        if (out_backprop_shape.size() != 4) {
            out_backprop_shape = {1, 2, 2, 1};
        }
        tensorflow::Tensor out_backprop_tensor(dtype, tensorflow::TensorShape(out_backprop_shape));
        fillTensorWithDataByType(out_backprop_tensor, dtype, data, offset, size);
        
        std::vector<int32_t> strides = {1, 1, 1, 1};
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t stride_val;
                std::memcpy(&stride_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                strides[i] = std::abs(stride_val) % 3 + 1;
            }
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "SAME" : "VALID";
        
        std::string data_format = "NHWC";
        
        std::vector<int32_t> dilations = {1, 1, 1, 1};
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t dilation_val;
                std::memcpy(&dilation_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dilations[i] = std::abs(dilation_val) % 3 + 1;
            }
        }
        
        std::cout << "Input sizes shape: [";
        for (size_t i = 0; i < input_sizes_shape.size(); ++i) {
            std::cout << input_sizes_shape[i];
            if (i < input_sizes_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Filter shape: [";
        for (size_t i = 0; i < filter_shape.size(); ++i) {
            std::cout << filter_shape[i];
            if (i < filter_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Out backprop shape: [";
        for (size_t i = 0; i < out_backprop_shape.size(); ++i) {
            std::cout << out_backprop_shape[i];
            if (i < out_backprop_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Strides: [";
        for (size_t i = 0; i < strides.size(); ++i) {
            std::cout << strides[i];
            if (i < strides.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Padding: " << padding << std::endl;
        std::cout << "Data format: " << data_format << std::endl;
        
        std::cout << "Dilations: [";
        for (size_t i = 0; i < dilations.size(); ++i) {
            std::cout << dilations[i];
            if (i < dilations.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_sizes_op = tensorflow::ops::Const(root, input_sizes_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop_tensor);
        
        auto depthwise_conv2d_backprop_input = tensorflow::ops::DepthwiseConv2dNativeBackpropInput(
            root, input_sizes_op, filter_op, out_backprop_op, strides, padding,
            tensorflow::ops::DepthwiseConv2dNativeBackpropInput::DataFormat(data_format)
                .Dilations(dilations));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({depthwise_conv2d_backprop_input}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully" << std::endl;
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}