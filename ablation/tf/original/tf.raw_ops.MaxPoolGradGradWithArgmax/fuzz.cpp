#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 6;
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
  switch (selector % 11) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_INT32;
      break;
    case 3:
      dtype = tensorflow::DT_UINT8;
      break;
    case 4:
      dtype = tensorflow::DT_INT16;
      break;
    case 5:
      dtype = tensorflow::DT_INT8;
      break;
    case 6:
      dtype = tensorflow::DT_INT64;
      break;
    case 7:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 8:
      dtype = tensorflow::DT_UINT16;
      break;
    case 9:
      dtype = tensorflow::DT_HALF;
      break;
    case 10:
      dtype = tensorflow::DT_UINT32;
      break;
    default:
      dtype = tensorflow::DT_UINT64;
      break;
  }
  return dtype;
}

tensorflow::DataType parseArgmaxDataType(uint8_t selector) {
  return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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

std::vector<int> parseKsizeStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> result(4);
    for (int i = 0; i < 4; ++i) {
        if (offset + sizeof(int) <= total_size) {
            int val;
            std::memcpy(&val, data + offset, sizeof(int));
            offset += sizeof(int);
            result[i] = std::abs(val) % 10 + 1;
        } else {
            result[i] = 1;
        }
    }
    return result;
}

std::string parsePadding(uint8_t selector) {
    return (selector % 2 == 0) ? "VALID" : "SAME";
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        tensorflow::DataType argmax_dtype = parseArgmaxDataType(data[offset++]);
        
        uint8_t input_rank = 4;
        std::vector<int64_t> input_shape = {2, 4, 4, 3};
        
        if (offset + 8 * input_rank <= size) {
            input_shape = parseShape(data, offset, size, input_rank);
            if (input_shape.size() != 4) {
                input_shape = {2, 4, 4, 3};
            }
        }

        std::vector<int64_t> grad_shape = input_shape;
        
        int64_t batch = input_shape[0];
        int64_t height = input_shape[1];
        int64_t width = input_shape[2];
        int64_t channels = input_shape[3];
        
        std::vector<int64_t> argmax_shape = {batch, height, width, channels};

        std::vector<int> ksize = parseKsizeStrides(data, offset, size);
        std::vector<int> strides = parseKsizeStrides(data, offset, size);
        
        std::string padding = parsePadding(data[offset++]);
        bool include_batch_in_index = (data[offset++] % 2 == 1);

        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor grad_tensor(input_dtype, tensorflow::TensorShape(grad_shape));
        tensorflow::Tensor argmax_tensor(argmax_dtype, tensorflow::TensorShape(argmax_shape));

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(argmax_tensor, argmax_dtype, data, offset, size);

        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.shape().dims(); ++i) {
            std::cout << input_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Grad tensor shape: ";
        for (int i = 0; i < grad_tensor.shape().dims(); ++i) {
            std::cout << grad_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Argmax tensor shape: ";
        for (int i = 0; i < argmax_tensor.shape().dims(); ++i) {
            std::cout << argmax_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Ksize: ";
        for (int val : ksize) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Strides: ";
        for (int val : strides) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Padding: " << padding << std::endl;
        std::cout << "Include batch in index: " << include_batch_in_index << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto argmax_placeholder = tensorflow::ops::Placeholder(root, argmax_dtype);

        auto max_pool_grad_grad = tensorflow::ops::MaxPoolGradGradWithArgmax(
            root, input_placeholder, grad_placeholder, argmax_placeholder,
            ksize, strides, padding,
            tensorflow::ops::MaxPoolGradGradWithArgmax::IncludeBatchInIndex(include_batch_in_index));

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {grad_placeholder.node()->name(), grad_tensor},
            {argmax_placeholder.node()->name(), argmax_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {max_pool_grad_grad.node()->name()}, {}, &outputs);

        if (status.ok() && !outputs.empty()) {
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