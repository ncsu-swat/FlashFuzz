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
    case tensorflow::DT_QUINT8:
      fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT32:
      fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 3) {  
    case 0:
      dtype = tensorflow::DT_QUINT8;
      break;
    case 1:
      dtype = tensorflow::DT_QINT8;
      break;
    case 2:
      dtype = tensorflow::DT_QINT32;
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
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseQuantizedDataType(data[offset++]);
        
        uint8_t input_rank = 4;
        uint8_t filter_rank = 4;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);
        
        if (input_shape.size() != 4 || filter_shape.size() != 4) {
            return 0;
        }
        
        filter_shape[3] = input_shape[3];
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);
        tensorflow::TensorShape min_input_shape({});
        tensorflow::TensorShape max_input_shape({});
        tensorflow::TensorShape min_filter_shape({});
        tensorflow::TensorShape max_filter_shape({});
        
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor filter_tensor(filter_dtype, filter_tensor_shape);
        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, min_input_shape);
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, max_input_shape);
        tensorflow::Tensor min_filter_tensor(tensorflow::DT_FLOAT, min_filter_shape);
        tensorflow::Tensor max_filter_tensor(tensorflow::DT_FLOAT, max_filter_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        
        min_input_tensor.scalar<float>()() = -1.0f;
        max_input_tensor.scalar<float>()() = 1.0f;
        min_filter_tensor.scalar<float>()() = -1.0f;
        max_filter_tensor.scalar<float>()() = 1.0f;
        
        std::vector<int32_t> strides = {1, 1, 1, 1};
        std::string padding = "VALID";
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Filter tensor shape: ";
        for (int i = 0; i < filter_tensor.dims(); ++i) {
            std::cout << filter_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, filter_dtype);
        auto min_input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto min_filter_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_filter_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto quantized_conv2d = tensorflow::ops::QuantizedConv2D(
            root,
            input_placeholder,
            filter_placeholder,
            min_input_placeholder,
            max_input_placeholder,
            min_filter_placeholder,
            max_filter_placeholder,
            strides,
            padding
        );
        
        tensorflow::GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {filter_placeholder.node()->name(), filter_tensor},
            {min_input_placeholder.node()->name(), min_input_tensor},
            {max_input_placeholder.node()->name(), max_input_tensor},
            {min_filter_placeholder.node()->name(), min_filter_tensor},
            {max_filter_placeholder.node()->name(), max_filter_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            quantized_conv2d.output.node()->name(),
            quantized_conv2d.min_output.node()->name(),
            quantized_conv2d.max_output.node()->name()
        };
        
        tensorflow::Status status = session->Run(inputs, output_names, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "QuantizedConv2D executed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "QuantizedConv2D execution failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}