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
    default:
      return;
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
      dtype = tensorflow::DT_INT32;
      break;
    case 3:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 4:
      dtype = tensorflow::DT_HALF;
      break;
    case 5:
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        
        if (input_rank < 4) {
            return 0;
        }
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        if (offset >= size) {
            return 0;
        }
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : input_shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        std::vector<int32_t> ksize_vec;
        std::vector<int32_t> strides_vec;
        
        for (int i = 0; i < input_rank && offset < size; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t ksize_val;
                std::memcpy(&ksize_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                ksize_val = 1 + (std::abs(ksize_val) % 5);
                ksize_vec.push_back(ksize_val);
            } else {
                ksize_vec.push_back(1);
            }
        }
        
        for (int i = 0; i < input_rank && offset < size; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t stride_val;
                std::memcpy(&stride_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                stride_val = 1 + (std::abs(stride_val) % 5);
                strides_vec.push_back(stride_val);
            } else {
                strides_vec.push_back(1);
            }
        }
        
        tensorflow::TensorShape ksize_shape({static_cast<int64_t>(ksize_vec.size())});
        tensorflow::Tensor ksize_tensor(tensorflow::DT_INT32, ksize_shape);
        auto ksize_flat = ksize_tensor.flat<int32_t>();
        for (size_t i = 0; i < ksize_vec.size(); ++i) {
            ksize_flat(i) = ksize_vec[i];
        }
        
        tensorflow::TensorShape strides_shape({static_cast<int64_t>(strides_vec.size())});
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, strides_shape);
        auto strides_flat = strides_tensor.flat<int32_t>();
        for (size_t i = 0; i < strides_vec.size(); ++i) {
            strides_flat(i) = strides_vec[i];
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "VALID" : "SAME";
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Ksize: ";
        for (int32_t k : ksize_vec) {
            std::cout << k << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Strides: ";
        for (int32_t s : strides_vec) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Padding: " << padding << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto ksize_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto strides_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        auto maxpool_op = tensorflow::ops::MaxPoolV2(root, input_placeholder, ksize_placeholder, strides_placeholder, padding);
        
        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {ksize_placeholder.node()->name(), ksize_tensor},
            {strides_placeholder.node()->name(), strides_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {maxpool_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MaxPoolV2 operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MaxPoolV2 operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}