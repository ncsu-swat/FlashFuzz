#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/platform/env.h>
#include <vector>
#include <algorithm>

constexpr uint8_t MIN_RANK = 2;
constexpr uint8_t MAX_RANK = 6;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 100;

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
    case tensorflow::DT_STRING:
      {
        auto flat = tensor.flat<tensorflow::tstring>();
        const size_t num_elements = flat.size();
        for (size_t i = 0; i < num_elements; ++i) {
          if (offset < total_size) {
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
              str += static_cast<char>(data[offset]);
              offset++;
            }
            flat(i) = str;
          } else {
            flat(i) = "";
          }
        }
      }
      break;
    default:
      break;
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 15) {  
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
      dtype = tensorflow::DT_STRING;
      break;
    case 7:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 8:
      dtype = tensorflow::DT_INT64;
      break;
    case 9:
      dtype = tensorflow::DT_BOOL;
      break;
    case 10:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 11:
      dtype = tensorflow::DT_UINT16;
      break;
    case 12:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 13:
      dtype = tensorflow::DT_HALF;
      break;
    case 14:
      dtype = tensorflow::DT_UINT32;
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

        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        uint8_t sparse_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> sparse_shape_vec = parseShape(data, offset, size, sparse_rank);
        
        if (sparse_shape_vec.empty() || sparse_shape_vec[0] <= 0) {
            return 0;
        }
        
        int64_t minibatch_size = sparse_shape_vec[0];
        int64_t num_indices = std::min(static_cast<int64_t>(10), minibatch_size * 2);
        
        tensorflow::TensorShape indices_shape({num_indices, sparse_rank});
        tensorflow::TensorShape values_shape({num_indices});
        tensorflow::TensorShape shape_shape({sparse_rank});
        
        tensorflow::Tensor sparse_indices(tensorflow::DT_INT64, indices_shape);
        tensorflow::Tensor sparse_values(values_dtype, values_shape);
        tensorflow::Tensor sparse_shape(tensorflow::DT_INT64, shape_shape);
        
        auto indices_flat = sparse_indices.flat<int64_t>();
        for (int64_t i = 0; i < num_indices; ++i) {
            for (int64_t j = 0; j < sparse_rank; ++j) {
                if (j == 0) {
                    indices_flat(i * sparse_rank + j) = i % minibatch_size;
                } else {
                    indices_flat(i * sparse_rank + j) = (i + j) % sparse_shape_vec[j];
                }
            }
        }
        
        std::sort(indices_flat.data(), indices_flat.data() + indices_flat.size());
        
        fillTensorWithDataByType(sparse_values, values_dtype, data, offset, size);
        
        auto shape_flat = sparse_shape.flat<int64_t>();
        for (int64_t i = 0; i < sparse_rank; ++i) {
            shape_flat(i) = sparse_shape_vec[i];
        }
        
        std::cout << "sparse_indices shape: ";
        for (int i = 0; i < sparse_indices.shape().dims(); ++i) {
            std::cout << sparse_indices.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "sparse_values shape: ";
        for (int i = 0; i < sparse_values.shape().dims(); ++i) {
            std::cout << sparse_values.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "sparse_shape shape: ";
        for (int i = 0; i < sparse_shape.shape().dims(); ++i) {
            std::cout << sparse_shape.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto sparse_indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto sparse_values_placeholder = tensorflow::ops::Placeholder(root, values_dtype);
        auto sparse_shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        std::string container = "";
        std::string shared_name = "";
        
        auto add_many_sparse = tensorflow::ops::AddManySparseToTensorsMap(
            root, sparse_indices_placeholder, sparse_values_placeholder, sparse_shape_placeholder,
            tensorflow::ops::AddManySparseToTensorsMap::Container(container)
                .SharedName(shared_name));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{sparse_indices_placeholder, sparse_indices},
             {sparse_values_placeholder, sparse_values},
             {sparse_shape_placeholder, sparse_shape}},
            {add_many_sparse}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation succeeded, output shape: ";
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