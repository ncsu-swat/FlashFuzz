#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>

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
  switch (selector % 4) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_HALF;
      break;
    case 3:
      dtype = tensorflow::DT_BFLOAT16;
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
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        uint8_t alpha_rank = parseRank(data[offset++]);
        std::vector<int64_t> alpha_dims = parseShape(data, offset, size, alpha_rank);

        if (offset >= size) {
            return 0;
        }

        tensorflow::TensorShape shape_tensor_shape(shape_dims);
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        fillTensorWithData<int32_t>(shape_tensor, data, offset, size);

        tensorflow::TensorShape alpha_tensor_shape(alpha_dims);
        tensorflow::Tensor alpha_tensor(dtype, alpha_tensor_shape);
        fillTensorWithDataByType(alpha_tensor, dtype, data, offset, size);

        int64_t seed_val = 0;
        int64_t seed2_val = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed2_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }

        std::cout << "Shape tensor: ";
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (int i = 0; i < shape_flat.size(); ++i) {
            std::cout << shape_flat(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Alpha tensor dtype: " << tensorflow::DataTypeString(dtype) << std::endl;
        std::cout << "Alpha tensor shape: ";
        for (int i = 0; i < alpha_tensor.dims(); ++i) {
            std::cout << alpha_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Seed: " << seed_val << ", Seed2: " << seed2_val << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto alpha_op = tensorflow::ops::Const(root, alpha_tensor);
        
        auto random_gamma = tensorflow::ops::RandomGamma(root, shape_op, alpha_op,
                                                        tensorflow::ops::RandomGamma::Seed(seed_val).Seed2(seed2_val).Dtype(dtype));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({random_gamma}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "RandomGamma operation successful. Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "RandomGamma operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}