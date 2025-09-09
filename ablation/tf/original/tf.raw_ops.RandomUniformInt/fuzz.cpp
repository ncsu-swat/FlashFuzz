#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/random_ops.h>
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 2) {  
    case 0:
      dtype = tensorflow::DT_INT32;
      break;
    case 1:
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

        tensorflow::DataType minval_maxval_dtype = parseDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape(shape_dims);
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        fillTensorWithDataByType(shape_tensor, tensorflow::DT_INT32, data, offset, size);
        
        tensorflow::TensorShape minval_shape({});
        tensorflow::Tensor minval_tensor(minval_maxval_dtype, minval_shape);
        fillTensorWithDataByType(minval_tensor, minval_maxval_dtype, data, offset, size);
        
        tensorflow::TensorShape maxval_shape({});
        tensorflow::Tensor maxval_tensor(minval_maxval_dtype, maxval_shape);
        fillTensorWithDataByType(maxval_tensor, minval_maxval_dtype, data, offset, size);
        
        int seed = 0;
        int seed2 = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }

        std::cout << "Shape tensor: ";
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (int i = 0; i < shape_flat.size(); ++i) {
            std::cout << shape_flat(i) << " ";
        }
        std::cout << std::endl;

        if (minval_maxval_dtype == tensorflow::DT_INT32) {
            auto minval_flat = minval_tensor.flat<int32_t>();
            auto maxval_flat = maxval_tensor.flat<int32_t>();
            std::cout << "Minval: " << minval_flat(0) << ", Maxval: " << maxval_flat(0) << std::endl;
        } else {
            auto minval_flat = minval_tensor.flat<int64_t>();
            auto maxval_flat = maxval_tensor.flat<int64_t>();
            std::cout << "Minval: " << minval_flat(0) << ", Maxval: " << maxval_flat(0) << std::endl;
        }
        
        std::cout << "Seed: " << seed << ", Seed2: " << seed2 << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto minval_op = tensorflow::ops::Const(root, minval_tensor);
        auto maxval_op = tensorflow::ops::Const(root, maxval_tensor);
        
        auto random_uniform_int = tensorflow::ops::RandomUniformInt(
            root, shape_op, minval_op, maxval_op,
            tensorflow::ops::RandomUniformInt::Attrs().Seed(seed).Seed2(seed2));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({random_uniform_int}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "RandomUniformInt operation succeeded" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "RandomUniformInt operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}