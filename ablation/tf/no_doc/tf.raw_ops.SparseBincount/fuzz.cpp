#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
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
  switch (selector % 4) {  
    case 0:
      dtype = tensorflow::DT_INT32;
      break;
    case 1:
      dtype = tensorflow::DT_INT64;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        
        uint8_t dense_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> dense_shape_shape = parseShape(data, offset, size, dense_shape_rank);
        
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) {
            return 0;
        }
        
        int32_t size_val = 0;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&size_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            size_val = std::abs(size_val) % 100 + 1;
        } else {
            size_val = 10;
        }
        
        bool binary_output = (data[offset % size] % 2) == 0;
        offset++;

        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(dense_shape_shape));
        fillTensorWithDataByType(dense_shape_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        size_tensor.scalar<int32_t>()() = size_val;
        
        tensorflow::Tensor weights_tensor;
        bool has_weights = (data[offset % size] % 2) == 0;
        if (has_weights) {
            weights_tensor = tensorflow::Tensor(values_dtype, tensorflow::TensorShape(values_shape));
            fillTensorWithDataByType(weights_tensor, values_dtype, data, offset, size);
        }
        
        std::cout << "Indices tensor shape: ";
        for (int i = 0; i < indices_tensor.shape().dims(); ++i) {
            std::cout << indices_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Values tensor shape: ";
        for (int i = 0; i < values_tensor.shape().dims(); ++i) {
            std::cout << values_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Dense shape tensor shape: ";
        for (int i = 0; i < dense_shape_tensor.shape().dims(); ++i) {
            std::cout << dense_shape_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Size: " << size_val << std::endl;
        std::cout << "Binary output: " << binary_output << std::endl;
        std::cout << "Has weights: " << has_weights << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto values_placeholder = tensorflow::ops::Placeholder(root, values_dtype);
        auto dense_shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto size_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        tensorflow::ops::SparseBincount::Attrs attrs;
        attrs = attrs.BinaryOutput(binary_output);
        
        tensorflow::ops::SparseBincount sparse_bincount_op;
        if (has_weights) {
            auto weights_placeholder = tensorflow::ops::Placeholder(root, values_dtype);
            sparse_bincount_op = tensorflow::ops::SparseBincount(
                root, indices_placeholder, values_placeholder, dense_shape_placeholder, 
                size_placeholder, weights_placeholder, attrs);
        } else {
            sparse_bincount_op = tensorflow::ops::SparseBincount(
                root, indices_placeholder, values_placeholder, dense_shape_placeholder, 
                size_placeholder, attrs);
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        if (has_weights) {
            session.Run({{indices_placeholder, indices_tensor},
                        {values_placeholder, values_tensor},
                        {dense_shape_placeholder, dense_shape_tensor},
                        {size_placeholder, size_tensor},
                        {tensorflow::ops::Placeholder(root, values_dtype), weights_tensor}},
                       {sparse_bincount_op.output}, &outputs);
        } else {
            session.Run({{indices_placeholder, indices_tensor},
                        {values_placeholder, values_tensor},
                        {dense_shape_placeholder, dense_shape_tensor},
                        {size_placeholder, size_tensor}},
                       {sparse_bincount_op.output}, &outputs);
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}