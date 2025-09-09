#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

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
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 7:
      dtype = tensorflow::DT_INT64;
      break;
    case 8:
      dtype = tensorflow::DT_BOOL;
      break;
    case 9:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 10:
      dtype = tensorflow::DT_UINT16;
      break;
    case 11:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 12:
      dtype = tensorflow::DT_HALF;
      break;
    case 13:
      dtype = tensorflow::DT_UINT32;
      break;
    case 14:
      dtype = tensorflow::DT_UINT64;
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
        
        uint8_t a_indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> a_indices_shape = parseShape(data, offset, size, a_indices_rank);
        
        uint8_t a_values_rank = parseRank(data[offset++]);
        std::vector<int64_t> a_values_shape = parseShape(data, offset, size, a_values_rank);
        
        uint8_t a_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> a_shape_shape = parseShape(data, offset, size, a_shape_rank);
        
        uint8_t b_indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_indices_shape = parseShape(data, offset, size, b_indices_rank);
        
        uint8_t b_values_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_values_shape = parseShape(data, offset, size, b_values_rank);
        
        uint8_t b_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_shape_shape = parseShape(data, offset, size, b_shape_rank);

        tensorflow::Tensor a_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(a_indices_shape));
        tensorflow::Tensor a_values_tensor(dtype, tensorflow::TensorShape(a_values_shape));
        tensorflow::Tensor a_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(a_shape_shape));
        tensorflow::Tensor b_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(b_indices_shape));
        tensorflow::Tensor b_values_tensor(dtype, tensorflow::TensorShape(b_values_shape));
        tensorflow::Tensor b_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(b_shape_shape));

        fillTensorWithData<int64_t>(a_indices_tensor, data, offset, size);
        fillTensorWithDataByType(a_values_tensor, dtype, data, offset, size);
        fillTensorWithData<int64_t>(a_shape_tensor, data, offset, size);
        fillTensorWithData<int64_t>(b_indices_tensor, data, offset, size);
        fillTensorWithDataByType(b_values_tensor, dtype, data, offset, size);
        fillTensorWithData<int64_t>(b_shape_tensor, data, offset, size);

        std::cout << "a_indices shape: ";
        for (auto dim : a_indices_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "a_values shape: ";
        for (auto dim : a_values_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "a_shape shape: ";
        for (auto dim : a_shape_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "b_indices shape: ";
        for (auto dim : b_indices_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "b_values shape: ";
        for (auto dim : b_values_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "b_shape shape: ";
        for (auto dim : b_shape_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto a_indices_op = tensorflow::ops::Const(root, a_indices_tensor);
        auto a_values_op = tensorflow::ops::Const(root, a_values_tensor);
        auto a_shape_op = tensorflow::ops::Const(root, a_shape_tensor);
        auto b_indices_op = tensorflow::ops::Const(root, b_indices_tensor);
        auto b_values_op = tensorflow::ops::Const(root, b_values_tensor);
        auto b_shape_op = tensorflow::ops::Const(root, b_shape_tensor);

        tensorflow::ops::SparseSparseMaximum sparse_sparse_maximum(root, 
                                                                   a_indices_op, 
                                                                   a_values_op, 
                                                                   a_shape_op,
                                                                   b_indices_op, 
                                                                   b_values_op, 
                                                                   b_shape_op);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_sparse_maximum.output_indices, 
                                                sparse_sparse_maximum.output_values, 
                                                sparse_sparse_maximum.output_shape}, &outputs);
        
        if (status.ok()) {
            std::cout << "SparseSparseMaximum operation executed successfully" << std::endl;
            std::cout << "Output indices shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "Output values shape: " << outputs[1].shape().DebugString() << std::endl;
            std::cout << "Output shape shape: " << outputs[2].shape().DebugString() << std::endl;
        } else {
            std::cout << "SparseSparseMaximum operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}