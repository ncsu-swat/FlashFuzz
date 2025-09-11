#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseGradDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
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

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType grad_dtype = parseGradDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t grad_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        uint8_t segment_ids_rank = parseRank(data[offset++]);
        
        if (indices_rank > 1 || segment_ids_rank > 1) {
            indices_rank = 1;
            segment_ids_rank = 1;
        }
        
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        std::vector<int64_t> segment_ids_shape = parseShape(data, offset, size, segment_ids_rank);
        
        if (indices_shape.empty()) indices_shape.push_back(1);
        if (segment_ids_shape.empty()) segment_ids_shape.push_back(1);
        if (grad_shape.empty()) grad_shape.push_back(1);
        
        if (indices_shape[0] != segment_ids_shape[0]) {
            segment_ids_shape[0] = indices_shape[0];
        }
        
        tensorflow::TensorShape grad_tensor_shape;
        for (auto dim : grad_shape) {
            grad_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape indices_tensor_shape;
        for (auto dim : indices_shape) {
            indices_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape segment_ids_tensor_shape;
        for (auto dim : segment_ids_shape) {
            segment_ids_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor grad_tensor(grad_dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, segment_ids_tensor_shape);
        
        fillTensorWithDataByType(grad_tensor, grad_dtype, data, offset, size);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        fillTensorWithDataByType(segment_ids_tensor, segment_ids_dtype, data, offset, size);
        
        int32_t output_dim0_value = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&output_dim0_value, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            output_dim0_value = std::abs(output_dim0_value) % 100 + 1;
        }
        
        tensorflow::Tensor output_dim0_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        output_dim0_tensor.scalar<int32_t>()() = output_dim0_value;
        
        auto grad_input = tensorflow::ops::Const(root, grad_tensor);
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto segment_ids_input = tensorflow::ops::Const(root, segment_ids_tensor);
        auto output_dim0_input = tensorflow::ops::Const(root, output_dim0_tensor);
        
        auto sparse_segment_sum_grad = tensorflow::ops::SparseSegmentSumGrad(
            root, grad_input, indices_input, segment_ids_input, output_dim0_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_segment_sum_grad}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
