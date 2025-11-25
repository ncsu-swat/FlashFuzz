#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/client/client_session.h"

// Constraints for tensor generation
#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace {

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 23) {
    case 0: dtype = tensorflow::DT_FLOAT; break;
    case 1: dtype = tensorflow::DT_DOUBLE; break;
    case 2: dtype = tensorflow::DT_INT32; break;
    case 3: dtype = tensorflow::DT_UINT8; break;
    case 4: dtype = tensorflow::DT_INT16; break;
    case 5: dtype = tensorflow::DT_INT8; break;
    case 6: dtype = tensorflow::DT_STRING; break;
    case 7: dtype = tensorflow::DT_COMPLEX64; break;
    case 8: dtype = tensorflow::DT_INT64; break;
    case 9: dtype = tensorflow::DT_BOOL; break;
    case 10: dtype = tensorflow::DT_QINT8; break;
    case 11: dtype = tensorflow::DT_QUINT8; break;
    case 12: dtype = tensorflow::DT_QINT32; break;
    case 13: dtype = tensorflow::DT_BFLOAT16; break;
    case 14: dtype = tensorflow::DT_QINT16; break;
    case 15: dtype = tensorflow::DT_QUINT16; break;
    case 16: dtype = tensorflow::DT_UINT16; break;
    case 17: dtype = tensorflow::DT_COMPLEX128; break;
    case 18: dtype = tensorflow::DT_HALF; break;
    case 19: dtype = tensorflow::DT_UINT32; break;
    case 20: dtype = tensorflow::DT_UINT64; break;
    default: dtype = tensorflow::DT_FLOAT; break;
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
            
            // Constrain dimensions to prevent OOM during fuzzing
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

void fillTensorWithString(tensorflow::Tensor& tensor, const uint8_t* data,
                          size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tensorflow::tstring>();
  const size_t num_elements = flat.size();

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset + 1 <= total_size) {
        // Use a small random length for strings
        size_t len = data[offset] % 16; 
        offset++;
        if (offset + len <= total_size) {
            flat(i) = tensorflow::tstring(reinterpret_cast<const char*>(data + offset), len);
            offset += len;
        } else {
            flat(i) = tensorflow::tstring("");
        }
    } else {
      flat(i) = tensorflow::tstring("");
    }
  }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case tensorflow::DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size); break;
    case tensorflow::DT_STRING:
      fillTensorWithString(tensor, data, offset, total_size); break;
    default:
      break;
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 8) return 0; // Require minimal size for basic headers

    size_t offset = 0;
    
    // 1. Construct 'input' tensor
    // Parse DataType
    uint8_t dtype_byte = data[offset++];
    tensorflow::DataType input_dtype = parseDataType(dtype_byte);
    
    // Parse Rank
    uint8_t rank_byte = data[offset++];
    uint8_t rank = parseRank(rank_byte);
    
    // Parse Shape
    std::vector<int64_t> input_shape_vec = parseShape(data, offset, size, rank);
    tensorflow::TensorShape input_shape(input_shape_vec);
    
    // Create and fill Input Tensor
    tensorflow::Tensor input_tensor(input_dtype, input_shape);
    fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);

    // 2. Construct 'begin' and 'size' tensors
    // These must be 1-D tensors of type int32 or int64.
    // Length must be equal to 'rank' of input tensor.
    
    if (offset >= size) return 0;
    
    // Choose index type (int32 vs int64)
    uint8_t idx_type_selector = data[offset++];
    tensorflow::DataType idx_dtype = (idx_type_selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
    
    // Shape for begin/size: [rank]
    tensorflow::TensorShape idx_tensor_shape({static_cast<int64_t>(rank)});
    
    tensorflow::Tensor begin_tensor(idx_dtype, idx_tensor_shape);
    tensorflow::Tensor size_tensor(idx_dtype, idx_tensor_shape);
    
    // Fill begin/size with fuzz data. 
    // Random values allow testing out-of-bounds, negative start indices, etc.
    fillTensorWithDataByType(begin_tensor, idx_dtype, data, offset, size);
    fillTensorWithDataByType(size_tensor, idx_dtype, data, offset, size);
    
    // 3. Build and Run the Graph
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    
    auto input_node = tensorflow::ops::Const(root, input_tensor);
    auto begin_node = tensorflow::ops::Const(root, begin_tensor);
    auto size_node = tensorflow::ops::Const(root, size_tensor);
    
    auto slice_op = tensorflow::ops::Slice(root, input_node, begin_node, size_node);
    
    tensorflow::ClientSession session(root);
    std::vector<tensorflow::Tensor> outputs;
    
    try {
        tensorflow::Status s = session.Run({slice_op}, &outputs);
        // We catch exceptions to keep fuzzing, and print them to verify activity.
        // Status errors are expected for invalid slices (which fuzzing generates).
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
    }

    return 0;
}