#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/public/session.h"

// Define constants expected by helpers
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 100

namespace tensorflow {

// Helper: Parse DataType
DataType parseDataType(uint8_t selector) {
  DataType dtype;
  switch (selector % 23) {
    case 0: dtype = DT_FLOAT; break;
    case 1: dtype = DT_DOUBLE; break;
    case 2: dtype = DT_INT32; break;
    case 3: dtype = DT_UINT8; break;
    case 4: dtype = DT_INT16; break;
    case 5: dtype = DT_INT8; break;
    case 6: dtype = DT_STRING; break;
    case 7: dtype = DT_COMPLEX64; break;
    case 8: dtype = DT_INT64; break;
    case 9: dtype = DT_BOOL; break;
    case 10: dtype = DT_QINT8; break;
    case 11: dtype = DT_QUINT8; break;
    case 12: dtype = DT_QINT32; break;
    case 13: dtype = DT_BFLOAT16; break;
    case 14: dtype = DT_QINT16; break;
    case 15: dtype = DT_QUINT16; break;
    case 16: dtype = DT_UINT16; break;
    case 17: dtype = DT_COMPLEX128; break;
    case 18: dtype = DT_HALF; break;
    case 19: dtype = DT_UINT32; break;
    case 20: dtype = DT_UINT64; break;
    default: dtype = DT_FLOAT; break;
  }
  return dtype;
}

// Helper: Parse Rank
uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

// Helper: Parse Shape
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
            
            // Map arbitrary int64 to valid range
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

// Helper: Fill Tensor
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

// Specialized for tstring to avoid memcpy on non-trivial type
template <>
void fillTensorWithData<tensorflow::tstring>(tensorflow::Tensor& tensor, const uint8_t* data,
                                             size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t len = data[offset] % 32; // Limit string size
            offset++;
            if (offset + len <= total_size) {
                flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
                offset += len;
            } else {
                flat(i) = "fuzz";
            }
        } else {
            flat(i) = "";
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_STRING:
      fillTensorWithData<tensorflow::tstring>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

} // namespace tensorflow

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Need at least enough data for selector bytes
    if (Size < 4) {
        return 0;
    }

    size_t offset = 0;

    // Parse params
    uint8_t dtype_byte = Data[offset++];
    uint8_t transpose_a_byte = Data[offset++];
    uint8_t transpose_b_byte = Data[offset++];
    uint8_t rank_byte_a = Data[offset++];
    
    // MatMul requires a and b to have same type.
    tensorflow::DataType dtype = tensorflow::parseDataType(dtype_byte);
    bool transpose_a = transpose_a_byte & 0x01;
    bool transpose_b = transpose_b_byte & 0x01;
    
    // Parse Shape A
    uint8_t rank_a = tensorflow::parseRank(rank_byte_a);
    std::vector<int64_t> shape_vec_a = tensorflow::parseShape(Data, offset, Size, rank_a);
    tensorflow::TensorShape shape_a(shape_vec_a);

    // Create Tensor A
    tensorflow::Tensor tensor_a(dtype, shape_a);
    // Fill Tensor A safely
    tensorflow::fillTensorWithDataByType(tensor_a, dtype, Data, offset, Size);

    // Parse Shape B
    // Re-check size for rank byte B
    if (offset >= Size) { 
        return 0; 
    }
    uint8_t rank_byte_b = Data[offset++];
    uint8_t rank_b = tensorflow::parseRank(rank_byte_b);
    std::vector<int64_t> shape_vec_b = tensorflow::parseShape(Data, offset, Size, rank_b);
    tensorflow::TensorShape shape_b(shape_vec_b);

    // Create Tensor B
    tensorflow::Tensor tensor_b(dtype, shape_b);
    // Fill Tensor B safely
    tensorflow::fillTensorWithDataByType(tensor_b, dtype, Data, offset, Size);

    // Build the graph
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    
    // MatMul Op
    // Note: grad_a and grad_b are often internal args or handled by gradients, 
    // not standard OpKernel attributes, so we focus on standard transpose attributes.
    auto op = tensorflow::ops::MatMul(root, tensor_a, tensor_b, 
                                      tensorflow::ops::MatMul::TransposeA(transpose_a)
                                      .TransposeB(transpose_b));

    // Run the session
    tensorflow::ClientSession session(root);
    std::vector<tensorflow::Tensor> outputs;
    
    // Execute and catch exceptions/statuses
    // MatMul will fail gracefully if shapes/types are incompatible (e.g. string type or rank!=2)
    tensorflow::Status status = session.Run({op}, &outputs);

    if (!status.ok()) {
        // Just consume the status for debugging if needed, but for fuzzing we assume
        // validity checks inside TF are working if they return a bad status instead of crashing.
        // std::cerr << "TF Error: " << status.ToString() << std::endl;
    }

    return 0;
}