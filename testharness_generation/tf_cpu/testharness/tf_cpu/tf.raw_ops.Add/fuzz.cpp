#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/framework/numeric_types.h"

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 8

// Helper: Parse supported data type for Add
// Add supports: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, string
tensorflow::DataType parseDataType(uint8_t selector) {
  switch (selector % 12) {
    case 0: return tensorflow::DT_FLOAT;
    case 1: return tensorflow::DT_DOUBLE;
    case 2: return tensorflow::DT_INT32;
    case 3: return tensorflow::DT_UINT8;
    case 4: return tensorflow::DT_INT16;
    case 5: return tensorflow::DT_INT8;
    case 6: return tensorflow::DT_INT64;
    case 7: return tensorflow::DT_COMPLEX64;
    case 8: return tensorflow::DT_COMPLEX128;
    case 9: return tensorflow::DT_BFLOAT16;
    case 10: return tensorflow::DT_HALF;
    case 11: return tensorflow::DT_STRING;
    default: return tensorflow::DT_FLOAT;
  }
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
            
            // Constrain dimensions to avoid excessive memory usage
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

void fillTensorWithString(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t len = data[offset++] % 32; // Limit string length
            if (offset + len <= total_size) {
                flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
                offset += len;
            } else {
                size_t remaining = total_size - offset;
                flat(i) = std::string(reinterpret_cast<const char*>(data + offset), remaining);
                offset += remaining;
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
      fillTensorWithString(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Avoid extremely small inputs
    if (Size < 1) return 0;

    try {
        size_t offset = 0;

        // 1. Parse DataType (used for both x and y as required by Op)
        tensorflow::DataType dtype = parseDataType(Data[offset++]);

        // 2. Parse Shape for X
        if (offset >= Size) return 0;
        uint8_t rank_x = parseRank(Data[offset++]);
        std::vector<int64_t> shape_vec_x = parseShape(Data, offset, Size, rank_x);
        tensorflow::TensorShape shape_x(shape_vec_x);

        // 3. Parse Shape for Y
        // We allow rank_y to differ to explore broadcasting edge cases.
        uint8_t rank_y = rank_x; 
        if (offset < Size) {
            rank_y = parseRank(Data[offset++]);
        }
        std::vector<int64_t> shape_vec_y = parseShape(Data, offset, Size, rank_y);
        tensorflow::TensorShape shape_y(shape_vec_y);

        // 4. Construct Tensors
        // Note: Incorrect shapes for broadcasting will cause session.Run to return Status error, which is handled.
        tensorflow::Tensor x(dtype, shape_x);
        tensorflow::Tensor y(dtype, shape_y);

        // 5. Fill Tensors with fuzz data
        fillTensorWithDataByType(x, dtype, Data, offset, Size);
        fillTensorWithDataByType(y, dtype, Data, offset, Size);

        // 6. Build Graph
        tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
        
        auto x_op = tensorflow::ops::Const(scope, x);
        auto y_op = tensorflow::ops::Const(scope, y);

        // Op: Add
        auto add_op = tensorflow::ops::Add(scope, x_op, y_op);

        // 7. Run Session
        tensorflow::ClientSession session(scope);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({add_op}, &outputs);

        if (!status.ok()) {
            // Uncomment to debug specific failures
            // std::cout << "TF Run Status: " << status.ToString() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}