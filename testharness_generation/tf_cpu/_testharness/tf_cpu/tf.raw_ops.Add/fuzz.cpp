#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"

// Define tensor constraints for fuzzing
#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 6

using namespace tensorflow;

// Helper: Parse Data Type (Restricted to Add's supported types)
// Add supports: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, 
// complex64, complex128, string.
tensorflow::DataType parseDataType(uint8_t selector) {
  static const tensorflow::DataType kSupportedTypes[] = {
      DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE,
      DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
      DT_COMPLEX64, DT_COMPLEX128, DT_STRING
  };
  return kSupportedTypes[selector % (sizeof(kSupportedTypes) / sizeof(tensorflow::DataType))];
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

// Helper: Fill Tensor with numeric data
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

// Helper: Fill Tensor with strings
void fillTensorWithString(tensorflow::Tensor& tensor, const uint8_t* data,
                          size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tstring>();
  const size_t num_elements = flat.size();

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset < total_size) {
        size_t len = data[offset] % 32; // Limit string length
        offset++;
        if (offset + len <= total_size) {
             flat(i) = string(reinterpret_cast<const char*>(data + offset), len);
             offset += len;
        } else {
             flat(i) = string(reinterpret_cast<const char*>(data + offset), total_size - offset);
             offset = total_size;
        }
    } else {
      flat(i) = "";
    }
  }
}

// Helper: Dispatch based on DataType
void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  if (dtype == DT_STRING) {
      fillTensorWithString(tensor, data, offset, total_size);
      return;
  }
  
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Basic validity check to ensure we can read at least dtype and ranks
    if (Size < 3) return 0;

    size_t offset = 0;

    try {
        // 1. Determine common DataType for both inputs
        DataType dtype = parseDataType(Data[offset++]);

        // 2. Construct Tensor X
        uint8_t rank_x = parseRank(Data[offset++]);
        std::vector<int64_t> shape_x = parseShape(Data, offset, Size, rank_x);
        Tensor tensor_x(dtype, TensorShape(shape_x));
        fillTensorWithDataByType(tensor_x, dtype, Data, offset, Size);

        // 3. Construct Tensor Y
        if (offset >= Size) return 0; // Not enough data for rank_y
        uint8_t rank_y = parseRank(Data[offset++]);
        std::vector<int64_t> shape_y = parseShape(Data, offset, Size, rank_y);
        Tensor tensor_y(dtype, TensorShape(shape_y));
        fillTensorWithDataByType(tensor_y, dtype, Data, offset, Size);

        // 4. Build TF Graph
        Scope root = Scope::NewRootScope();
        
        auto x_op = ops::Const(root, tensor_x);
        auto y_op = ops::Const(root, tensor_y);
        
        // Target Op: tf.raw_ops.Add
        auto add_op = ops::Add(root, x_op, y_op);

        // 5. Run Session
        ClientSession session(root);
        std::vector<Tensor> outputs;
        
        // This triggers the Op Kernel execution
        // We catch validation errors (e.g., incompatible broadcast shapes) via Status
        Status status = session.Run({add_op}, &outputs);

        if (!status.ok()) {
            // Uncomment for debugging:
            // std::cout << "TF Error: " << status.ToString() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
    }
    
    return 0;
}