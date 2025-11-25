#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

// Define tensor shape limits for fuzzing
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tensorflow {

// Helper to parse data type from a byte
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

// Helper to parse rank
uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

// Helper to parse dimensions for a shape
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

// Generic tensor filler
template <typename T>
void fillTensorWithData(Tensor& tensor, const uint8_t* data,
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

// Specialization for string
template <>
void fillTensorWithData<tstring>(Tensor& tensor, const uint8_t* data,
                                 size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tstring>();
  const size_t num_elements = flat.size();

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset < total_size) {
      // Pick a small length for the string (0-16 bytes)
      size_t len = data[offset] % 17; 
      offset++;
      if (offset + len <= total_size) {
          flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
          offset += len;
      } else {
          flat(i) = "";
      }
    } else {
      flat(i) = "";
    }
  }
}

// Dispatcher for tensor filling
void fillTensorWithDataByType(Tensor& tensor,
                              DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case DT_FLOAT: fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case DT_DOUBLE: fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case DT_INT32: fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    case DT_UINT8: fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case DT_INT16: fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case DT_INT8: fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case DT_INT64: fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
    case DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case DT_UINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case DT_UINT32: fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
    case DT_UINT64: fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
    case DT_BFLOAT16: fillTensorWithData<bfloat16>(tensor, data, offset, total_size); break;
    case DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case DT_COMPLEX64: fillTensorWithData<complex64>(tensor, data, offset, total_size); break;
    case DT_COMPLEX128: fillTensorWithData<complex128>(tensor, data, offset, total_size); break;
    case DT_STRING: fillTensorWithData<tstring>(tensor, data, offset, total_size); break;
    // For quantized types, we can treat them as their storage types or defaults. 
    // Simplified here to map to int8/int16/int32 or ignore (leaving uninitialized/zeroed).
    case DT_QINT8: fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case DT_QUINT8: fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case DT_QINT16: fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case DT_QUINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case DT_QINT32: fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    default: break; 
  }
}

} // namespace tensorflow

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Need at least a few bytes for configuration
    if (Size < 3) return 0;

    using namespace tensorflow;

    size_t offset = 0;

    try {
        // 1. Parse Input Tensor Config
        uint8_t dtype_byte = Data[offset++];
        DataType input_dtype = parseDataType(dtype_byte);
        
        uint8_t rank_byte = Data[offset++];
        uint8_t input_rank = parseRank(rank_byte);

        // 2. Parse Input Tensor Shape
        std::vector<int64_t> input_shape_vec = parseShape(Data, offset, Size, input_rank);
        TensorShape input_shape(input_shape_vec);
        
        // 3. Create and Fill Input Tensor
        Tensor input_tensor(input_dtype, input_shape);
        // Ensure we don't proceed if tensor size is unreasonably huge to prevent OOM in fuzzing
        if (input_tensor.NumElements() > 1000000) {
            return 0;
        }
        fillTensorWithDataByType(input_tensor, input_dtype, Data, offset, Size);

        // 4. Parse Target Shape Argument
        // The second argument to Reshape is a 1-D tensor containing the new shape.
        if (offset >= Size) return 0;
        uint8_t target_rank_byte = Data[offset++];
        uint8_t target_rank = parseRank(target_rank_byte); // This is the size of the 1D shape tensor

        // We use INT32 for the shape tensor as it's common, though INT64 is also valid.
        Tensor target_shape_tensor(DT_INT32, TensorShape({target_rank}));
        auto target_flat = target_shape_tensor.flat<int32>();

        for (int i = 0; i < target_rank; ++i) {
            if (offset + sizeof(int32_t) <= Size) {
                int32_t val;
                std::memcpy(&val, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                
                // Constrain values to be interesting for reshape:
                // -1 is special. Small positive numbers are useful.
                // Huge numbers usually result in immediate size mismatch error.
                // Map random int to range [-1, 20] slightly biased.
                int32_t mod_val = val % 22; 
                target_flat(i) = mod_val - 1; // Range: -1 to 20
            } else {
                target_flat(i) = 1;
            }
        }

        // Debug output
        // std::cout << "Input Dtype: " << DataTypeString(input_dtype) 
        //           << ", Input Shape: " << input_shape.DebugString() 
        //           << ", Target Shape Tensor: " << target_shape_tensor.SummarizeValue(10) << std::endl;

        // 5. Construct Graph and Run Op
        Scope root = Scope::NewRootScope();
        
        auto input_op = ops::Const(root, input_tensor);
        auto shape_op = ops::Const(root, target_shape_tensor);
        
        auto reshape_op = ops::Reshape(root, input_op, shape_op);

        ClientSession session(root);
        std::vector<Tensor> outputs;
        
        // Run the graph. This will trigger the Shape inference and Op execution.
        Status status = session.Run({reshape_op}, &outputs);

        // We intentionally ignore the status. The fuzzer's goal is to find crashes (segfaults, aborts),
        // not logic errors returned via Status. Valid API usage errors are expected.
        
    } catch (const std::exception &e) {
        // Catch C++ exceptions to keep the fuzzer running, printing for visibility.
        // std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}