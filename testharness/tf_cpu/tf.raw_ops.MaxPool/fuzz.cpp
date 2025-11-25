#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"

// Define helper constants
#define MAX_POOL_RANK 4

// Helper to fill tensor with raw data
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

// Dispatcher to fill tensor based on DataType
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
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
      break;
    default:
      // Unsupported types for this fuzzer or fallback
      break;
  }
}

// Select a valid DataType for MaxPool
tensorflow::DataType getMaxPoolDataType(uint8_t selector) {
    // MaxPool supports: half, bfloat16, float32, float64, int32, int64, 
    // uint8, int16, int8, uint16, qint8
    switch(selector % 11) {
        case 0: return tensorflow::DT_FLOAT;
        case 1: return tensorflow::DT_DOUBLE;
        case 2: return tensorflow::DT_INT32;
        case 3: return tensorflow::DT_INT64;
        case 4: return tensorflow::DT_UINT8;
        case 5: return tensorflow::DT_INT16;
        case 6: return tensorflow::DT_INT8;
        case 7: return tensorflow::DT_UINT16;
        case 8: return tensorflow::DT_HALF;
        case 9: return tensorflow::DT_BFLOAT16;
        case 10: return tensorflow::DT_QINT8;
        default: return tensorflow::DT_FLOAT;
    }
}

// Parse input tensor shape (Fixed Rank 4 for MaxPool)
std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int64_t> shape;
    shape.reserve(MAX_POOL_RANK);
    const auto sizeof_dim = sizeof(int32_t);

    for (uint8_t i = 0; i < MAX_POOL_RANK; ++i) {
        if (offset + sizeof_dim <= total_size) {
            int32_t dim_val;
            std::memcpy(&dim_val, data + offset, sizeof_dim);
            offset += sizeof_dim;
            
            // Constrain dims to avoid OOM, but allow enough var (1..32)
            int64_t safe_dim = 1 + (std::abs(dim_val) % 32); 
            shape.push_back(safe_dim);
        } else {
             shape.push_back(1);
        }
    }
    return shape;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Require minimum size for basic config
    if (Size < 16) return 0;
    
    size_t offset = 0;

    // --- 1. Parse Attributes ---
    
    // Data Format: NHWC (0) or NCHW (1)
    uint8_t fmt_byte = Data[offset++] % 2;
    std::string data_format = (fmt_byte == 0) ? "NHWC" : "NCHW";

    // Padding: SAME, VALID, EXPLICIT
    uint8_t pad_byte = Data[offset++] % 3;
    std::string padding_type;
    std::vector<int> explicit_paddings;

    if (pad_byte == 0) padding_type = "SAME";
    else if (pad_byte == 1) padding_type = "VALID";
    else {
        padding_type = "EXPLICIT";
        // Explicit padding needs 8 ints (2 per dimension for 4D)
        for(int i=0; i<8; ++i) {
             if (offset + sizeof(int32_t) <= Size) {
                int32_t val;
                std::memcpy(&val, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                explicit_paddings.push_back(std::abs(val) % 16); 
             } else {
                explicit_paddings.push_back(0);
             }
        }
    }

    // KSize and Strides (4 ints each)
    std::vector<int> ksize;
    std::vector<int> strides;
    
    auto consume_ints = [&](std::vector<int>& target) {
        for(int i=0; i<4; ++i) {
             if (offset + sizeof(int32_t) <= Size) {
                int32_t val;
                std::memcpy(&val, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                int v = 1 + (std::abs(val) % 8); 
                target.push_back(v);
             } else {
                target.push_back(1);
             }
        }
    };

    consume_ints(ksize);
    consume_ints(strides);

    // --- 2. Parse Input Tensor ---
    if (offset >= Size) return 0;
    
    tensorflow::DataType dtype = getMaxPoolDataType(Data[offset++]);
    std::vector<int64_t> shape_vec = parseShape(Data, offset, Size);
    tensorflow::TensorShape shape(shape_vec);

    // --- 3. Construct and Run Graph ---
    try {
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        tensorflow::Tensor input_tensor(dtype, shape);
        fillTensorWithDataByType(input_tensor, dtype, Data, offset, Size);
        
        auto input_node = tensorflow::ops::Const(root, input_tensor);
        
    auto max_pool = tensorflow::ops::MaxPool(root, input_node, ksize, strides, padding_type,
                                             tensorflow::ops::MaxPool::DataFormat(data_format)
                                                 .ExplicitPaddings(explicit_paddings));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Run the op. Invalid inputs are handled by Status return.
        tensorflow::Status status = session.Run({max_pool}, &outputs);
        
        // Optionally catch non-OK status but generally we just want to ensure no crash.
        // if (!status.ok()) { /* logic */ }

    } catch (const std::exception &e) {
        // Catch C++ exceptions to keep fuzzer alive
    } catch (...) {
        // Catch all
    }

    return 0;
}
