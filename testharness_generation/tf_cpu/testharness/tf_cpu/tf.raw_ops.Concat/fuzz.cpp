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
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"

#define MIN_RANK 0
#define MAX_RANK 5
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

void fillTensorWithDataString(tensorflow::Tensor& tensor, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tensorflow::tstring>();
  const size_t num_elements = flat.size();
  
  for (size_t i = 0; i < num_elements; ++i) {
    if (offset < total_size) {
      size_t len = data[offset] % 32; 
      offset++;
      if (offset + len > total_size) {
          len = total_size - offset;
      }
      flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
      offset += len;
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
      fillTensorWithDataString(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    if (Size < 4) return 0;

    try {
        size_t offset = 0;
        
        // 1. Number of tensors to concat (2 to 5)
        uint8_t num_tensors = (Data[offset++] % 4) + 2;

        // 2. Data Type
        tensorflow::DataType dtype = parseDataType(Data[offset++]);

        // 3. Rank
        uint8_t rank = parseRank(Data[offset++]);
        
        // 4. Concat Dimension
        int32_t concat_dim_val = 0;
        if (rank > 0) {
            if (offset < Size) {
                 concat_dim_val = Data[offset++] % rank; 
            }
        } 

        // 5. Parse a Base Shape (TensorFlow valid dimensions)
        std::vector<int64_t> base_shape = parseShape(Data, offset, Size, rank);

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        std::vector<tensorflow::Input> values_ops;
        values_ops.reserve(num_tensors);

        for (uint8_t i = 0; i < num_tensors; ++i) {
            std::vector<int64_t> current_shape = base_shape;
            
            // To create valid Concat scenarios, tensors must match except in concat_dim.
            // We use the base_shape and modify the concat_dim dimension.
            if (rank > 0 && offset < Size) {
                // Change size of the concat dimension based on input data
                uint8_t variant = Data[offset++];
                int64_t dim_mod = (variant % 8) + 1; 
                current_shape[concat_dim_val] = dim_mod;
            }

            tensorflow::Tensor t(dtype, tensorflow::TensorShape(current_shape));
            fillTensorWithDataByType(t, dtype, Data, offset, Size);
            values_ops.push_back(tensorflow::ops::Const(root, t));
        }

        // Create concat_dim tensor (scalar int32)
        tensorflow::Tensor dim_t(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        dim_t.scalar<int32_t>()() = concat_dim_val;
        auto dim_op = tensorflow::ops::Const(root, dim_t);

        // Build Concat Op
        // tf.raw_ops.Concat takes (concat_dim, values)
        auto op = tensorflow::ops::Concat(root, dim_op, values_ops);

        // Run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // We catch exception outside, so we can ignore return status.
        session.Run({op.output}, &outputs);

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}