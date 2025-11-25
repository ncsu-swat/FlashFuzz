#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using namespace tensorflow;

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
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
      // For types like DT_STRING, we leave the tensor empty/default-initialized
      // to avoid complex memory management in fuzzer or invalid memcpy.
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Basic size checks
    if (Size < 2) return 0;
    if (Size > 50 * 1024 * 1024) return 0;

    try {
        size_t offset = 0;

        // Parse DataType
        uint8_t dtype_selector = Data[offset++];
        DataType dtype = parseDataType(dtype_selector);

        // Parse Rank
        uint8_t rank_selector = 0;
        if (offset < Size) rank_selector = Data[offset++];
        uint8_t rank = parseRank(rank_selector);

        // Parse Shape
        std::vector<int64_t> shape_vec = parseShape(Data, offset, Size, rank);
        TensorShape shape(shape_vec);
        
        // Prevent OOM with reasonable element count limit
        if (shape.num_elements() > 2000000) return 0;

        // Create Tensor
        Tensor input_tensor(dtype, shape);
        
        // Fill Tensor
        fillTensorWithDataByType(input_tensor, dtype, Data, offset, Size);

        // Construct TF Graph
        Scope root = Scope::NewRootScope();
        
        // Input Op
        auto input = ops::Const(root, input_tensor);
        
        // Sigmoid Op
        auto sigmoid = ops::Sigmoid(root, input);

        // Execute Graph
        ClientSession session(root);
        std::vector<Tensor> outputs;
        Status status = session.Run({sigmoid}, &outputs);

        // Ignore failure status, we are fuzzing for crashes
        if (!status.ok()) {
            // std::cout << "Status: " << status.ToString() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        // Catch-all
    }

    return 0;
}