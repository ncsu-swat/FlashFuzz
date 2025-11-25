#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/platform/tstring.h"

using namespace tensorflow;

// Fuzzing configuration
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 16

// Helper to select a valid data type for Softmax.
// Softmax supports: half, bfloat16, float32, float64.
DataType parseAllowedDataType(uint8_t selector) {
  switch (selector % 4) {
    case 0:
      return DT_FLOAT;
    case 1:
      return DT_DOUBLE;
    case 2:
      return DT_HALF;
    case 3:
      return DT_BFLOAT16;
    default:
      return DT_FLOAT;
  }
}

// Parse rank from input byte
uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

// Parse shape dimensions from input data
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
            
            // Map the random value to a constrained valid dimension size
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

// Template helper to fill tensor with data of type T
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
      flat(i) = T(0);
    }
  }
}

// Dispatcher to fill tensor based on DataType
void fillTensorWithDataByType(Tensor& tensor,
                              DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
    case DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case DT_BFLOAT16:
      fillTensorWithData<bfloat16>(tensor, data, offset, total_size);
      break;
    case DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      // Should not be reached given parseAllowedDataType
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Ensure minimal size for parsing
    if (Size < 1) return 0;

    try
    {
        size_t offset = 0;

        // 1. Select Data Type
        DataType dtype = parseAllowedDataType(Data[offset]);
        offset++;

        // 2. Select Rank
        uint8_t rank = 0;
        if (offset < Size) {
            rank = parseRank(Data[offset]);
            offset++;
        }

        // 3. Parse Shape
        std::vector<int64_t> shape_dims = parseShape(Data, offset, Size, rank);
        
        TensorShape shape;
        if (TensorShape::BuildTensorShape(shape_dims, &shape) != absl::OkStatus()) {
            return 0;
        }

        // 4. Create Tensor
        Tensor logits_tensor(dtype, shape);
        
        // Prevent excessive memory allocation
        if (logits_tensor.NumElements() > 500000) {
            return 0;
        }

        // 5. Fill Tensor with fuzz data
        fillTensorWithDataByType(logits_tensor, dtype, Data, offset, Size);

        // 6. Build Graph
        Scope root = Scope::NewRootScope();
        
        // Create Softmax op: tf.raw_ops.Softmax(logits=logits_tensor)
        auto op = ops::Softmax(root, logits_tensor);

        // 7. Run Session
        ClientSession session(root);
        std::vector<Tensor> outputs;
        
        // We don't check status strictly as invalid inputs are expected to fail gracefully
        Status status = session.Run({op}, &outputs);
        
        // Optional: print failure for local debugging
        // if (!status.ok()) {
        //     std::cout << "Softmax failed: " << status.ToString() << std::endl;
        // }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught." << std::endl;
    }

    return 0;
}