#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "unsupported/Eigen/CXX11/Tensor"

#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

// Using statements for brevity
using tensorflow::DT_BFLOAT16;
using tensorflow::DT_BOOL;
using tensorflow::DT_COMPLEX128;
using tensorflow::DT_COMPLEX64;
using tensorflow::DT_DOUBLE;
using tensorflow::DT_FLOAT;
using tensorflow::DT_HALF;
using tensorflow::DT_INT16;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_INT8;
using tensorflow::DT_QINT16;
using tensorflow::DT_QINT32;
using tensorflow::DT_QINT8;
using tensorflow::DT_QUINT16;
using tensorflow::DT_QUINT8;
using tensorflow::DT_STRING;
using tensorflow::DT_UINT16;
using tensorflow::DT_UINT32;
using tensorflow::DT_UINT64;
using tensorflow::DT_UINT8;

// Helper function to select a random TensorFlow data type.
tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 21) { // Excluding quantized types as they are complex
    case 0:
      dtype = DT_FLOAT;
      break;
    case 1:
      dtype = DT_DOUBLE;
      break;
    case 2:
      dtype = DT_INT32;
      break;
    case 3:
      dtype = DT_UINT8;
      break;
    case 4:
      dtype = DT_INT16;
      break;
    case 5:
      dtype = DT_INT8;
      break;
    case 6:
      dtype = DT_STRING;
      break;
    case 7:
      dtype = DT_COMPLEX64;
      break;
    case 8:
      dtype = DT_INT64;
      break;
    case 9:
      dtype = DT_BOOL;
      break;
    case 10:
      dtype = DT_BFLOAT16;
      break;
    case 11:
      dtype = DT_UINT16;
      break;
    case 12:
      dtype = DT_COMPLEX128;
      break;
    case 13:
      dtype = DT_HALF;
      break;
    case 14:
      dtype = DT_UINT32;
      break;
    case 15:
      dtype = DT_UINT64;
      break;
    // Add other types if needed, for now keeping it to common ones.
    default:
      dtype = DT_FLOAT;
      break;
  }
  return dtype;
}

// Helper function to generate a random tensor rank.
uint8_t parseRank(uint8_t byte) {
  constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
  uint8_t rank = byte % range + MIN_RANK;
  return rank;
}

// Helper function to generate a random tensor shape.
std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset,
                                  size_t total_size, uint8_t rank) {
  if (rank == 0) {
    return {};
  }

  std::vector<int64_t> shape;
  shape.reserve(rank);
  const auto sizeof_dim = sizeof(uint8_t);

  for (uint8_t i = 0; i < rank; ++i) {
    if (offset + sizeof_dim <= total_size) {
      uint8_t dim_val_byte;
      std::memcpy(&dim_val_byte, data + offset, sizeof_dim);
      offset += sizeof_dim;

      int64_t dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                        (dim_val_byte % (MAX_TENSOR_SHAPE_DIMS_TF -
                                         MIN_TENSOR_SHAPE_DIMS_TF + 1));
      shape.push_back(dim_val);
    } else {
      shape.push_back(1);
    }
  }

  return shape;
}

// Generic template to fill tensor with data from the fuzzer input.
template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
  auto flat = tensor.flat<T>();
  const size_t num_elements = flat.size();
  if (num_elements == 0) return;
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

// Fills a tensor based on its DataType.
void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
    case DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    case DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size);
      break;
    case DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size);
      break;
    case DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
      break;
    case DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
      break;
    case DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
      break;
    default:
      // Unsupported types will be left zero-initialized.
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) {
    return 0;  // Not enough data for basic parameters.
  }

  size_t offset = 0;

  try {
    // Fuzz parameters
    const int num_tensors = (data[offset++] % 4) + 2;  // [2, 5] tensors
    tensorflow::DataType dtype = parseDataType(data[offset++]);
    uint8_t rank = parseRank(data[offset++]);

    // Fuzz concat_dim. Generate values around the valid range [0, rank-1]
    // to test edge cases like -1, rank, etc.
    int32_t concat_dim_val = 0;
    if (offset < size) {
      if (rank > 0) {
        // Generates values in [-1, rank]
        concat_dim_val = (data[offset++] % (rank + 2)) - 1;
      }
    }

    // Generate a base shape for the tensors.
    std::vector<int64_t> base_shape = parseShape(data, offset, size, rank);

    std::vector<tensorflow::Tensor> values_tensors;
    values_tensors.reserve(num_tensors);

    // Create and populate the tensors to be concatenated.
    for (int i = 0; i < num_tensors; ++i) {
      std::vector<int64_t> current_shape = base_shape;

      // If concat_dim is valid, fuzz the size of that dimension.
      // Otherwise, all shapes remain identical.
      if (rank > 0 && concat_dim_val >= 0 && concat_dim_val < rank) {
        uint8_t dim_size_byte = 0;
        if (offset < size) {
          dim_size_byte = data[offset++];
        }
        // Generate a new size for the concatenation axis.
        current_shape[concat_dim_val] = MIN_TENSOR_SHAPE_DIMS_TF +
                                        (dim_size_byte % (MAX_TENSOR_SHAPE_DIMS_TF -
                                                         MIN_TENSOR_SHAPE_DIMS_TF + 1));
      }

      tensorflow::TensorShape tensor_shape;
      if (!tensorflow::TensorShape::BuildTensorShape(current_shape, &tensor_shape).ok()) {
        continue; // Skip if shape is invalid.
      }
      
      tensorflow::Tensor tensor(dtype, tensor_shape);

      // Fill tensor with data.
      if (dtype == DT_STRING) {
        auto string_tensor = tensor.flat<tensorflow::tstring>();
        for (int j = 0; j < string_tensor.size(); ++j) {
          uint8_t str_len = 0;
          if (offset < size) {
            str_len = data[offset++] % 16; // Keep strings short.
          }
          if (offset + str_len <= size) {
            string_tensor(j).assign(reinterpret_cast<const char*>(data + offset), str_len);
            offset += str_len;
          }
        }
      } else {
        fillTensorWithDataByType(tensor, dtype, data, offset, size);
      }
      values_tensors.push_back(tensor);
    }
    
    // The op requires at least 2 tensors.
    if (values_tensors.size() < 2) {
      return 0;
    }

    // Convert vector<Tensor> to vector<Input> for the op.
    std::vector<tensorflow::Input> values_inputs;
    for (const auto& t : values_tensors) {
        values_inputs.push_back(tensorflow::Input(t));
    }

    // Setup TensorFlow graph and session.
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    
    // Create the scalar tensor for `concat_dim`.
    tensorflow::Tensor concat_dim_tensor(DT_INT32, {});
    concat_dim_tensor.scalar<int32_t>()() = concat_dim_val;

    // Define the Concat operation.
    auto concat_op = tensorflow::ops::Concat(root.WithOpName("fuzz_concat"),
                                             values_inputs, concat_dim_tensor);

    // Execute the graph.
    tensorflow::ClientSession session(root);
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session.Run({concat_op}, &outputs);
    // We don't check the status. A non-OK status is an expected outcome for
    // invalid inputs. The fuzzer's goal is to find inputs that cause crashes
    // (segfaults, etc.), not ones that return errors.

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }
  return 0;
}