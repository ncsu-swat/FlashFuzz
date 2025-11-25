#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"

// Define constants for fuzzing tensor properties
#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace {

// Helper to select a TensorFlow DataType from a byte.
// Covers a wide range of types to test op validation.
tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 21) {
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_INT32;
      break;
    case 3:
      dtype = tensorflow::DT_UINT8;
      break;
    case 4:
      dtype = tensorflow::DT_INT16;
      break;
    case 5:
      dtype = tensorflow::DT_INT8;
      break;
    case 6:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 7:
      dtype = tensorflow::DT_INT64;
      break;
    case 8:
      dtype = tensorflow::DT_BOOL;
      break;
    case 9:
      dtype = tensorflow::DT_QINT8;
      break;
    case 10:
      dtype = tensorflow::DT_QUINT8;
      break;
    case 11:
      dtype = tensorflow::DT_QINT32;
      break;
    case 12:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 13:
      dtype = tensorflow::DT_QINT16;
      break;
    case 14:
      dtype = tensorflow::DT_QUINT16;
      break;
    case 15:
      dtype = tensorflow::DT_UINT16;
      break;
    case 16:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 17:
      dtype = tensorflow::DT_HALF;
      break;
    case 18:
      dtype = tensorflow::DT_UINT32;
      break;
    case 19:
      dtype = tensorflow::DT_UINT64;
      break;
    case 20:
      // Unsupported type to test validation
      dtype = tensorflow::DT_STRING;
      break;
    default:
      dtype = tensorflow::DT_FLOAT;
      break;
  }
  return dtype;
}

// Helper to select a tensor rank from a byte.
uint8_t parseRank(uint8_t byte) {
  constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
  uint8_t rank = byte % range + MIN_RANK;
  return rank;
}

// Helper to generate a tensor shape from fuzzer data.
// Consumes data from the input buffer and updates the offset.
std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset,
                                size_t total_size, uint8_t rank) {
  if (rank == 0) {
    return {};
  }

  std::vector<int64_t> shape;
  shape.reserve(rank);
  const auto sizeof_dim = sizeof(uint8_t);

  for (uint8_t i = 0; i < rank; ++i) {
    if (offset < total_size) {
      uint8_t dim_byte = data[offset++];
      int64_t dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                        (dim_byte % (MAX_TENSOR_SHAPE_DIMS_TF -
                                     MIN_TENSOR_SHAPE_DIMS_TF + 1));
      shape.push_back(dim_val);
    } else {
      shape.push_back(1);
    }
  }

  return shape;
}

// Generic template to fill a tensor with data from the fuzzer buffer.
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
      // Fill with default value if we run out of fuzz data
      flat(i) = T{};
    }
  }
}

// Dispatches to the correct typed fill function based on DataType.
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
      // Do nothing for unsupported types like string, quantized, etc.
      // The op kernel will handle the type error.
      break;
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // We need at least a few bytes for attributes and ranks.
  if (size < 5) {
    return 0;
  }

  try {
    size_t offset = 0;

    // Consume booleans for transpose attributes
    bool transpose_a = data[offset++] % 2;
    bool transpose_b = data[offset++] % 2;

    // Consume data type, ensuring it's the same for both tensors
    tensorflow::DataType dtype = parseDataType(data[offset++]);

    // Construct tensor 'a'
    uint8_t rank_a = parseRank(data[offset++]);
    std::vector<int64_t> shape_vec_a = parseShape(data, offset, size, rank_a);
    tensorflow::TensorShape shape_a(shape_vec_a);
    tensorflow::Tensor tensor_a(dtype, shape_a);
    fillTensorWithDataByType(tensor_a, dtype, data, offset, size);

    // Construct tensor 'b'
    // Ensure we still have data for rank_b before proceeding
    if (offset >= size) return 0;
    uint8_t rank_b = parseRank(data[offset++]);
    std::vector<int64_t> shape_vec_b = parseShape(data, offset, size, rank_b);
    tensorflow::TensorShape shape_b(shape_vec_b);
    tensorflow::Tensor tensor_b(dtype, shape_b);
    fillTensorWithDataByType(tensor_b, dtype, data, offset, size);

    // Build and run the graph
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    auto op_a = tensorflow::ops::Const(root.WithOpName("a"), tensor_a);
    auto op_b = tensorflow::ops::Const(root.WithOpName("b"), tensor_b);

    auto matmul_op = tensorflow::ops::MatMul(
        root.WithOpName("matmul"), op_a, op_b,
        tensorflow::ops::MatMul::TransposeA(transpose_a).TransposeB(
            transpose_b));

    // If scope construction fails (e.g., shape inference), exit gracefully.
    if (!root.ok()) {
      return 0;
    }

    tensorflow::GraphDef graph;
    tensorflow::Status status = root.ToGraphDef(&graph);
    if (!status.ok()) {
      return 0;
    }

    // Create a session and run the graph.
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    status = session->Create(graph);
    if (!status.ok()) {
      return 0;
    }

    std::vector<tensorflow::Tensor> outputs;
    // Run the graph. A non-OK status is an expected outcome for invalid inputs.
    // The fuzzer's goal is to find inputs that cause crashes (unhandled
    // signals) or hangs, not just errors that TF handles gracefully.
    session->Run({}, {}, {"matmul"}, &outputs);

  } catch (const tensorflow::errors::InvalidArgument& e) {
    // Catch expected exceptions for invalid arguments.
    std::cout << "Caught expected exception: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    // Catch any other unexpected exceptions.
    std::cout << "Caught unexpected exception: " << e.what() << std::endl;
  }

  return 0;
}