#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/tstring.h"

// Eigen headers for complex and half types
#include "unsupported/Eigen/CXX11/Tensor"

// Fuzzing constants to bound tensor sizes and prevent OOMs
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace tf_fuzz {

using namespace tensorflow;

// Utility functions to generate fuzzed inputs

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 21) {  // Reduced to avoid quantized types for simplicity
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
    default:
      dtype = DT_FLOAT;
      break;
  }
  return dtype;
}

uint8_t parseRank(uint8_t byte) {
  constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
  uint8_t rank = byte % range + MIN_RANK;
  return rank;
}

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
      uint8_t dim_val;
      std::memcpy(&dim_val, data + offset, sizeof_dim);
      offset += sizeof_dim;
      shape.push_back(dim_val % (MAX_TENSOR_SHAPE_DIMS_TF + 1));
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tstring>();
  const size_t num_elements = flat.size();

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset < total_size) {
      uint8_t str_len = data[offset] % 64;  // Keep strings reasonably short
      offset++;
      if (offset + str_len <= total_size) {
        flat(i).assign(reinterpret_cast<const char*>(data + offset), str_len);
        offset += str_len;
      } else {
        flat(i).assign("");
      }
    } else {
      flat(i).assign("");
    }
  }
}

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
      fillTensorWithData<int32>(tensor, data, offset, total_size);
      break;
    case DT_UINT8:
      fillTensorWithData<uint8>(tensor, data, offset, total_size);
      break;
    case DT_INT16:
      fillTensorWithData<int16>(tensor, data, offset, total_size);
      break;
    case DT_INT8:
      fillTensorWithData<int8>(tensor, data, offset, total_size);
      break;
    case DT_STRING:
      fillStringTensor(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX64:
      fillTensorWithData<complex64>(tensor, data, offset, total_size);
      break;
    case DT_INT64:
      fillTensorWithData<int64>(tensor, data, offset, total_size);
      break;
    case DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case DT_BFLOAT16:
      fillTensorWithData<bfloat16>(tensor, data, offset, total_size);
      break;
    case DT_UINT16:
      fillTensorWithData<uint16>(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX128:
      fillTensorWithData<complex128>(tensor, data, offset, total_size);
      break;
    case DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case DT_UINT32:
      fillTensorWithData<uint32>(tensor, data, offset, total_size);
      break;
    case DT_UINT64:
      fillTensorWithData<uint64>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

}  // namespace tf_fuzz

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // We need at least 3 bytes for types and rank selectors.
  if (size < 3) {
    return 0;
  }

  try {
    using namespace tensorflow;
    using namespace tf_fuzz;

    size_t offset = 0;

    // 1. Select data types and rank for the input tensor.
    DataType input_dtype = parseDataType(data[offset++]);
    DataType index_dtype = (data[offset++] % 2 == 0) ? DT_INT32 : DT_INT64;
    uint8_t input_rank = parseRank(data[offset++]);

    // 2. Construct the main input tensor.
    std::vector<int64_t> input_dims =
        parseShape(data, offset, size, input_rank);
    TensorShape input_shape;
    if (!TensorShape::BuildTensorShape(input_dims, &input_shape).ok()) {
      return 0; // Invalid shape, skip.
    }
    Tensor input_tensor(input_dtype, input_shape);
    fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);

    // 3. Construct the 'begin' and 'size' tensors. Their shape is a 1-D
    //    tensor with a single element for each dimension of the input.
    TensorShape index_shape;
    if (!TensorShape::BuildTensorShape({(int64_t)input_rank}, &index_shape).ok()) {
      return 0;
    }

    Tensor begin_tensor(index_dtype, index_shape);
    Tensor size_tensor(index_dtype, index_shape);

    if (index_dtype == DT_INT32) {
      fillTensorWithData<int32>(begin_tensor, data, offset, size);
      fillTensorWithData<int32>(size_tensor, data, offset, size);
    } else {  // DT_INT64
      fillTensorWithData<int64>(begin_tensor, data, offset, size);
      fillTensorWithData<int64>(size_tensor, data, offset, size);
    }

    // 4. Build the NodeDef for the Slice op.
    NodeDef node_def;
    NodeDefBuilder builder("slice_op", "Slice");
    builder.Input(FakeInput(input_dtype));
    builder.Input(FakeInput(index_dtype));
    builder.Input(FakeInput(index_dtype));
    builder.Attr("T", input_dtype);
    builder.Attr("Index", index_dtype);

    Status status = builder.Finalize(&node_def);
    if (!status.ok()) {
      // It's possible to generate invalid type combinations that the builder
      // rejects. We can just ignore these cases.
      return 0;
    }

    // 5. Run the op with FuzzSession.
    FuzzSession session;
    std::vector<Tensor> inputs = {input_tensor, begin_tensor, size_tensor};
    // FuzzSession::Run might return a non-OK status for invalid inputs,
    // which is an expected outcome. We just want to catch crashes.
    (void)session.Run(node_def, inputs);

  } catch (const std::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cout << "Unknown exception caught" << std::endl;
    return -1;
  }

  return 0;
}