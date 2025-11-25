#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

// Constants for fuzzing tensor properties
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace {

// Helper to consume a value of type T from the fuzzing data buffer.
template <typename T>
T Consume(const uint8_t* data, size_t size, size_t& offset) {
  if (offset + sizeof(T) > size) {
    return T{};
  }
  T value;
  std::memcpy(&value, data + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

// Helper functions provided in the user request, adapted for self-containment.

tensorflow::DataType parseDataType(const uint8_t* data, size_t& offset,
                                   size_t total_size) {
  if (offset >= total_size) {
    return tensorflow::DT_FLOAT;
  }
  uint8_t selector = Consume<uint8_t>(data, total_size, offset);
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

uint8_t parseRank(const uint8_t* data, size_t& offset, size_t total_size) {
  uint8_t byte = Consume<uint8_t>(data, total_size, offset);
  constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
  uint8_t rank = MIN_RANK + (byte % range);
  return rank;
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset,
                                size_t total_size, uint8_t rank) {
  if (rank == 0) {
    return {};
  }
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (uint8_t i = 0; i < rank; ++i) {
    uint8_t dim_val_byte = Consume<uint8_t>(data, total_size, offset);
    shape.push_back(MIN_TENSOR_SHAPE_DIMS_TF +
                    (dim_val_byte % (MAX_TENSOR_SHAPE_DIMS_TF -
                                     MIN_TENSOR_SHAPE_DIMS_TF + 1)));
  }
  return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
  auto flat = tensor.flat<T>();
  const size_t num_elements = flat.size();
  for (size_t i = 0; i < num_elements; ++i) {
    flat(i) = Consume<T>(data, total_size, offset);
  }
}

template <>
void fillTensorWithData<tensorflow::tstring>(tensorflow::Tensor& tensor,
                                             const uint8_t* data,
                                             size_t& offset,
                                             size_t total_size) {
  auto flat = tensor.flat<tensorflow::tstring>();
  for (size_t i = 0; i < flat.size(); ++i) {
    if (offset < total_size) {
      uint8_t len = std::min((size_t)Consume<uint8_t>(data, total_size, offset),
                             total_size - offset);
      if (offset + len <= total_size) {
        flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
        offset += len;
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
    case tensorflow::DT_FLOAT: fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case tensorflow::DT_DOUBLE: fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT32: fillTensorWithData<tensorflow::int32>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT8: fillTensorWithData<tensorflow::uint8>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT16: fillTensorWithData<tensorflow::int16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT8: fillTensorWithData<tensorflow::int8>(tensor, data, offset, total_size); break;
    case tensorflow::DT_STRING: fillTensorWithData<tensorflow::tstring>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX64: fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT64: fillTensorWithData<tensorflow::int64>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QINT8: fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QUINT8: fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QINT32: fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BFLOAT16: fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QINT16: fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QUINT16: fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT16: fillTensorWithData<tensorflow::uint16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX128: fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size); break;
    case tensorflow::DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT32: fillTensorWithData<tensorflow::uint32>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT64: fillTensorWithData<tensorflow::uint64>(tensor, data, offset, total_size); break;
    default: return;
  }
}

// Helper to format a vector as a string list "[a,b,c]"
template <typename T>
std::string VectorToString(const std::vector<T>& vec) {
  if (vec.empty()) {
    return "[]";
  }
  std::string s = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    s += std::to_string(vec[i]);
    if (i < vec.size() - 1) {
      s += ",";
    }
  }
  s += "]";
  return s;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  size_t offset = 0;

  // Create input tensor
  tensorflow::DataType input_dtype = parseDataType(data, offset, size);
  uint8_t input_rank = parseRank(data, offset, size);
  auto input_shape_vec = parseShape(data, offset, size, input_rank);
  tensorflow::TensorShape input_shape;
  if (!tensorflow::TensorShape::BuildTensorShape(input_shape_vec, &input_shape)
           .ok()) {
    return 0;  // Invalid shape
  }

  tensorflow::Tensor input_tensor(input_dtype, input_shape);
  if (input_shape.num_elements() < 500000) {  // Limit memory
    fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
  }

  // Create filter tensor
  tensorflow::DataType filter_dtype = parseDataType(data, offset, size);
  uint8_t filter_rank = parseRank(data, offset, size);
  auto filter_shape_vec = parseShape(data, offset, size, filter_rank);
  tensorflow::TensorShape filter_shape;
  if (!tensorflow::TensorShape::BuildTensorShape(filter_shape_vec,
                                                  &filter_shape)
           .ok()) {
    return 0;  // Invalid shape
  }
  tensorflow::Tensor filter_tensor(filter_dtype, filter_shape);
  if (filter_shape.num_elements() < 500000) {  // Limit memory
    fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
  }

  // Get attributes
  auto stride_h = Consume<uint8_t>(data, size, offset) % 4 + 1;
  auto stride_w = Consume<uint8_t>(data, size, offset) % 4 + 1;
  auto dilation_h = Consume<uint8_t>(data, size, offset) % 4 + 1;
  auto dilation_w = Consume<uint8_t>(data, size, offset) % 4 + 1;

  std::string padding_str = "VALID";
  uint8_t padding_choice = Consume<uint8_t>(data, size, offset);
  switch (padding_choice % 3) {
    case 0: padding_str = "SAME"; break;
    case 1: padding_str = "VALID"; break;
    case 2: padding_str = "EXPLICIT"; break;
  }

  std::string data_format_str =
      (Consume<uint8_t>(data, size, offset) % 2 == 0) ? "NHWC" : "NCHW";
  
  std::vector<int32_t> strides = (data_format_str == "NHWC")
                                     ? std::vector<int32_t>{1, (int32_t)stride_h, (int32_t)stride_w, 1}
                                     : std::vector<int32_t>{1, 1, (int32_t)stride_h, (int32_t)stride_w};
  std::vector<int32_t> dilations = (data_format_str == "NHWC")
                                       ? std::vector<int32_t>{1, (int32_t)dilation_h, (int32_t)dilation_w, 1}
                                       : std::vector<int32_t>{1, 1, (int32_t)dilation_h, (int32_t)dilation_w};

  std::vector<int32_t> explicit_paddings;
  if (padding_str == "EXPLICIT") {
    for (int i = 0; i < 8; ++i) {
      explicit_paddings.push_back(Consume<uint8_t>(data, size, offset) % 4);
    }
  }

  try {
    std::string op_spec = tensorflow::strings::StrCat(
        "T: ", tensorflow::DataTypeString(input_dtype),
        "; strides: ", VectorToString(strides),
        "; dilations: ", VectorToString(dilations),
        "; padding: '", padding_str, "'",
        "; data_format: '", data_format_str, "'");

    if (padding_str == "EXPLICIT") {
      tensorflow::strings::StrAppend(
          &op_spec, "; explicit_paddings: ", VectorToString(explicit_paddings));
    }

    std::unique_ptr<tensorflow::test::OpKernelTester> tester;
    // OpKernelTester constructor can CHECK-fail for invalid op specs
    tester = std::make_unique<tensorflow::test::OpKernelTester>("Conv2D",
                                                                op_spec);

    if (tester->OpKernelRegistered().ok()) {
      tester->AddInput(input_tensor);
      tester->AddInput(filter_tensor);
      tester->RunOpKernel();
    }
  } catch (const std::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}