#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "unsupported/Eigen/CXX11/Tensor"

#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace {

// Helper to convert non-OK statuses to exceptions to be caught by the harness
void ThrowIfError(const tensorflow::Status& s, const char* context) {
  if (!s.ok()) {
    throw std::runtime_error(std::string(context) + ": " + s.ToString());
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 23) {
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
      dtype = tensorflow::DT_STRING;
      break;
    case 7:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 8:
      dtype = tensorflow::DT_INT64;
      break;
    case 9:
      dtype = tensorflow::DT_BOOL;
      break;
    case 10:
      dtype = tensorflow::DT_QINT8;
      break;
    case 11:
      dtype = tensorflow::DT_QUINT8;
      break;
    case 12:
      dtype = tensorflow::DT_QINT32;
      break;
    case 13:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 14:
      dtype = tensorflow::DT_QINT16;
      break;
    case 15:
      dtype = tensorflow::DT_QUINT16;
      break;
    case 16:
      dtype = tensorflow::DT_UINT16;
      break;
    case 17:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 18:
      dtype = tensorflow::DT_HALF;
      break;
    case 19:
      dtype = tensorflow::DT_UINT32;
      break;
    case 20:
      dtype = tensorflow::DT_UINT64;
      break;
    default:
      dtype = tensorflow::DT_FLOAT;
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
  const auto sizeof_dim = sizeof(int64_t);

  for (uint8_t i = 0; i < rank; ++i) {
    if (offset + sizeof_dim <= total_size) {
      int64_t dim_val;
      std::memcpy(&dim_val, data + offset, sizeof_dim);
      offset += sizeof_dim;

      dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                static_cast<int64_t>(
                    (static_cast<uint64_t>(std::abs(dim_val)) %
                     static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF -
                                           MIN_TENSOR_SHAPE_DIMS_TF + 1)));

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
      return;
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // We need at least 2 bytes for dtype and rank.
  if (size < 2) {
    return 0;
  }

  try {
    size_t offset = 0;

    // 1. Determine DataType
    tensorflow::DataType dtype = parseDataType(data[offset++]);

    // 2. Determine Rank
    uint8_t rank = parseRank(data[offset++]);

    // 3. Determine Shape
    std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
    tensorflow::TensorShape shape;
    ThrowIfError(tensorflow::TensorShape::BuildTensorShape(shape_dims, &shape),
                 "BuildTensorShape");

    // 4. Create and fill input tensor 'x'
    tensorflow::Tensor input_tensor(dtype, shape);
    if (input_tensor.NumElements() > 0) {
      fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
    }

    // 5. Build NodeDef for the Sigmoid op
    tensorflow::NodeDef def;
    tensorflow::NodeDefBuilder builder("sigmoid_fuzz_op", "Sigmoid");
    builder.Attr("T", dtype);
    builder.Input(tensorflow::FakeInput(dtype));
    ThrowIfError(builder.Finalize(&def), "Finalize NodeDefBuilder");

    // 6. Create the OpKernelTester for the op on the CPU
    tensorflow::OpKernelTester tester(def, "CPU");

    // 7. Set the input tensor for the test
    tester.SetInput(0, input_tensor);

    // 8. Run the op kernel
    ThrowIfError(tester.RunOpKernel(), "RunOpKernel");

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
  }
  return 0;
}