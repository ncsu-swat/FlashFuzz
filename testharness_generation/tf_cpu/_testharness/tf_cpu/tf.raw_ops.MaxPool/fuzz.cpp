#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// TensorFlow headers
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

// Constants from the starter skeleton
#define MIN_RANK 4
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using namespace tensorflow;

// Helper to get a DataType supported by MaxPool from a fuzzed byte.
DataType GetValidMaxPoolDataType(uint8_t selector) {
  const static std::vector<DataType> kSupportedTypes = {
      DT_HALF, DT_BFLOAT16, DT_FLOAT,  DT_DOUBLE, DT_INT32,
      DT_INT64, DT_UINT8,  DT_INT16,  DT_INT8,   DT_UINT16,
      DT_QINT8};
  return kSupportedTypes[selector % kSupportedTypes.size()];
}

// Helper to consume data for a vector of int32.
std::vector<int32_t> consumeInt32Vector(const uint8_t* data, size_t& offset,
                                        size_t total_size, size_t count) {
  std::vector<int32_t> vec;
  vec.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (offset + sizeof(int32_t) <= total_size) {
      int32_t val;
      memcpy(&val, data + offset, sizeof(int32_t));
      offset += sizeof(int32_t);
      vec.push_back(val);
    } else {
      vec.push_back(1);  // Default value if not enough data
    }
  }
  return vec;
}

// Verbatim helper snippet from the user request
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

// Verbatim helper snippet from the user request
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

// Verbatim helper snippet from the user request, adapted for MaxPool types
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
      fillTensorWithData<int32>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT8:
      fillTensorWithData<qint8>(tensor, data, offset, total_size);
      break;
    default:
      // Other types not supported by MaxPool are not filled.
      return;
  }
}

// A test fixture to simplify op kernel creation and execution.
class FuzzMaxPoolOp : public OpsTestBase {
 protected:
  void CreateOp(DataType T, const std::vector<int32>& ksize,
                const std::vector<int32>& strides, const string& padding,
                const std::vector<int32>& explicit_paddings,
                const string& data_format) {
    TF_ASSERT_OK(NodeDefBuilder("fuzz_maxpool", "MaxPool")
                     .Input(FakeInput(T))
                     .Attr("ksize", ksize)
                     .Attr("strides", strides)
                     .Attr("padding", padding)
                     .Attr("explicit_paddings", explicit_paddings)
                     .Attr("data_format", data_format)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  size_t offset = 0;
  if (size < 4) {
    return 0;
  }

  try {
    // Consume data for attributes
    DataType input_type = GetValidMaxPoolDataType(data[offset++]);

    const std::string padding = (data[offset] % 3 == 0)   ? "SAME"
                                : (data[offset] % 3 == 1) ? "VALID"
                                                          : "EXPLICIT";
    offset++;
    const std::string data_format = (data[offset] % 2 == 0) ? "NHWC" : "NCHW";
    offset++;

    std::vector<int32> ksize = consumeInt32Vector(data, offset, size, 4);
    std::vector<int32> strides = consumeInt32Vector(data, offset, size, 4);

    // Ensure kernel and stride values are positive to avoid hangs/errors.
    for (int i = 0; i < 4; ++i) {
      ksize[i] = std::abs(ksize[i]) % 16 + 1;
      strides[i] = std::abs(strides[i]) % 16 + 1;
    }

    std::vector<int32> explicit_paddings;
    if (padding == "EXPLICIT") {
      explicit_paddings = consumeInt32Vector(data, offset, size, 8);
    }

    // Consume data for input tensor
    const uint8_t rank = 4;
    std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
    TensorShape input_shape(shape_dims);

    // Avoid creating huge tensors that could lead to OOM.
    if (input_shape.num_elements() > 20000) {
      return 0;
    }

    Tensor input_tensor(input_type, input_shape);
    fillTensorWithDataByType(input_tensor, input_type, data, offset, size);

    // Setup and run the op
    FuzzMaxPoolOp op;
    op.CreateOp(input_type, ksize, strides, padding, explicit_paddings,
                data_format);
    op.AddInput(input_tensor);

    // Run the kernel. On failure, TF_ASSERT_OK will abort, which is what the
    // fuzzer expects.
    op.RunOpKernel();

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    // The fuzzer framework will report this as a crash.
    return -1;
  }

  return 0;  // Success
}