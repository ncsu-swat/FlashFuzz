#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/platform/tstring.h"

using namespace tensorflow;

// Helper to select only Conv2D compatible data types
tensorflow::DataType parseAllowedDataType(uint8_t selector) {
  switch (selector % 5) {
    case 0: return DT_FLOAT;
    case 1: return DT_DOUBLE;
    case 2: return DT_INT32;
    case 3: return DT_HALF;
    case 4: return DT_BFLOAT16;
    default: return DT_FLOAT;
  }
}

// Template helper to fill tensor with data from fuzz input
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
      // Fill remaining with 0 to avoid uninitialized memory
      flat(i) = static_cast<T>(0);
    }
  }
}

// Dispatcher for filling tensor based on DataType
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

// Parse a 4D shape from fuzz data, keeping dimensions small
std::vector<int64_t> parseShape4D(const uint8_t* data, size_t& offset, size_t total_size) {
  std::vector<int64_t> shape;
  shape.reserve(4);
  for (int i = 0; i < 4; ++i) {
    if (offset < total_size) {
      // Modulo 16 + 1 ensures dimensions 1..16 to limit memory usage
      shape.push_back((data[offset] % 16) + 1);
      offset++;
    } else {
      shape.push_back(1);
    }
  }
  return shape;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Need enough data for parameters and at least some shape info
  if (size < 32) return 0;

  size_t offset = 0;

  // 1. Randomly select parameters
  DataType dtype = parseAllowedDataType(data[offset++]);
  
  // Data format: NHWC or NCHW
  bool is_nhwc = (data[offset++] % 2) == 0;
  std::string data_format = is_nhwc ? "NHWC" : "NCHW";
  
  // Padding
  uint8_t pad_byte = data[offset++];
  std::string padding;
  bool is_explicit = false;
  int pad_mode = pad_byte % 3;
  if (pad_mode == 0) padding = "SAME";
  else if (pad_mode == 1) padding = "VALID";
  else { padding = "EXPLICIT"; is_explicit = true; }

  // Strides (4 ints)
  std::vector<int> strides;
  for (int i = 0; i < 4; ++i) {
    // Strides usually 1 or 2, rarely larger for fuzzing interest
    strides.push_back((data[offset++] % 4) + 1);
  }

  // Dilations (4 ints)
  std::vector<int> dilations;
  for (int i = 0; i < 4; ++i) {
    dilations.push_back((data[offset++] % 3) + 1);
  }

  // Explicit paddings (8 ints) if EXPLICIT, else empty
  std::vector<int> explicit_paddings;
  if (is_explicit) {
    for (int i = 0; i < 8; ++i) {
      if (offset < size) {
        explicit_paddings.push_back(data[offset++] % 8);
      } else {
        explicit_paddings.push_back(0);
      }
    }
  }

  // Use Cudnn
  bool use_cudnn = (offset < size) ? (data[offset++] % 2) : true;

  // 2. Parse Shapes
  // Input: 4D
  std::vector<int64_t> input_shape_vec = parseShape4D(data, offset, size);
  
  // Filter: 4D
  std::vector<int64_t> filter_shape_vec = parseShape4D(data, offset, size);

  // Constraint: Input channels must match filter input channels.
  // NHWC: [Batch, H, W, Channel]. Filter: [H, W, InChannel, OutChannel].
  // NCHW: [Batch, Channel, H, W].
  int in_channel_dim = is_nhwc ? 3 : 1;
  int64_t in_channels = input_shape_vec[in_channel_dim];
  // Filter input channel is always dimension 2 in the filter tensor definition
  filter_shape_vec[2] = in_channels;

  TensorShape input_shape(input_shape_vec);
  TensorShape filter_shape(filter_shape_vec);

  // Safety check on tensor sizes to avoid massive allocations/timeouts
  // 16^4 is 65536, which is small, but if dims are larger it can blow up.
  if (input_shape.num_elements() > 200000 || filter_shape.num_elements() > 200000) {
    return 0;
  }

  // 3. Prepare Tensors
  Tensor input_tensor(dtype, input_shape);
  Tensor filter_tensor(dtype, filter_shape);

  fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
  fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);

  // 4. Construct Graph
  Scope root = Scope::NewRootScope();
  
  auto input_node = ops::Const(root.WithOpName("input"), input_tensor);
  auto filter_node = ops::Const(root.WithOpName("filter"), filter_tensor);

  auto conv = ops::Conv2D(root.WithOpName("conv"), input_node, filter_node, 
                          strides, padding, 
                          ops::Conv2D::DataFormat(data_format)
                          .Dilations(dilations)
                          .UseCudnnOnGpu(use_cudnn)
                          .ExplicitPaddings(explicit_paddings));

  // 5. Execute
  ClientSession session(root);
  std::vector<Tensor> outputs;
  
  try {
      // Run the op.
      Status status = session.Run({conv}, &outputs);
      // We do not check status as we expect many invalid configs (e.g. dimensions mismatch with strides)
      // The fuzzer's goal is to find crashes/assertions.
  } catch (...) {
      // Catch C++ exceptions if any
  }

  return 0;
}