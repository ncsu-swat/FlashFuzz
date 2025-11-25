#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

// Define constants for fuzzing
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace {

using namespace tensorflow;

// Helper to select a random DataType
tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 21) {  // Excludes STRING and variants for simplicity
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
      dtype = DT_COMPLEX64;
      break;
    case 7:
      dtype = DT_INT64;
      break;
    case 8:
      dtype = DT_BOOL;
      break;
    case 9:
      dtype = DT_QINT8;
      break;
    case 10:
      dtype = DT_QUINT8;
      break;
    case 11:
      dtype = DT_QINT32;
      break;
    case 12:
      dtype = DT_BFLOAT16;
      break;
    case 13:
      dtype = DT_QINT16;
      break;
    case 14:
      dtype = DT_QUINT16;
      break;
    case 15:
      dtype = DT_UINT16;
      break;
    case 16:
      dtype = DT_COMPLEX128;
      break;
    case 17:
      dtype = DT_HALF;
      break;
    case 18:
      dtype = DT_UINT32;
      break;
    case 19:
      dtype = DT_UINT64;
      break;
    case 20:
      dtype = DT_STRING;
      break;
    default:
      dtype = DT_FLOAT;
      break;
  }
  return dtype;
}

// Helper to select a random rank
uint8_t parseRank(uint8_t byte) {
  constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
  uint8_t rank = byte % range + MIN_RANK;
  return rank;
}

// Helper to construct a shape vector from fuzz data
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
      shape.push_back(dim_val % MAX_TENSOR_SHAPE_DIMS_TF);
    } else {
      shape.push_back(MIN_TENSOR_SHAPE_DIMS_TF);
    }
  }

  return shape;
}

// Template helper to fill tensor with data
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

// Helper to dispatch tensor filling based on DataType
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
    case DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case DT_UINT16:
      fillTensorWithData<uint16>(tensor, data, offset, total_size);
      break;
    case DT_UINT32:
      fillTensorWithData<uint32>(tensor, data, offset, total_size);
      break;
    case DT_UINT64:
      fillTensorWithData<uint64>(tensor, data, offset, total_size);
      break;
    case DT_BFLOAT16:
      fillTensorWithData<bfloat16>(tensor, data, offset, total_size);
      break;
    case DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX64:
      fillTensorWithData<complex64>(tensor, data, offset, total_size);
      break;
    case DT_COMPLEX128:
      fillTensorWithData<complex128>(tensor, data, offset, total_size);
      break;
    case DT_QINT8:
      fillTensorWithData<qint8>(tensor, data, offset, total_size);
      break;
    case DT_QUINT8:
      fillTensorWithData<quint8>(tensor, data, offset, total_size);
      break;
    case DT_QINT16:
      fillTensorWithData<qint16>(tensor, data, offset, total_size);
      break;
    case DT_QUINT16:
      fillTensorWithData<quint16>(tensor, data, offset, total_size);
      break;
    case DT_QINT32:
      fillTensorWithData<qint32>(tensor, data, offset, total_size);
      break;
    case DT_STRING: {
      auto flat = tensor.flat<tstring>();
      for (int i = 0; i < flat.size(); ++i) {
        if (offset < total_size) {
          uint8_t len = *(data + offset) % 16;
          offset++;
          if (offset + len <= total_size) {
            flat(i).assign(reinterpret_cast<const char*>(data + offset), len);
            offset += len;
          }
        }
      }
      break;
    }
    default:
      // Unsupported types are skipped.
      break;
  }
}

// Helper to fill the 'shape' tensor with potentially valid/invalid values
template <typename T>
void fillShapeTensor(Tensor& shape_tensor, const uint8_t* data, size_t& offset,
                     size_t total_size) {
  auto flat = shape_tensor.flat<T>();
  for (int i = 0; i < flat.size(); ++i) {
    if (offset + sizeof(T) <= total_size) {
      T val;
      memcpy(&val, data + offset, sizeof(T));
      offset += sizeof(T);
      // Introduce special values like -1
      if ((val & 0xFF) == 0xFF) {
        flat(i) = -1;
      } else {
        flat(i) = static_cast<T>(std::abs(val) % (MAX_TENSOR_SHAPE_DIMS_TF + 5));
      }
    } else {
      flat(i) = 1;
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // We need at least a few bytes to define the tensors.
  if (size < 10) {
    return 0;
  }

  size_t offset = 0;

  // 1. Create the input 'tensor'
  DataType tensor_dtype = parseDataType(data[offset++]);
  uint8_t tensor_rank = parseRank(data[offset++]);
  std::vector<int64_t> tensor_dims =
      parseShape(data, offset, size, tensor_rank);
  TensorShape tensor_shape;
  if (!TensorShape::BuildTensorShape(tensor_dims, &tensor_shape).ok()) {
    return 0;  // Invalid shape, cannot proceed.
  }

  // Avoid creating tensors that are too large and would OOM.
  if (tensor_shape.num_elements() > 10000) {
    return 0;
  }
  
  Tensor input_tensor(tensor_dtype, tensor_shape);
  fillTensorWithDataByType(input_tensor, tensor_dtype, data, offset, size);

  // 2. Create the 'shape' tensor
  DataType shape_dtype = (data[offset++] % 2 == 0) ? DT_INT32 : DT_INT64;
  // 'shape' must be a 1-D tensor.
  uint8_t new_dims_count = data[offset++] % (MAX_RANK + 2);
  TensorShape shape_tensor_shape({new_dims_count});
  Tensor shape_tensor(shape_dtype, shape_tensor_shape);
  if (shape_dtype == DT_INT32) {
    fillShapeTensor<int32>(shape_tensor, data, offset, size);
  } else {
    fillShapeTensor<int64_t>(shape_tensor, data, offset, size);
  }

  // 3. Set up the OpKernel and its context
  try {
    static std::unique_ptr<Device> device = ([] {
      SessionOptions options;
      std::vector<std::unique_ptr<Device>> devices;
      if (!DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0",
                                      &devices)
               .ok()) {
        return std::unique_ptr<Device>(nullptr);
      }
      return std::move(devices[0]);
    })();
    if (!device) {
      return 0;
    }

    NodeDef node_def;
    node_def.set_name("fuzz_reshape_op");
    node_def.set_op("Reshape");
    AddNodeAttr("T", input_tensor.dtype(), &node_def);
    AddNodeAttr("Tshape", shape_tensor.dtype(), &node_def);

    Status status;
    std::unique_ptr<OpKernel> op_kernel = CreateOpKernel(
        DEVICE_CPU, device.get(), device->GetAllocator(AllocatorAttributes()),
        node_def, TF_GRAPH_DEF_VERSION, &status);
    if (!status.ok()) {
      return 0;
    }

    gtl::InlinedVector<TensorValue, 4> inputs;
    inputs.emplace_back(&input_tensor);
    inputs.emplace_back(&shape_tensor);

    OpKernelContext::Params params;
    params.device = device.get();
    params.op_kernel = op_kernel.get();
    params.inputs = inputs;
    std::vector<AllocatorAttributes> output_attrs(op_kernel->num_outputs());
    params.output_attr_array = output_attrs.data();

    OpKernelContext context(&params);
    op_kernel->Compute(&context);

    // We don't need to check the output, just that it doesn't crash.
    // The fuzzer will report any crashes found.
    (void)context.status();

  } catch (const std::exception& e) {
    // This is useful for debugging but can be noisy.
    // std::cerr << "Caught exception: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    // Catch all other types of exceptions.
    return -1;
  }

  return 0;
}