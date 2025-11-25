#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>

#include "tensorflow/core/common_runtime/device.h"
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
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session_options.h"

#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using namespace tensorflow;

// Helper function to parse data type from fuzz input
tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 23) {
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
      dtype = DT_QINT8;
      break;
    case 11:
      dtype = DT_QUINT8;
      break;
    case 12:
      dtype = DT_QINT32;
      break;
    case 13:
      dtype = DT_BFLOAT16;
      break;
    case 14:
      dtype = DT_QINT16;
      break;
    case 15:
      dtype = DT_QUINT16;
      break;
    case 16:
      dtype = DT_UINT16;
      break;
    case 17:
      dtype = DT_COMPLEX128;
      break;
    case 18:
      dtype = DT_HALF;
      break;
    case 19:
      dtype = DT_UINT32;
      break;
    case 20:
      dtype = DT_UINT64;
      break;
    default:
      dtype = DT_FLOAT;
      break;
  }
  return dtype;
}

// Helper function to parse rank from fuzz input
uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

// Helper function to parse shape from fuzz input
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

// Helper function to fill tensor data from fuzz input
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

// Helper function to dispatch tensor filling based on data type
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

// Global context to hold the TensorFlow device, initialized once.
struct FuzzContext {
    std::unique_ptr<tensorflow::Device> device;

    FuzzContext() {
        SessionOptions options;
        std::vector<std::unique_ptr<Device>> devices;
        Status status = DeviceFactory::AddDevices(
            options, "/job:localhost/replica:0/task:0", &devices);
        if (!status.ok() || devices.empty()) {
            std::cerr << "Failed to create CPU device: " << status.ToString() << std::endl;
            std::abort();
        }
        device = std::move(devices[0]);
    }
};

static FuzzContext fuzz_context;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return 0; // Not enough data for dtype and rank.
    }

    size_t offset = 0;

    // 1. Construct the 'logits' tensor from fuzzer data.
    DataType dtype = parseDataType(data[offset++]);
    uint8_t rank = parseRank(data[offset++]);
    std::vector<int64_t> shape_vec = parseShape(data, offset, size, rank);
    
    TensorShape shape;
    // Let the op handle invalid shapes from fuzzed data.
    Status shape_status = TensorShape::BuildTensorShape(shape_vec, &shape);
    
    Tensor logits_tensor(dtype, shape);
    if (logits_tensor.NumElements() > 0) {
        fillTensorWithDataByType(logits_tensor, dtype, data, offset, size);
    }
    
    // 2. Set up and run the Softmax OpKernel.
    try {
        // Create NodeDef for the Softmax op.
        NodeDef node_def;
        NodeDefBuilder builder("fuzz_softmax_op", "Softmax");
        builder.Input("logits", 0, dtype);
        builder.Attr("T", dtype);
        if (!builder.Finalize(&node_def).ok()) {
            return 0; // Invalid attributes, nothing to fuzz.
        }

        // Create the OpKernel.
        Status kernel_status;
        std::unique_ptr<OpKernel> op_kernel = CreateOpKernel(
            DEVICE_CPU, fuzz_context.device.get(),
            fuzz_context.device->GetAllocator(AllocatorAttributes()),
            node_def, TF_GRAPH_DEF_VERSION, &kernel_status);
        
        // If the op doesn't support the fuzzed dtype, kernel creation will fail.
        // This is an expected outcome, not a crash.
        if (!kernel_status.ok()) {
            return 0;
        }

        // Prepare inputs for the OpKernelContext.
        gtl::InlinedVector<TensorValue, 4> inputs;
        inputs.push_back(TensorValue(&logits_tensor));

        // Prepare parameters for the OpKernelContext.
        OpKernelContext::Params params;
        params.device = fuzz_context.device.get();
        params.frame_iter = FrameAndIter(0, 0);
        params.inputs = &inputs;
        params.op_kernel = op_kernel.get();
        
        int num_outputs = op_kernel->num_outputs();
        std::vector<AllocatorAttributes> output_attrs(num_outputs);
        params.output_attr_array = output_attrs.data();

        // Create the OpKernelContext.
        OpKernelContext context(&params, num_outputs);

        // Execute the op.
        op_kernel->Compute(&context);

        // We are primarily looking for crashes, so we don't need to assert on the status.
        (void)context.status();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}