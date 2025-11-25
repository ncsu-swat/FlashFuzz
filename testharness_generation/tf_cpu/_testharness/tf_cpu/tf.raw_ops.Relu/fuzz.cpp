#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/bfloat16.h"

// Define fuzzer-controlled constants for tensor generation
#define MIN_RANK 0
#define MAX_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 16

// Using tensorflow namespace for convenience
using namespace tensorflow;

// Helper function to parse a DataType from the fuzzer input
DataType parseDataType(uint8_t selector) {
  DataType dtype;
  switch (selector % 21) { // Adjusted to number of supported + some invalid types
    case 0: dtype = DT_FLOAT; break;
    case 1: dtype = DT_DOUBLE; break;
    case 2: dtype = DT_INT32; break;
    case 3: dtype = DT_UINT8; break;
    case 4: dtype = DT_INT16; break;
    case 5: dtype = DT_INT8; break;
    case 6: dtype = DT_STRING; break; // Invalid for Relu, tests API robustness
    case 7: dtype = DT_COMPLEX64; break; // Invalid for Relu
    case 8: dtype = DT_INT64; break;
    case 9: dtype = DT_BOOL; break; // Invalid for Relu
    case 10: dtype = DT_QINT8; break;
    case 11: dtype = DT_QUINT8; break; // Invalid for Relu
    case 12: dtype = DT_QINT32; break; // Invalid for Relu
    case 13: dtype = DT_BFLOAT16; break;
    case 14: dtype = DT_QINT16; break; // Invalid for Relu
    case 15: dtype = DT_QUINT16; break; // Invalid for Relu
    case 16: dtype = DT_UINT16; break;
    case 17: dtype = DT_COMPLEX128; break; // Invalid for Relu
    case 18: dtype = DT_HALF; break;
    case 19: dtype = DT_UINT32; break;
    case 20: dtype = DT_UINT64; break;
    default: dtype = DT_FLOAT; break;
  }
  return dtype;
}

// Helper function to parse a tensor rank from the fuzzer input
uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

// Helper function to parse a tensor shape from the fuzzer input
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

// Template function to fill a tensor's buffer with data from the fuzzer input
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
      flat(i) = T{};
    }
  }
}

// Dispatcher function to call the correct typed version of fillTensorWithData
void fillTensorWithDataByType(Tensor& tensor, DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case DT_FLOAT: fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case DT_DOUBLE: fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case DT_INT32: fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    case DT_UINT8: fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case DT_INT16: fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case DT_INT8: fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case DT_INT64: fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
    case DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case DT_UINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case DT_UINT32: fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
    case DT_UINT64: fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
    case DT_BFLOAT16: fillTensorWithData<bfloat16>(tensor, data, offset, total_size); break;
    case DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case DT_COMPLEX64: fillTensorWithData<complex64>(tensor, data, offset, total_size); break;
    case DT_COMPLEX128: fillTensorWithData<complex128>(tensor, data, offset, total_size); break;
    case DT_QINT8: fillTensorWithData<qint8>(tensor, data, offset, total_size); break;
    default: return;
  }
}

// LibFuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // We need at least 2 bytes for dtype and rank.
    if (size < 2) {
        return 0;
    }

    try {
        size_t offset = 0;

        // 1. Determine tensor properties from fuzzer data
        DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_vec = parseShape(data, offset, size, rank);
        
        TensorShape shape;
        Status status = TensorShapeUtils::MakeShape(shape_vec, &shape);
        if (!status.ok()) {
            return 0; // Invalid shape, skip this input
        }

        // Avoid creating tensors that are too large and would cause OOM.
        if (shape.num_elements() > 1000000) {
            return 0;
        }

        // 2. Create and fill the input tensor
        Tensor input_tensor(dtype, shape);
        if (dtype == DT_STRING) {
            auto flat = input_tensor.flat<tstring>();
            if (flat.size() > 0 && offset < size) {
                flat(0) = std::string(reinterpret_cast<const char*>(data + offset), size - offset);
            }
        } else {
            fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        }

        // 3. Build and run the TensorFlow graph for the Relu op
        Scope root = Scope::NewRootScope();
        auto input_op = ops::Const(root.WithOpName("input"), input_tensor);
        auto relu_op = ops::Relu(root.WithOpName("relu"), input_op);
        
        ClientSession session(root);
        std::vector<Tensor> outputs;
        
        // Execute the graph. The API might return a bad status or throw an
        // exception for invalid inputs, both are handled.
        session.Run({relu_op}, &outputs);

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        // The fuzzer should continue running even after finding an issue.
        return 0;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return 0;
    }

    return 0; // Success
}