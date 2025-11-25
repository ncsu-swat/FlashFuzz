#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/lib/core/status.h"

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using namespace tensorflow;

// --- Helper Functions ---

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype;
  switch (selector % 23) {
    case 0: dtype = DT_FLOAT; break;
    case 1: dtype = DT_DOUBLE; break;
    case 2: dtype = DT_INT32; break;
    case 3: dtype = DT_UINT8; break;
    case 4: dtype = DT_INT16; break;
    case 5: dtype = DT_INT8; break;
    case 6: dtype = DT_STRING; break;
    case 7: dtype = DT_COMPLEX64; break;
    case 8: dtype = DT_INT64; break;
    case 9: dtype = DT_BOOL; break;
    case 10: dtype = DT_QINT8; break;
    case 11: dtype = DT_QUINT8; break;
    case 12: dtype = DT_QINT32; break;
    case 13: dtype = DT_BFLOAT16; break;
    case 14: dtype = DT_QINT16; break;
    case 15: dtype = DT_QUINT16; break;
    case 16: dtype = DT_UINT16; break;
    case 17: dtype = DT_COMPLEX128; break;
    case 18: dtype = DT_HALF; break;
    case 19: dtype = DT_UINT32; break;
    case 20: dtype = DT_UINT64; break;
    default: dtype = DT_FLOAT; break;
  }
  return dtype;
}

uint8_t parseRank(uint8_t byte)
{
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

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
      fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size); break;
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size); break;
    default:
      break;
  }
}

// --- Fuzz Target ---

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Basic strict check
    if (Size < 2) return 0;
    
    // Parse inputs
    size_t offset = 0;
    
    uint8_t dtype_byte = Data[offset++];
    tensorflow::DataType dtype = parseDataType(dtype_byte);
    
    uint8_t rank_byte = Data[offset++];
    uint8_t rank = parseRank(rank_byte);

    // Filter DataTypes supported by tf.raw_ops.Relu to avoid noisy/irrelevant failures.
    // Supported: float32, float64, int32, uint8, int16, int8, int64, bfloat16, uint16, half, uint32, uint64, qint8
    bool valid_dtype = false;
    switch(dtype) {
        case DT_FLOAT: case DT_DOUBLE: case DT_INT32: case DT_UINT8: 
        case DT_INT16: case DT_INT8: case DT_INT64: case DT_BFLOAT16: 
        case DT_UINT16: case DT_HALF: case DT_UINT32: case DT_UINT64: 
        case DT_QINT8:
            valid_dtype = true; break;
        default: 
            valid_dtype = false; break;
    }
    if (!valid_dtype) return 0;

    // Parse Shape
    std::vector<int64_t> shape_dims = parseShape(Data, offset, Size, rank);
    tensorflow::TensorShape shape(shape_dims);
    
    // Safety check against OOM
    if (shape.num_elements() > 1000000) return 0;

    // Construct Tensor
    tensorflow::Tensor input_tensor(dtype, shape);
    fillTensorWithDataByType(input_tensor, dtype, Data, offset, Size);

    // Build Graph: Placeholder -> Relu
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    
    auto input_op = tensorflow::ops::Placeholder(root.WithOpName("input"), dtype);
    auto relu_op = tensorflow::ops::Relu(root.WithOpName("output"), input_op);

    tensorflow::GraphDef graph_def;
    tensorflow::Status status = root.ToGraphDef(&graph_def);
    if (!status.ok()) return 0;

    // Prepare Session
    tensorflow::SessionOptions options;
    // Optimize for fuzzing speed/stability
    options.config.set_intra_op_parallelism_threads(1);
    options.config.set_inter_op_parallelism_threads(1);

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
    status = session->Create(graph_def);
    if (!status.ok()) return 0;

    std::vector<tensorflow::Tensor> outputs;
    
    try {
        // Run graph
        status = session->Run({{"input", input_tensor}}, {"output"}, {}, &outputs);
        if (!status.ok()) {
            // Uncomment to debug specific failures
            // std::cout << "TF Runtime Error: " << status.ToString() << std::endl;
        } else {
            // Force data read to trigger any lazy evaluation issues if present
            if (!outputs.empty()) {
                (void)outputs[0].NumElements();
            }
        }
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
    }

    return 0;
}
