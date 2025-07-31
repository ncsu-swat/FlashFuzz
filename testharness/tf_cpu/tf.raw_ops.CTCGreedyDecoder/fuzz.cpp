#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
    }
    return dtype;
}

uint8_t parseRank(uint8_t byte) {
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
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType inputs_dtype = parseDataType(data[offset++]);
        
        uint8_t inputs_rank = 3;
        std::vector<int64_t> inputs_shape = {3, 2, 4};
        
        if (offset + 3 * sizeof(int64_t) <= size) {
            inputs_shape = parseShape(data, offset, size, inputs_rank);
            if (inputs_shape.size() != 3) {
                inputs_shape = {3, 2, 4};
            }
        }
        
        tensorflow::TensorShape inputs_tensor_shape;
        for (auto dim : inputs_shape) {
            inputs_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor inputs_tensor(inputs_dtype, inputs_tensor_shape);
        fillTensorWithDataByType(inputs_tensor, inputs_dtype, data, offset, size);
        
        int32_t batch_size = static_cast<int32_t>(inputs_shape[1]);
        tensorflow::TensorShape seq_len_shape;
        seq_len_shape.AddDim(batch_size);
        tensorflow::Tensor seq_len_tensor(tensorflow::DT_INT32, seq_len_shape);
        
        auto seq_len_flat = seq_len_tensor.flat<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t len;
                std::memcpy(&len, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                len = std::abs(len) % static_cast<int32_t>(inputs_shape[0]) + 1;
                seq_len_flat(i) = len;
            } else {
                seq_len_flat(i) = static_cast<int32_t>(inputs_shape[0]);
            }
        }
        
        bool merge_repeated = false;
        if (offset < size) {
            merge_repeated = (data[offset++] % 2) == 1;
        }
        
        int blank_index = -1;
        if (offset + sizeof(int32_t) <= size) {
            int32_t idx;
            std::memcpy(&idx, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            blank_index = static_cast<int>(idx % static_cast<int32_t>(inputs_shape[2]));
        }
        
        auto inputs_op = tensorflow::ops::Const(root, inputs_tensor);
        auto seq_len_op = tensorflow::ops::Const(root, seq_len_tensor);
        
        // Use raw_ops namespace for CTCGreedyDecoder
        auto ctc_decoder = tensorflow::ops::CTCGreedyDecoder(
            root, inputs_op, seq_len_op, merge_repeated);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({ctc_decoder.decoded_indices,
                                                ctc_decoder.decoded_values,
                                                ctc_decoder.decoded_shape,
                                                ctc_decoder.log_probability}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}