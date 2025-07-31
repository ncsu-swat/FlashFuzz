#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank_gradients = parseRank(data[offset++]);
        std::vector<int64_t> shape_gradients = parseShape(data, offset, size, rank_gradients);
        
        uint8_t rank_inputs = parseRank(data[offset++]);
        std::vector<int64_t> shape_inputs = parseShape(data, offset, size, rank_inputs);
        
        if (shape_gradients.empty() || shape_inputs.empty()) {
            return 0;
        }
        
        int64_t d_dim = shape_gradients.back();
        std::vector<int64_t> shape_min = {d_dim};
        std::vector<int64_t> shape_max = {d_dim};
        
        tensorflow::TensorShape tf_shape_gradients(shape_gradients);
        tensorflow::TensorShape tf_shape_inputs(shape_inputs);
        tensorflow::TensorShape tf_shape_min(shape_min);
        tensorflow::TensorShape tf_shape_max(shape_max);
        
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, tf_shape_gradients);
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, tf_shape_inputs);
        tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, tf_shape_min);
        tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, tf_shape_max);
        
        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(inputs_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        auto min_flat = min_tensor.flat<float>();
        auto max_flat = max_tensor.flat<float>();
        for (int i = 0; i < min_flat.size(); ++i) {
            if (min_flat(i) >= max_flat(i)) {
                max_flat(i) = min_flat(i) + 1.0f;
            }
        }
        
        int num_bits = 8;
        bool narrow_range = false;
        
        if (offset < size) {
            num_bits = 2 + (data[offset++] % 15);
        }
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }
        
        auto gradients_op = tensorflow::ops::Const(root, gradients_tensor);
        auto inputs_op = tensorflow::ops::Const(root, inputs_tensor);
        auto min_op = tensorflow::ops::Const(root, min_tensor);
        auto max_op = tensorflow::ops::Const(root, max_tensor);
        
        auto fake_quant_grad = tensorflow::ops::FakeQuantWithMinMaxVarsPerChannelGradient(
            root, gradients_op, inputs_op, min_op, max_op,
            tensorflow::ops::FakeQuantWithMinMaxVarsPerChannelGradient::Attrs()
                .NumBits(num_bits)
                .NarrowRange(narrow_range)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({fake_quant_grad.backprops_wrt_input, 
                                                fake_quant_grad.backprop_wrt_min,
                                                fake_quant_grad.backprop_wrt_max}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}