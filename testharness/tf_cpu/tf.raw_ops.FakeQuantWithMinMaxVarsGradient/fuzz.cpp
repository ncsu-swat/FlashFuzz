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
    std::cerr << "Error: " << message << std::endl;
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t gradients_rank = parseRank(data[offset++]);
        std::vector<int64_t> gradients_shape = parseShape(data, offset, size, gradients_rank);
        
        uint8_t inputs_rank = parseRank(data[offset++]);
        std::vector<int64_t> inputs_shape = parseShape(data, offset, size, inputs_rank);
        
        tensorflow::TensorShape gradients_tensor_shape(gradients_shape);
        tensorflow::TensorShape inputs_tensor_shape(inputs_shape);
        
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, gradients_tensor_shape);
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, inputs_tensor_shape);
        
        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(inputs_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int num_bits = 8;
        bool narrow_range = false;
        
        if (offset < size) {
            num_bits = 2 + (data[offset++] % 7);
        }
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }
        
        auto gradients_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto inputs_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto min_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto fake_quant_grad = tensorflow::ops::FakeQuantWithMinMaxVarsGradient(
            root, gradients_placeholder, inputs_placeholder, min_placeholder, max_placeholder,
            tensorflow::ops::FakeQuantWithMinMaxVarsGradient::Attrs()
                .NumBits(num_bits)
                .NarrowRange(narrow_range)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{gradients_placeholder, gradients_tensor},
             {inputs_placeholder, inputs_tensor},
             {min_placeholder, min_tensor},
             {max_placeholder, max_tensor}},
            {fake_quant_grad.backprops_wrt_input, fake_quant_grad.backprop_wrt_min, fake_quant_grad.backprop_wrt_max},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
