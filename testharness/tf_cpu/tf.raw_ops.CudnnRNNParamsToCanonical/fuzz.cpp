#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

std::string parseRnnMode(uint8_t selector) {
    switch (selector % 4) {
        case 0: return "rnn_relu";
        case 1: return "rnn_tanh";
        case 2: return "lstm";
        case 3: return "gru";
    }
    return "lstm";
}

std::string parseInputMode(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "linear_input";
        case 1: return "skip_input";
        case 2: return "auto_select";
    }
    return "linear_input";
}

std::string parseDirection(uint8_t selector) {
    switch (selector % 2) {
        case 0: return "unidirectional";
        case 1: return "bidirectional";
    }
    return "unidirectional";
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        int32_t num_layers_val = 1 + (data[offset++] % 5);
        
        if (offset >= size) return 0;
        int32_t num_units_val = 1 + (data[offset++] % 100);
        
        if (offset >= size) return 0;
        int32_t input_size_val = 1 + (data[offset++] % 100);
        
        if (offset >= size) return 0;
        int32_t num_params_val = 1 + (data[offset++] % 20);
        
        if (offset >= size) return 0;
        tensorflow::DataType params_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        std::string rnn_mode = parseRnnMode(data[offset++]);
        
        if (offset >= size) return 0;
        std::string input_mode = parseInputMode(data[offset++]);
        
        if (offset >= size) return 0;
        std::string direction = parseDirection(data[offset++]);
        
        if (offset >= size) return 0;
        float dropout = (data[offset++] % 101) / 100.0f;
        
        if (offset + 1 >= size) return 0;
        int32_t seed = static_cast<int32_t>(data[offset++]);
        int32_t seed2 = static_cast<int32_t>(data[offset++]);

        tensorflow::Tensor num_layers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_layers_tensor.scalar<int32_t>()() = num_layers_val;

        tensorflow::Tensor num_units_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_units_tensor.scalar<int32_t>()() = num_units_val;

        tensorflow::Tensor input_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        input_size_tensor.scalar<int32_t>()() = input_size_val;

        if (offset >= size) return 0;
        uint8_t params_rank = parseRank(data[offset++]);
        std::vector<int64_t> params_shape = parseShape(data, offset, size, params_rank);
        
        tensorflow::TensorShape params_tensor_shape;
        for (int64_t dim : params_shape) {
            params_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor params_tensor(params_dtype, params_tensor_shape);
        fillTensorWithDataByType(params_tensor, params_dtype, data, offset, size);

        auto num_layers_op = tensorflow::ops::Const(root, num_layers_tensor);
        auto num_units_op = tensorflow::ops::Const(root, num_units_tensor);
        auto input_size_op = tensorflow::ops::Const(root, input_size_tensor);
        auto params_op = tensorflow::ops::Const(root, params_tensor);

        // Use raw_ops namespace for CudnnRNNParamsToCanonical
        auto cudnn_rnn_params_to_canonical = tensorflow::ops::internal::CudnnRNNParamsToCanonical(
            root,
            num_layers_op,
            num_units_op,
            input_size_op,
            params_op,
            num_params_val,
            rnn_mode,
            input_mode,
            direction,
            dropout,
            seed,
            seed2
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({cudnn_rnn_params_to_canonical.weights, cudnn_rnn_params_to_canonical.biases}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}