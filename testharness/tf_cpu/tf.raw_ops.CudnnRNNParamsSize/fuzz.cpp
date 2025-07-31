#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
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

tensorflow::DataType parseDataTypeT(uint8_t selector) {
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

tensorflow::DataType parseDataTypeS(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType T_dtype = parseDataTypeT(data[offset++]);
        tensorflow::DataType S_dtype = parseDataTypeS(data[offset++]);
        
        std::string rnn_mode = parseRnnMode(data[offset++]);
        std::string input_mode = parseInputMode(data[offset++]);
        std::string direction = parseDirection(data[offset++]);
        
        float dropout = 0.0f;
        if (offset < size) {
            uint8_t dropout_byte = data[offset++];
            dropout = static_cast<float>(dropout_byte) / 255.0f;
        }
        
        int64_t seed = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        int64_t seed2 = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        int64_t num_proj = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&num_proj, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_proj = std::abs(num_proj) % 100;
        }

        uint8_t num_layers_rank = parseRank(data[offset++]);
        std::vector<int64_t> num_layers_shape = parseShape(data, offset, size, num_layers_rank);
        tensorflow::Tensor num_layers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(num_layers_shape));
        fillTensorWithDataByType(num_layers_tensor, tensorflow::DT_INT32, data, offset, size);

        uint8_t num_units_rank = parseRank(data[offset++]);
        std::vector<int64_t> num_units_shape = parseShape(data, offset, size, num_units_rank);
        tensorflow::Tensor num_units_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(num_units_shape));
        fillTensorWithDataByType(num_units_tensor, tensorflow::DT_INT32, data, offset, size);

        uint8_t input_size_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_size_shape = parseShape(data, offset, size, input_size_rank);
        tensorflow::Tensor input_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(input_size_shape));
        fillTensorWithDataByType(input_size_tensor, tensorflow::DT_INT32, data, offset, size);

        auto num_layers_op = tensorflow::ops::Const(root, num_layers_tensor);
        auto num_units_op = tensorflow::ops::Const(root, num_units_tensor);
        auto input_size_op = tensorflow::ops::Const(root, input_size_tensor);

        // Use raw_ops namespace for CudnnRNNParamsSize
        auto cudnn_rnn_params_size = tensorflow::ops::internal::CudnnRNNParamsSize(
            root, num_layers_op, num_units_op, input_size_op,
            T_dtype, S_dtype, rnn_mode, input_mode, direction, dropout,
            seed, seed2, num_proj
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({cudnn_rnn_params_size}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}