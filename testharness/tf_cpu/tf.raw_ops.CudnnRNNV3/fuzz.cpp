#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
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
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::string rnn_mode = parseRnnMode(data[offset++]);
        std::string input_mode = parseInputMode(data[offset++]);
        std::string direction = parseDirection(data[offset++]);
        
        bool is_training = (data[offset++] % 2) == 1;
        bool time_major = (data[offset++] % 2) == 1;
        
        float dropout = static_cast<float>(data[offset++]) / 255.0f;
        int seed = static_cast<int>(data[offset++]);
        int seed2 = static_cast<int>(data[offset++]);
        int num_proj = static_cast<int>(data[offset++] % 10);
        
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        if (input_shape.size() != 3) {
            input_shape = {2, 3, 4};
        }
        
        int64_t seq_length = input_shape[time_major ? 0 : 1];
        int64_t batch_size = input_shape[time_major ? 1 : 0];
        int64_t input_size = input_shape[2];
        int64_t num_units = input_size;
        int64_t num_layers = 1;
        int64_t dir_count = (direction == "bidirectional") ? 2 : 1;
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        std::vector<int64_t> h_shape = time_major ? 
            std::vector<int64_t>{num_layers * dir_count, batch_size, num_units} :
            std::vector<int64_t>{batch_size, num_layers * dir_count, num_units};
        tensorflow::TensorShape input_h_shape(h_shape);
        tensorflow::Tensor input_h_tensor(dtype, input_h_shape);
        fillTensorWithDataByType(input_h_tensor, dtype, data, offset, size);
        
        tensorflow::TensorShape input_c_shape(h_shape);
        tensorflow::Tensor input_c_tensor(dtype, input_c_shape);
        fillTensorWithDataByType(input_c_tensor, dtype, data, offset, size);
        
        int64_t params_size = 1000;
        tensorflow::TensorShape params_shape({params_size});
        tensorflow::Tensor params_tensor(dtype, params_shape);
        fillTensorWithDataByType(params_tensor, dtype, data, offset, size);
        
        tensorflow::TensorShape seq_len_shape({batch_size});
        tensorflow::Tensor seq_len_tensor(tensorflow::DT_INT32, seq_len_shape);
        auto seq_len_flat = seq_len_tensor.flat<int32_t>();
        for (int i = 0; i < batch_size; ++i) {
            seq_len_flat(i) = seq_length;
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto input_h_op = tensorflow::ops::Const(root, input_h_tensor);
        auto input_c_op = tensorflow::ops::Const(root, input_c_tensor);
        auto params_op = tensorflow::ops::Const(root, params_tensor);
        auto seq_len_op = tensorflow::ops::Const(root, seq_len_tensor);
        
        // Use raw_ops namespace for CudnnRNNV3
        auto cudnn_rnn_op = tensorflow::ops::internal::CudnnRNNV3(
            root,
            input_op,
            input_h_op,
            input_c_op,
            params_op,
            seq_len_op,
            tensorflow::Input(), // host_reserved
            rnn_mode,
            input_mode,
            direction,
            dropout,
            seed,
            seed2,
            num_proj,
            is_training,
            time_major
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({cudnn_rnn_op.output, cudnn_rnn_op.output_h, 
                                                 cudnn_rnn_op.output_c, cudnn_rnn_op.reserve_space,
                                                 cudnn_rnn_op.host_reserved}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
