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
#include <cstring>
#include <vector>
#include <iostream>

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
    if (size < 100) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        uint8_t input_h_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_h_shape = parseShape(data, offset, size, input_h_rank);
        
        uint8_t input_c_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_c_shape = parseShape(data, offset, size, input_c_rank);
        
        uint8_t params_rank = 1;
        std::vector<int64_t> params_shape = {10};
        
        uint8_t seq_len_rank = 1;
        std::vector<int64_t> seq_len_shape = {2};
        
        uint8_t output_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape = parseShape(data, offset, size, output_rank);
        
        uint8_t output_h_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_h_shape = parseShape(data, offset, size, output_h_rank);
        
        uint8_t output_c_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_c_shape = parseShape(data, offset, size, output_c_rank);
        
        uint8_t reserve_space_rank = 1;
        std::vector<int64_t> reserve_space_shape = {100};
        
        uint8_t host_reserved_rank = 1;
        std::vector<int64_t> host_reserved_shape = {50};

        tensorflow::Tensor input_tensor(dtype, tensorflow::TensorShape(input_shape));
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor input_h_tensor(dtype, tensorflow::TensorShape(input_h_shape));
        fillTensorWithDataByType(input_h_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor input_c_tensor(dtype, tensorflow::TensorShape(input_c_shape));
        fillTensorWithDataByType(input_c_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor params_tensor(dtype, tensorflow::TensorShape(params_shape));
        fillTensorWithDataByType(params_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor seq_len_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(seq_len_shape));
        fillTensorWithData<int32_t>(seq_len_tensor, data, offset, size);
        
        tensorflow::Tensor output_tensor(dtype, tensorflow::TensorShape(output_shape));
        fillTensorWithDataByType(output_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor output_h_tensor(dtype, tensorflow::TensorShape(output_h_shape));
        fillTensorWithDataByType(output_h_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor output_c_tensor(dtype, tensorflow::TensorShape(output_c_shape));
        fillTensorWithDataByType(output_c_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor output_backprop_tensor(dtype, tensorflow::TensorShape(output_shape));
        fillTensorWithDataByType(output_backprop_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor output_h_backprop_tensor(dtype, tensorflow::TensorShape(output_h_shape));
        fillTensorWithDataByType(output_h_backprop_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor output_c_backprop_tensor(dtype, tensorflow::TensorShape(output_c_shape));
        fillTensorWithDataByType(output_c_backprop_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor reserve_space_tensor(dtype, tensorflow::TensorShape(reserve_space_shape));
        fillTensorWithDataByType(reserve_space_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor host_reserved_tensor(tensorflow::DT_INT8, tensorflow::TensorShape(host_reserved_shape));
        fillTensorWithData<int8_t>(host_reserved_tensor, data, offset, size);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto input_h_op = tensorflow::ops::Const(root, input_h_tensor);
        auto input_c_op = tensorflow::ops::Const(root, input_c_tensor);
        auto params_op = tensorflow::ops::Const(root, params_tensor);
        auto seq_len_op = tensorflow::ops::Const(root, seq_len_tensor);
        auto output_op = tensorflow::ops::Const(root, output_tensor);
        auto output_h_op = tensorflow::ops::Const(root, output_h_tensor);
        auto output_c_op = tensorflow::ops::Const(root, output_c_tensor);
        auto output_backprop_op = tensorflow::ops::Const(root, output_backprop_tensor);
        auto output_h_backprop_op = tensorflow::ops::Const(root, output_h_backprop_tensor);
        auto output_c_backprop_op = tensorflow::ops::Const(root, output_c_backprop_tensor);
        auto reserve_space_op = tensorflow::ops::Const(root, reserve_space_tensor);
        auto host_reserved_op = tensorflow::ops::Const(root, host_reserved_tensor);

        std::string rnn_mode = parseRnnMode(data[offset % size]);
        std::string input_mode = parseInputMode(data[(offset + 1) % size]);
        std::string direction = parseDirection(data[(offset + 2) % size]);
        
        float dropout = 0.0f;
        int seed = 0;
        int seed2 = 0;
        int num_proj = 0;
        bool time_major = true;

        // Use raw_ops approach since CudnnRNNBackpropV3 is not directly available in ops namespace
        auto attrs = tensorflow::ops::Raw::Attrs()
            .Set("rnn_mode", rnn_mode)
            .Set("input_mode", input_mode)
            .Set("direction", direction)
            .Set("dropout", dropout)
            .Set("seed", seed)
            .Set("seed2", seed2)
            .Set("num_proj", num_proj)
            .Set("time_major", time_major);

        auto cudnn_rnn_backprop = tensorflow::ops::Raw::CudnnRNNBackpropV3(
            root,
            input_op,
            input_h_op,
            input_c_op,
            params_op,
            seq_len_op,
            output_op,
            output_h_op,
            output_c_op,
            output_backprop_op,
            output_h_backprop_op,
            output_c_backprop_op,
            reserve_space_op,
            host_reserved_op,
            attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            {cudnn_rnn_backprop.output[0], cudnn_rnn_backprop.output[1], 
             cudnn_rnn_backprop.output[2], cudnn_rnn_backprop.output[3]}
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}