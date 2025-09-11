#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include <iostream>
#include <cstring>
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

std::string parseStringAttribute(const uint8_t* data, size_t& offset, size_t total_size, const std::vector<std::string>& options) {
    if (offset < total_size) {
        uint8_t selector = data[offset++];
        return options[selector % options.size()];
    }
    return options[0];
}

float parseFloat(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset + sizeof(float) <= total_size) {
        float value;
        std::memcpy(&value, data + offset, sizeof(float));
        offset += sizeof(float);
        return std::abs(value);
    }
    return 0.0f;
}

int parseInt(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset + sizeof(int) <= total_size) {
        int value;
        std::memcpy(&value, data + offset, sizeof(int));
        offset += sizeof(int);
        return std::abs(value);
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<std::string> rnn_modes = {"rnn_relu", "rnn_tanh", "lstm", "gru"};
        std::vector<std::string> input_modes = {"linear_input", "skip_input", "auto_select"};
        std::vector<std::string> directions = {"unidirectional", "bidirectional"};
        
        std::string rnn_mode = parseStringAttribute(data, offset, size, rnn_modes);
        std::string input_mode = parseStringAttribute(data, offset, size, input_modes);
        std::string direction = parseStringAttribute(data, offset, size, directions);
        
        float dropout = parseFloat(data, offset, size);
        if (dropout > 1.0f) dropout = 1.0f;
        
        int seed = parseInt(data, offset, size);
        int seed2 = parseInt(data, offset, size);

        int seq_length = 2;
        int batch_size = 2;
        int input_size = 4;
        int num_units = 4;
        int num_layers = 1;
        int dir_multiplier = (direction == "bidirectional") ? 2 : 1;

        tensorflow::TensorShape input_shape({seq_length, batch_size, input_size});
        tensorflow::TensorShape input_h_shape({num_layers * dir_multiplier, batch_size, num_units});
        tensorflow::TensorShape input_c_shape({num_layers * dir_multiplier, batch_size, num_units});
        tensorflow::TensorShape output_shape({seq_length, batch_size, dir_multiplier * num_units});
        tensorflow::TensorShape output_h_shape({num_layers * dir_multiplier, batch_size, num_units});
        tensorflow::TensorShape output_c_shape({num_layers * dir_multiplier, batch_size, num_units});
        
        int params_size = 1000;
        tensorflow::TensorShape params_shape({params_size});
        tensorflow::TensorShape reserve_space_shape({1000});

        tensorflow::Tensor input_tensor(dtype, input_shape);
        tensorflow::Tensor input_h_tensor(dtype, input_h_shape);
        tensorflow::Tensor input_c_tensor(dtype, input_c_shape);
        tensorflow::Tensor params_tensor(dtype, params_shape);
        tensorflow::Tensor output_tensor(dtype, output_shape);
        tensorflow::Tensor output_h_tensor(dtype, output_h_shape);
        tensorflow::Tensor output_c_tensor(dtype, output_c_shape);
        tensorflow::Tensor output_backprop_tensor(dtype, output_shape);
        tensorflow::Tensor output_h_backprop_tensor(dtype, output_h_shape);
        tensorflow::Tensor output_c_backprop_tensor(dtype, output_c_shape);
        tensorflow::Tensor reserve_space_tensor(dtype, reserve_space_shape);

        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(input_h_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(input_c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(params_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_h_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_backprop_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_h_backprop_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(output_c_backprop_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(reserve_space_tensor, dtype, data, offset, size);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto input_h_op = tensorflow::ops::Const(root, input_h_tensor);
        auto input_c_op = tensorflow::ops::Const(root, input_c_tensor);
        auto params_op = tensorflow::ops::Const(root, params_tensor);
        auto output_op = tensorflow::ops::Const(root, output_tensor);
        auto output_h_op = tensorflow::ops::Const(root, output_h_tensor);
        auto output_c_op = tensorflow::ops::Const(root, output_c_tensor);
        auto output_backprop_op = tensorflow::ops::Const(root, output_backprop_tensor);
        auto output_h_backprop_op = tensorflow::ops::Const(root, output_h_backprop_tensor);
        auto output_c_backprop_op = tensorflow::ops::Const(root, output_c_backprop_tensor);
        auto reserve_space_op = tensorflow::ops::Const(root, reserve_space_tensor);

        tensorflow::Node* node;
        tensorflow::NodeBuilder builder("cudnn_rnn_backprop", "CudnnRNNBackprop");
        builder.Input(input_op.node())
               .Input(input_h_op.node())
               .Input(input_c_op.node())
               .Input(params_op.node())
               .Input(output_op.node())
               .Input(output_h_op.node())
               .Input(output_c_op.node())
               .Input(output_backprop_op.node())
               .Input(output_h_backprop_op.node())
               .Input(output_c_backprop_op.node())
               .Input(reserve_space_op.node())
               .Attr("rnn_mode", rnn_mode)
               .Attr("input_mode", input_mode)
               .Attr("direction", direction)
               .Attr("dropout", dropout)
               .Attr("seed", seed)
               .Attr("seed2", seed2)
               .Attr("T", dtype);

        tensorflow::Status build_status = builder.Finalize(root.graph(), &node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({tensorflow::Output(node, 0), 
                                                 tensorflow::Output(node, 1),
                                                 tensorflow::Output(node, 2),
                                                 tensorflow::Output(node, 3)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
