#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        int32_t num_layers_val = static_cast<int32_t>(data[offset++] % 5 + 1);
        
        if (offset >= size) return 0;
        int32_t num_units_val = static_cast<int32_t>(data[offset++] % 10 + 1);
        
        if (offset >= size) return 0;
        int32_t input_size_val = static_cast<int32_t>(data[offset++] % 10 + 1);

        tensorflow::Tensor num_layers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_layers_tensor.scalar<int32_t>()() = num_layers_val;
        
        tensorflow::Tensor num_units_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_units_tensor.scalar<int32_t>()() = num_units_val;
        
        tensorflow::Tensor input_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        input_size_tensor.scalar<int32_t>()() = input_size_val;

        auto num_layers_op = tensorflow::ops::Const(root, num_layers_tensor);
        auto num_units_op = tensorflow::ops::Const(root, num_units_tensor);
        auto input_size_op = tensorflow::ops::Const(root, input_size_tensor);

        if (offset >= size) return 0;
        tensorflow::DataType weights_dtype = parseDataType(data[offset++]);

        if (offset >= size) return 0;
        int num_weight_tensors = data[offset++] % 5 + 1;
        
        std::vector<tensorflow::ops::Const> weight_ops;
        for (int i = 0; i < num_weight_tensors; ++i) {
            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor weight_tensor(weights_dtype, tensor_shape);
            fillTensorWithDataByType(weight_tensor, weights_dtype, data, offset, size);
            
            weight_ops.push_back(tensorflow::ops::Const(root, weight_tensor));
        }

        std::vector<tensorflow::ops::Const> bias_ops;
        for (int i = 0; i < num_weight_tensors; ++i) {
            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor bias_tensor(weights_dtype, tensor_shape);
            fillTensorWithDataByType(bias_tensor, weights_dtype, data, offset, size);
            
            bias_ops.push_back(tensorflow::ops::Const(root, bias_tensor));
        }

        if (offset >= size) return 0;
        std::string rnn_mode = parseRnnMode(data[offset++]);
        
        if (offset >= size) return 0;
        std::string input_mode = parseInputMode(data[offset++]);
        
        if (offset >= size) return 0;
        std::string direction = parseDirection(data[offset++]);

        float dropout = 0.0f;
        if (offset < size) {
            dropout = static_cast<float>(data[offset++]) / 255.0f;
        }

        int seed = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }

        int seed2 = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }

        std::vector<tensorflow::Output> weight_outputs;
        for (const auto& weight_op : weight_ops) {
            weight_outputs.push_back(weight_op);
        }

        std::vector<tensorflow::Output> bias_outputs;
        for (const auto& bias_op : bias_ops) {
            bias_outputs.push_back(bias_op);
        }

        // Use raw_ops API instead of the missing cudnn_rnn_ops.h
        tensorflow::NodeDef node_def;
        node_def.set_op("CudnnRNNCanonicalToParams");
        node_def.set_name("cudnn_rnn_canonical_to_params");
        
        // Add inputs to NodeDef
        *node_def.add_input() = num_layers_op.node()->name();
        *node_def.add_input() = num_units_op.node()->name();
        *node_def.add_input() = input_size_op.node()->name();
        
        for (const auto& weight : weight_outputs) {
            *node_def.add_input() = weight.node()->name();
        }
        
        for (const auto& bias : bias_outputs) {
            *node_def.add_input() = bias.node()->name();
        }
        
        // Add attributes to NodeDef
        auto attr_map = node_def.mutable_attr();
        (*attr_map)["T"].set_type(weights_dtype);
        (*attr_map)["num_params"].set_i(weight_outputs.size() + bias_outputs.size());
        (*attr_map)["rnn_mode"].set_s(rnn_mode);
        (*attr_map)["input_mode"].set_s(input_mode);
        (*attr_map)["direction"].set_s(direction);
        (*attr_map)["dropout"].set_f(dropout);
        (*attr_map)["seed"].set_i(seed);
        (*attr_map)["seed2"].set_i(seed2);
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}