#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType weights_dtype = parseDataType(data[offset++]);
        
        tensorflow::Tensor num_layers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_layers_tensor.scalar<int32_t>()() = (data[offset++] % 4) + 1;
        
        tensorflow::Tensor num_units_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_units_tensor.scalar<int32_t>()() = (data[offset++] % 64) + 1;
        
        tensorflow::Tensor input_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        input_size_tensor.scalar<int32_t>()() = (data[offset++] % 64) + 1;
        
        int32_t num_layers = num_layers_tensor.scalar<int32_t>()();
        int32_t num_units = num_units_tensor.scalar<int32_t>()();
        int32_t input_size = input_size_tensor.scalar<int32_t>()();
        
        std::string rnn_mode;
        switch (data[offset++] % 4) {
            case 0: rnn_mode = "rnn_relu"; break;
            case 1: rnn_mode = "rnn_tanh"; break;
            case 2: rnn_mode = "lstm"; break;
            case 3: rnn_mode = "gru"; break;
        }
        
        std::string input_mode;
        switch (data[offset++] % 3) {
            case 0: input_mode = "linear_input"; break;
            case 1: input_mode = "skip_input"; break;
            case 2: input_mode = "auto_select"; break;
        }
        
        std::string direction;
        switch (data[offset++] % 2) {
            case 0: direction = "unidirectional"; break;
            case 1: direction = "bidirectional"; break;
        }
        
        float dropout = 0.0f;
        int seed = 0;
        int seed2 = 0;
        int num_proj = 0;
        
        if (offset < size) {
            dropout = static_cast<float>(data[offset++]) / 255.0f;
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        if (offset < size) {
            num_proj = data[offset++] % 32;
        }
        
        int dir_count = (direction == "bidirectional") ? 2 : 1;
        
        std::vector<tensorflow::Output> weights;
        std::vector<tensorflow::Output> biases;
        
        int num_weight_matrices = 0;
        int num_bias_vectors = 0;
        
        if (rnn_mode == "lstm") {
            num_weight_matrices = 8 * num_layers * dir_count;
            num_bias_vectors = 8 * num_layers * dir_count;
        } else if (rnn_mode == "gru") {
            num_weight_matrices = 6 * num_layers * dir_count;
            num_bias_vectors = 6 * num_layers * dir_count;
        } else {
            num_weight_matrices = 2 * num_layers * dir_count;
            num_bias_vectors = 2 * num_layers * dir_count;
        }
        
        for (int i = 0; i < num_weight_matrices && i < 16; ++i) {
            int64_t rows = (i % 2 == 0) ? num_units : input_size;
            int64_t cols = num_units;
            
            tensorflow::TensorShape weight_shape({rows, cols});
            tensorflow::Tensor weight_tensor(weights_dtype, weight_shape);
            fillTensorWithDataByType(weight_tensor, weights_dtype, data, offset, size);
            
            auto weight_const = tensorflow::ops::Const(root, weight_tensor);
            weights.push_back(weight_const);
        }
        
        for (int i = 0; i < num_bias_vectors && i < 16; ++i) {
            tensorflow::TensorShape bias_shape({num_units});
            tensorflow::Tensor bias_tensor(weights_dtype, bias_shape);
            fillTensorWithDataByType(bias_tensor, weights_dtype, data, offset, size);
            
            auto bias_const = tensorflow::ops::Const(root, bias_tensor);
            biases.push_back(bias_const);
        }
        
        if (weights.empty()) {
            tensorflow::TensorShape weight_shape({input_size, num_units});
            tensorflow::Tensor weight_tensor(weights_dtype, weight_shape);
            fillTensorWithDataByType(weight_tensor, weights_dtype, data, offset, size);
            auto weight_const = tensorflow::ops::Const(root, weight_tensor);
            weights.push_back(weight_const);
        }
        
        if (biases.empty()) {
            tensorflow::TensorShape bias_shape({num_units});
            tensorflow::Tensor bias_tensor(weights_dtype, bias_shape);
            fillTensorWithDataByType(bias_tensor, weights_dtype, data, offset, size);
            auto bias_const = tensorflow::ops::Const(root, bias_tensor);
            biases.push_back(bias_const);
        }
        
        auto num_layers_const = tensorflow::ops::Const(root, num_layers_tensor);
        auto num_units_const = tensorflow::ops::Const(root, num_units_tensor);
        auto input_size_const = tensorflow::ops::Const(root, input_size_tensor);
        
        std::vector<tensorflow::NodeBuilder::NodeOut> weight_inputs;
        weight_inputs.reserve(weights.size());
        for (const auto& weight : weights) {
            weight_inputs.emplace_back(weight.node(), weight.index());
        }

        std::vector<tensorflow::NodeBuilder::NodeOut> bias_inputs;
        bias_inputs.reserve(biases.size());
        for (const auto& bias : biases) {
            bias_inputs.emplace_back(bias.node(), bias.index());
        }

        tensorflow::Node* node = nullptr;
        auto status = tensorflow::NodeBuilder("cudnn_rnn_canonical_to_params_v2", "CudnnRNNCanonicalToParamsV2")
                          .Input(num_layers_const.node())
                          .Input(num_units_const.node())
                          .Input(input_size_const.node())
                          .Input(weight_inputs)
                          .Input(bias_inputs)
                          .Attr("T", weights_dtype)
                          .Attr("num_params_weights", static_cast<int>(weights.size()))
                          .Attr("num_params_biases", static_cast<int>(biases.size()))
                          .Attr("rnn_mode", rnn_mode)
                          .Attr("input_mode", input_mode)
                          .Attr("direction", direction)
                          .Attr("dropout", dropout)
                          .Attr("seed", seed)
                          .Attr("seed2", seed2)
                          .Attr("num_proj", num_proj)
                          .Finalize(root.graph(), &node);

        if (!status.ok() || node == nullptr) {
            return 0;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = session.Run({tensorflow::Output(node, 0)}, &outputs);
        if (!run_status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
