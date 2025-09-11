#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include <iostream>
#include <cstring>
#include <vector>

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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t string_length = std::min(static_cast<size_t>(32), total_size - offset);
            if (offset + 1 <= total_size) {
                string_length = data[offset] % 32;
                offset++;
            }
            
            if (offset + string_length <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), string_length);
                flat(i) = str;
                offset += string_length;
            } else {
                flat(i) = "{}";
            }
        } else {
            flat(i) = "{}";
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor json_examples_tensor(tensorflow::DT_STRING, tensor_shape);
        fillStringTensor(json_examples_tensor, data, offset, size);
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < tensor_shape.dims(); ++i) {
            std::cout << tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        auto json_examples_input = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        auto decode_op = tensorflow::ops::DecodeJSONExample(root, json_examples_input);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{json_examples_input, json_examples_tensor}}, {decode_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
