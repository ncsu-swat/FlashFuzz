#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <string>

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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 32 + 1), total_size - offset - 1);
            offset++;
            
            if (offset + str_len <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("test");
                offset = total_size;
            }
        } else {
            flat(i) = tensorflow::tstring("default");
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        uint8_t pattern_rank = parseRank(data[offset++]);
        std::vector<int64_t> pattern_shape = parseShape(data, offset, size, pattern_rank);
        
        uint8_t rewrite_rank = parseRank(data[offset++]);
        std::vector<int64_t> rewrite_shape = parseShape(data, offset, size, rewrite_rank);
        
        bool replace_global = (data[offset++] % 2) == 1;

        tensorflow::TensorShape input_tensor_shape;
        for (auto dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor input_tensor(tensorflow::DT_STRING, input_tensor_shape);
        fillStringTensor(input_tensor, data, offset, size);

        tensorflow::TensorShape pattern_tensor_shape;
        for (auto dim : pattern_shape) {
            pattern_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor pattern_tensor(tensorflow::DT_STRING, pattern_tensor_shape);
        fillStringTensor(pattern_tensor, data, offset, size);

        tensorflow::TensorShape rewrite_tensor_shape;
        for (auto dim : rewrite_shape) {
            rewrite_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor rewrite_tensor(tensorflow::DT_STRING, rewrite_tensor_shape);
        fillStringTensor(rewrite_tensor, data, offset, size);

        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        auto pattern_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        auto rewrite_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);

        auto regex_replace_op = tensorflow::ops::RegexReplace(
            root, 
            input_placeholder, 
            pattern_placeholder, 
            rewrite_placeholder,
            tensorflow::ops::RegexReplace::ReplaceGlobal(replace_global)
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor}, 
             {pattern_placeholder, pattern_tensor}, 
             {rewrite_placeholder, rewrite_tensor}},
            {regex_replace_op}, 
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
