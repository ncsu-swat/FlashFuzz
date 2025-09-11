#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <string>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    return tensorflow::DT_STRING;
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
            size_t string_length = std::min(static_cast<size_t>(data[offset] % 20 + 1), total_size - offset - 1);
            offset++;
            
            if (offset + string_length <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), string_length);
                flat(i) = tensorflow::tstring(str);
                offset += string_length;
            } else {
                flat(i) = tensorflow::tstring("test");
                offset = total_size;
            }
        } else {
            flat(i) = tensorflow::tstring("default");
        }
    }
}

std::string parseInputEncoding(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "UTF-8";
        case 1: return "UTF-16";
        case 2: return "US-ASCII";
        default: return "UTF-8";
    }
}

std::string parseErrors(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "strict";
        case 1: return "replace";
        case 2: return "ignore";
        default: return "replace";
    }
}

tensorflow::DataType parseTsplits(uint8_t selector) {
    switch (selector % 2) {
        case 0: return tensorflow::DT_INT32;
        case 1: return tensorflow::DT_INT64;
        default: return tensorflow::DT_INT64;
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
        
        tensorflow::Tensor input_tensor(tensorflow::DT_STRING, tensor_shape);
        fillStringTensor(input_tensor, data, offset, size);
        
        std::string input_encoding = parseInputEncoding(data[offset % size]);
        offset++;
        std::string errors = parseErrors(data[offset % size]);
        offset++;
        
        int32_t replacement_char = 65533;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&replacement_char, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            replacement_char = std::abs(replacement_char) % 1114111;
        }
        
        bool replace_control_characters = false;
        if (offset < size) {
            replace_control_characters = (data[offset] % 2) == 1;
            offset++;
        }
        
        tensorflow::DataType tsplits = parseTsplits(data[offset % size]);
        
        auto input = tensorflow::ops::Const(root, input_tensor);
        auto input_encoding_op = tensorflow::ops::Const(root, input_encoding);
        auto errors_op = tensorflow::ops::Const(root, errors);
        auto replacement_char_op = tensorflow::ops::Const(root, replacement_char);
        auto replace_control_characters_op = tensorflow::ops::Const(root, replace_control_characters);
        
        std::vector<tensorflow::Output> outputs;
        
        tensorflow::NodeBuilder builder = tensorflow::NodeBuilder("UnicodeDecode", "UnicodeDecode")
            .Input(input)
            .Input(input_encoding_op)
            .Attr("errors", errors)
            .Attr("replacement_char", replacement_char)
            .Attr("replace_control_characters", replace_control_characters)
            .Attr("Tsplits", tsplits);
        
        tensorflow::Node* node;
        tensorflow::Status status = builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        status = session.Run({tensorflow::Output(node, 0), tensorflow::Output(node, 1)}, &output_tensors);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
