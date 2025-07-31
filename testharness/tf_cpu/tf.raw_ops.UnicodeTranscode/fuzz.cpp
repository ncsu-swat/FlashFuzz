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

std::string parseInputEncoding(uint8_t selector) {
    std::vector<std::string> encodings = {"UTF-8", "UTF-16", "UTF-16-BE", "UTF-16-LE", "UTF-32", "UTF-32-BE", "UTF-32-LE", "US-ASCII", "ISO-8859-1"};
    return encodings[selector % encodings.size()];
}

std::string parseOutputEncoding(uint8_t selector) {
    std::vector<std::string> encodings = {"UTF-8", "UTF-16-BE", "UTF-32-BE"};
    return encodings[selector % encodings.size()];
}

std::string parseErrors(uint8_t selector) {
    std::vector<std::string> errors = {"strict", "replace", "ignore"};
    return errors[selector % errors.size()];
}

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 20 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                char c = static_cast<char>(data[offset] % 128);
                if (c == 0) c = 'A';
                str += c;
                offset++;
            }
            
            if (str.empty()) {
                str = "Hello";
            }
            
            flat(i) = str;
        } else {
            flat(i) = "Default";
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
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
        
        if (offset >= size) return 0;
        
        std::string input_encoding = parseInputEncoding(data[offset++]);
        if (offset >= size) return 0;
        
        std::string output_encoding = parseOutputEncoding(data[offset++]);
        if (offset >= size) return 0;
        
        std::string errors = parseErrors(data[offset++]);
        if (offset >= size) return 0;
        
        int64_t replacement_char = 65533;
        if (offset + sizeof(int32_t) <= size) {
            int32_t temp_char;
            std::memcpy(&temp_char, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            replacement_char = std::abs(temp_char) % 1114111;
            if (replacement_char == 0) replacement_char = 65533;
        }
        
        bool replace_control_characters = false;
        if (offset < size) {
            replace_control_characters = (data[offset++] % 2) == 1;
        }
        
        auto input_node = tensorflow::ops::Const(root, input_tensor);
        
        auto unicode_transcode = tensorflow::ops::UnicodeTranscode(
            root,
            input_node,
            input_encoding,
            output_encoding,
            tensorflow::ops::UnicodeTranscode::Attrs()
                .Errors(errors)
                .ReplacementChar(replacement_char)
                .ReplaceControlCharacters(replace_control_characters)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({unicode_transcode}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}