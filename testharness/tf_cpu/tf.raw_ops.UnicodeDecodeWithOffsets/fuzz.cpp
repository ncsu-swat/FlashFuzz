#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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

tensorflow::DataType parseTsplitsDataType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_INT64;
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
            size_t str_len = std::min(static_cast<size_t>(20), total_size - offset);
            if (str_len > 0) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = str;
                offset += str_len;
            } else {
                flat(i) = "test";
            }
        } else {
            flat(i) = "test";
        }
    }
}

std::string parseInputEncoding(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "UTF-8";
        case 1:
            return "UTF-16";
        case 2:
            return "US-ASCII";
        default:
            return "UTF-8";
    }
}

std::string parseErrors(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "strict";
        case 1:
            return "replace";
        case 2:
            return "ignore";
        default:
            return "replace";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
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
            replacement_char = std::abs(replacement_char) % 1114112;
        }
        
        bool replace_control_characters = false;
        if (offset < size) {
            replace_control_characters = (data[offset] % 2) == 1;
            offset++;
        }
        
        tensorflow::DataType tsplits_dtype = parseTsplitsDataType(data[offset % size]);
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        // Use raw_ops namespace for UnicodeDecodeWithOffsets
        auto unicode_decode = tensorflow::ops::internal::UnicodeDecodeWithOffsets(
            root.WithOpName("UnicodeDecodeWithOffsets"),
            input_placeholder,
            input_encoding,
            errors,
            replacement_char,
            replace_control_characters,
            tsplits_dtype
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor}},
            {unicode_decode.row_splits, unicode_decode.char_values, unicode_decode.char_to_byte_starts},
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
