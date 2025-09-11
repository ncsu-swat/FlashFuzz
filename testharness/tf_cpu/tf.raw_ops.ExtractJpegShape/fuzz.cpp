#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 1000

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
}

tensorflow::DataType parseOutputType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_INT32;
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
            size_t string_length = std::min(static_cast<size_t>(256), total_size - offset);
            if (offset + 1 <= total_size) {
                string_length = std::min(string_length, static_cast<size_t>(data[offset]));
                offset++;
            }
            
            if (offset + string_length <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), string_length);
                flat(i) = str;
                offset += string_length;
            } else {
                flat(i) = "";
            }
        } else {
            flat(i) = "";
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType output_type = parseOutputType(data[offset++]);
        
        uint8_t contents_rank = parseRank(data[offset++]);
        std::vector<int64_t> contents_shape = parseShape(data, offset, size, contents_rank);
        
        tensorflow::TensorShape contents_tensor_shape;
        for (int64_t dim : contents_shape) {
            contents_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor contents_tensor(tensorflow::DT_STRING, contents_tensor_shape);
        fillStringTensor(contents_tensor, data, offset, size);
        
        auto contents_input = tensorflow::ops::Const(root, contents_tensor);
        
        tensorflow::ops::ExtractJpegShape::Attrs attrs;
        attrs = attrs.OutputType(output_type);
        
        auto extract_jpeg_shape_op = tensorflow::ops::ExtractJpegShape(root, contents_input, attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({extract_jpeg_shape_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
