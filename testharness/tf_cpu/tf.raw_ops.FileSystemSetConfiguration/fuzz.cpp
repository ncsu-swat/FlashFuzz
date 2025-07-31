#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
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
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 1) {  
        case 0:
            dtype = tensorflow::DT_STRING;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t str_len = std::min(static_cast<size_t>(32), total_size - offset);
            if (str_len > 0) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("");
            }
        } else {
            flat(i) = tensorflow::tstring("");
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType scheme_dtype = parseDataType(data[offset++]);
        uint8_t scheme_rank = parseRank(data[offset++]);
        std::vector<int64_t> scheme_shape = parseShape(data, offset, size, scheme_rank);
        
        tensorflow::Tensor scheme_tensor(scheme_dtype, tensorflow::TensorShape(scheme_shape));
        fillStringTensor(scheme_tensor, data, offset, size);
        
        tensorflow::DataType key_dtype = parseDataType(data[offset++]);
        uint8_t key_rank = parseRank(data[offset++]);
        std::vector<int64_t> key_shape = parseShape(data, offset, size, key_rank);
        
        tensorflow::Tensor key_tensor(key_dtype, tensorflow::TensorShape(key_shape));
        fillStringTensor(key_tensor, data, offset, size);
        
        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        uint8_t value_rank = parseRank(data[offset++]);
        std::vector<int64_t> value_shape = parseShape(data, offset, size, value_rank);
        
        tensorflow::Tensor value_tensor(value_dtype, tensorflow::TensorShape(value_shape));
        fillStringTensor(value_tensor, data, offset, size);

        auto scheme_input = tensorflow::ops::Const(root, scheme_tensor);
        auto key_input = tensorflow::ops::Const(root, key_tensor);
        auto value_input = tensorflow::ops::Const(root, value_tensor);

        // Use raw op directly since FileSystemSetConfiguration is not available in the C++ API
        auto file_system_set_config = tensorflow::Operation(
            root.WithOpName("FileSystemSetConfiguration")
                .WithAttr("scheme", scheme_input)
                .WithAttr("key", key_input)
                .WithAttr("value", value_input)
                .WithDevice("/cpu:0")
                .node()->name());

        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({file_system_set_config}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}