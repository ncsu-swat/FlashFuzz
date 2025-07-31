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
        std::cerr << "Error: " << message << std::endl;
    }
}

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size, size_t max_length = 100) {
    if (offset >= total_size) {
        return "";
    }
    
    size_t length = std::min(max_length, total_size - offset);
    if (length > 0) {
        uint8_t len_byte = data[offset];
        offset++;
        length = std::min(static_cast<size_t>(len_byte % 50), total_size - offset);
    }
    
    std::string result;
    result.reserve(length);
    
    for (size_t i = 0; i < length && offset < total_size; ++i) {
        char c = static_cast<char>(data[offset]);
        if (c >= 32 && c <= 126) {
            result.push_back(c);
        } else {
            result.push_back('a');
        }
        offset++;
    }
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3) {
        return 0;
    }
    
    size_t offset = 0;
    
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");
    
    try {
        std::string container = parseString(data, offset, size, 20);
        std::string shared_name = parseString(data, offset, size, 20);
        
        std::cout << "Container: '" << container << "'" << std::endl;
        std::cout << "Shared name: '" << shared_name << "'" << std::endl;
        
        auto identity_reader = tensorflow::ops::IdentityReader(
            root.WithOpName("identity_reader"),
            tensorflow::ops::IdentityReader::Container(container)
                .SharedName(shared_name)
        );
        
        std::cout << "Created IdentityReader operation" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({identity_reader}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        std::cout << "Session ran successfully, outputs size: " << outputs.size() << std::endl;
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "Output tensor dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
        }
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }
    
    return 0;
}