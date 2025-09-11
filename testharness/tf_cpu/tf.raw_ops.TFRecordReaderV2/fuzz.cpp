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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size, size_t max_length = 100) {
    if (offset >= total_size) {
        return "";
    }
    
    size_t length = data[offset] % max_length;
    offset++;
    
    std::string result;
    for (size_t i = 0; i < length && offset < total_size; ++i) {
        result += static_cast<char>(data[offset]);
        offset++;
    }
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string container = parseString(data, offset, size, 20);
        std::string shared_name = parseString(data, offset, size, 20);
        std::string compression_type = parseString(data, offset, size, 20);
        
        std::cout << "Container: " << container << std::endl;
        std::cout << "Shared name: " << shared_name << std::endl;
        std::cout << "Compression type: " << compression_type << std::endl;
        
        auto reader_op = tensorflow::ops::TFRecordReader(
            root,
            tensorflow::ops::TFRecordReader::Container(container)
                .SharedName(shared_name)
                .CompressionType(compression_type)
        );
        
        std::cout << "TFRecordReader operation created successfully" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({reader_op}, &outputs);
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
            std::cout << "Output tensor type: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
