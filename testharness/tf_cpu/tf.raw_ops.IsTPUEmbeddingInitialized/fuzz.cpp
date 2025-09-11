#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <cstring>
#include <iostream>
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

std::string parseConfig(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "";
    }
    
    uint8_t config_length = data[offset] % 32;
    offset++;
    
    std::string config;
    for (uint8_t i = 0; i < config_length && offset < total_size; ++i) {
        config += static_cast<char>(data[offset] % 128);
        offset++;
    }
    
    return config;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 1) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string config = parseConfig(data, offset, size);
        
        std::cout << "Config: " << config << std::endl;
        
        // Use raw_ops directly instead of the missing tpu_ops.h
        auto is_tpu_embedding_initialized = tensorflow::ops::Operation(
            root.WithOpName("IsTPUEmbeddingInitialized"),
            "IsTPUEmbeddingInitialized",
            {},  // No input tensors
            {}, // No output types needed as it's automatically inferred
            {{"config", config}} // Attributes
        );
        
        std::cout << "Created IsTPUEmbeddingInitialized operation" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({is_tpu_embedding_initialized}, &outputs);
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
            
            if (outputs[0].dtype() == tensorflow::DT_BOOL) {
                auto flat = outputs[0].flat<bool>();
                if (flat.size() > 0) {
                    std::cout << "Output value: " << (flat(0) ? "true" : "false") << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
