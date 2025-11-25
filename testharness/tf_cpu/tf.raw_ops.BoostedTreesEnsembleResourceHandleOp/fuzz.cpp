#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "";
    }
    
    uint8_t length = data[offset] % 32;
    offset++;
    
    std::string result;
    for (uint8_t i = 0; i < length && offset < total_size; ++i) {
        char c = static_cast<char>(data[offset] % 95 + 32);
        result += c;
        offset++;
    }
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 2) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string container = parseString(data, offset, size);
        std::string shared_name = parseString(data, offset, size);
        
        std::cout << "Container: '" << container << "'" << std::endl;
        std::cout << "Shared name: '" << shared_name << "'" << std::endl;
        
        tensorflow::Node* op_node = nullptr;
        auto builder = tensorflow::NodeBuilder(
                           root.GetUniqueNameForOp("BoostedTreesEnsembleResourceHandleOp"),
                           "BoostedTreesEnsembleResourceHandleOp")
                           .Attr("container", container)
                           .Attr("shared_name", shared_name);
        root.UpdateStatus(builder.Finalize(root.graph(), &op_node));
        if (!root.ok() || op_node == nullptr) {
            return 0;
        }

        tensorflow::Output op(op_node, 0);
        
        std::cout << "Operation created successfully" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({op}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        std::cout << "Session run successfully, outputs size: " << outputs.size() << std::endl;
        
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
