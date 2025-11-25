#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

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

std::vector<std::string> parseDeviceNames(const uint8_t* data, size_t& offset, size_t total_size) {
    static const std::vector<std::string> kDeviceTypes = {"CPU", "GPU", "TPU", "XLA_CPU", "XLA_GPU"};
    std::vector<std::string> device_names;

    if (offset < total_size) {
        uint8_t num_devices = (data[offset++] % 5) + 1;
        device_names.reserve(num_devices);
        for (uint8_t i = 0; i < num_devices && offset < total_size; ++i) {
            device_names.push_back(kDeviceTypes[data[offset++] % kDeviceTypes.size()]);
        }
    }

    if (device_names.empty()) {
        device_names.push_back("CPU");
    } else if (std::find(device_names.begin(), device_names.end(), "CPU") == device_names.end()) {
        device_names.push_back("CPU");
    }

    return device_names;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::vector<std::string> device_names = parseDeviceNames(data, offset, size);

        tensorflow::Node* device_index_node = nullptr;
        tensorflow::Status status = tensorflow::NodeBuilder(
                                        root.GetUniqueNameForOp("DeviceIndex"),
                                        "DeviceIndex")
                                        .Attr("device_names", device_names)
                                        .Device("/cpu:0")
                                        .Finalize(root.graph(), &device_index_node);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("NodeBuilder failed: " + status.ToString(), data, size);
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        status = session.Run({tensorflow::Output(device_index_node, 0)}, &outputs);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Session run failed: " + status.ToString(), data, size);
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
