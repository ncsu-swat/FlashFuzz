#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
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
    std::vector<std::string> device_names;
    
    if (offset >= total_size) {
        device_names.push_back("/cpu:0");
        return device_names;
    }
    
    uint8_t num_devices = data[offset] % 5 + 1;
    offset++;
    
    for (uint8_t i = 0; i < num_devices; ++i) {
        if (offset >= total_size) {
            device_names.push_back("/cpu:0");
            continue;
        }
        
        uint8_t name_length = data[offset] % 20 + 1;
        offset++;
        
        std::string device_name = "/device:";
        for (uint8_t j = 0; j < name_length && offset < total_size; ++j) {
            char c = static_cast<char>((data[offset] % 26) + 'a');
            device_name += c;
            offset++;
        }
        device_names.push_back(device_name);
    }
    
    if (device_names.empty()) {
        device_names.push_back("/cpu:0");
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
        
        std::cout << "Device names: ";
        for (const auto& name : device_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        // Create a constant tensor with device names
        tensorflow::Tensor device_names_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({static_cast<int64_t>(device_names.size())}));
        auto device_names_flat = device_names_tensor.flat<tensorflow::tstring>();
        for (size_t i = 0; i < device_names.size(); ++i) {
            device_names_flat(i) = device_names[i];
        }
        
        auto device_names_const = tensorflow::ops::Const(root, device_names_tensor);
        
        // Create the DeviceIndex op using raw_ops
        tensorflow::Output device_index_op = tensorflow::ops::_internal::DeviceIndex(root, device_names_const);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({device_index_op}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        if (!outputs.empty()) {
            std::cout << "DeviceIndex output: " << outputs[0].scalar<int32_t>()() << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
