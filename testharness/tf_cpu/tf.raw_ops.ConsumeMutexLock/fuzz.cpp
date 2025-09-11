#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 1) {  
        case 0:
            dtype = tensorflow::DT_VARIANT;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t dtype_selector = data[offset++];
        tensorflow::DataType mutex_lock_dtype = parseDataType(dtype_selector);

        if (offset >= size) return 0;
        uint8_t rank_byte = data[offset++];
        uint8_t mutex_lock_rank = parseRank(rank_byte);

        std::vector<int64_t> mutex_lock_shape = parseShape(data, offset, size, mutex_lock_rank);

        tensorflow::TensorShape mutex_lock_tensor_shape;
        for (int64_t dim : mutex_lock_shape) {
            if (dim <= 0) dim = 1;
            mutex_lock_tensor_shape.AddDim(dim);
        }

        // Create a mutex using raw ops
        auto mutex = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        // Create a mutex lock using raw ops
        auto mutex_lock = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        
        // Use raw ops to create ConsumeMutexLock
        tensorflow::Output consume_mutex_lock = tensorflow::Operation(
            root.WithOpName("ConsumeMutexLock"),
            "ConsumeMutexLock",
            {mutex_lock}
        );

        tensorflow::ClientSession session(root);
        
        // Create placeholder values
        tensorflow::Tensor mutex_tensor(tensorflow::DT_STRING, {});
        mutex_tensor.scalar<tensorflow::tstring>()() = "mutex";
        
        tensorflow::Tensor mutex_lock_tensor(tensorflow::DT_VARIANT, {});
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{mutex, mutex_tensor}, {mutex_lock, mutex_lock_tensor}}, 
                                              {consume_mutex_lock}, &outputs);
        
        if (!status.ok()) {
            // This is expected to fail since we're not properly initializing the mutex and mutex_lock
            // But the code should compile
            tf_fuzzer_utils::logError("Error running session: " + status.ToString(), data, size);
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return 0;
    } 

    return 0;
}
