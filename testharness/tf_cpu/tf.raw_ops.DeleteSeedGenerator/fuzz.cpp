#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
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
    if (size < 10) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType handle_dtype = tensorflow::DT_RESOURCE;
        uint8_t handle_rank = parseRank(data[offset++]);
        std::vector<int64_t> handle_shape = parseShape(data, offset, size, handle_rank);
        
        tensorflow::Tensor handle_tensor(handle_dtype, tensorflow::TensorShape(handle_shape));
        
        tensorflow::DataType deleter_dtype = tensorflow::DT_VARIANT;
        uint8_t deleter_rank = parseRank(data[offset++]);
        std::vector<int64_t> deleter_shape = parseShape(data, offset, size, deleter_rank);
        
        tensorflow::Tensor deleter_tensor(deleter_dtype, tensorflow::TensorShape(deleter_shape));

        auto handle_input = tensorflow::ops::Placeholder(root, handle_dtype, 
            tensorflow::ops::Placeholder::Shape(tensorflow::TensorShape(handle_shape)));
        auto deleter_input = tensorflow::ops::Placeholder(root, deleter_dtype,
            tensorflow::ops::Placeholder::Shape(tensorflow::TensorShape(deleter_shape)));

        // Use raw operation instead of ops namespace
        tensorflow::Output delete_op = tensorflow::Operation(
            root.WithOpName("DeleteSeedGenerator"),
            "DeleteSeedGenerator",
            {handle_input, deleter_input});

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
            {handle_input.node()->name(), handle_tensor},
            {deleter_input.node()->name(), deleter_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(inputs, {delete_op.node()->name()}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}