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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 2) {  
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_RESOURCE;
            break;
        default:
            dtype = tensorflow::DT_RESOURCE;
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
        uint8_t reader_dtype_selector = data[offset++];
        uint8_t reader_rank_byte = data[offset++];
        uint8_t queue_dtype_selector = data[offset++];
        uint8_t queue_rank_byte = data[offset++];

        tensorflow::DataType reader_dtype = parseDataType(reader_dtype_selector);
        uint8_t reader_rank = parseRank(reader_rank_byte);
        std::vector<int64_t> reader_shape = parseShape(data, offset, size, reader_rank);

        tensorflow::DataType queue_dtype = parseDataType(queue_dtype_selector);
        uint8_t queue_rank = parseRank(queue_rank_byte);
        std::vector<int64_t> queue_shape = parseShape(data, offset, size, queue_rank);

        tensorflow::TensorShape reader_tensor_shape;
        for (int64_t dim : reader_shape) {
            reader_tensor_shape.AddDim(dim);
        }

        tensorflow::TensorShape queue_tensor_shape;
        for (int64_t dim : queue_shape) {
            queue_tensor_shape.AddDim(dim);
        }

        tensorflow::Tensor reader_handle_tensor(reader_dtype, reader_tensor_shape);
        tensorflow::Tensor queue_handle_tensor(queue_dtype, queue_tensor_shape);

        auto reader_handle = tensorflow::ops::Placeholder(root.WithOpName("reader_handle"), reader_dtype);
        auto queue_handle = tensorflow::ops::Placeholder(root.WithOpName("queue_handle"), queue_dtype);

        // Use raw operation instead of ops::ReaderReadV2
        auto reader_read_v2 = tensorflow::ops::Operation(
            root.WithOpName("reader_read_v2"),
            "ReaderReadV2",
            {reader_handle, queue_handle}
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{reader_handle, reader_handle_tensor}, {queue_handle, queue_handle_tensor}}, 
                                               {tensorflow::Output(reader_read_v2, 0), tensorflow::Output(reader_read_v2, 1)}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
