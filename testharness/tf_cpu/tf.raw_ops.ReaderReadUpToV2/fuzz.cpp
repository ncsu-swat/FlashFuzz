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
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
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

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
    auto flat = tensor.flat<T>();
    const size_t num_elements = flat.size();
    const size_t element_size = sizeof(T);

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + element_size <= total_size) {
            T value;
            std::memcpy(&value, data + offset, element_size);
            offset += element_size;
            flat(i) = value;
        } else {
            flat(i) = T{};
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_RESOURCE:
            break;
        case tensorflow::DT_STRING:
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor reader_handle_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        tensorflow::Tensor queue_handle_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        uint8_t num_records_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        
        std::vector<int64_t> num_records_shape = parseShape(data, offset, size, num_records_rank);
        if (offset >= size) return 0;
        
        tensorflow::TensorShape num_records_tensor_shape;
        for (auto dim : num_records_shape) {
            num_records_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor num_records_tensor(tensorflow::DT_INT64, num_records_tensor_shape);
        fillTensorWithDataByType(num_records_tensor, tensorflow::DT_INT64, data, offset, size);

        auto reader_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto queue_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto num_records = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);

        // Use raw operation instead of ops namespace
        auto reader_read_up_to_v2_op = tensorflow::Operation(root.WithOpName("ReaderReadUpToV2")
            .WithInput(reader_handle)
            .WithInput(queue_handle)
            .WithInput(num_records));
        
        auto keys = tensorflow::Output(reader_read_up_to_v2_op, 0);
        auto values = tensorflow::Output(reader_read_up_to_v2_op, 1);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{reader_handle, reader_handle_tensor}, 
                                                 {queue_handle, queue_handle_tensor}, 
                                                 {num_records, num_records_tensor}}, 
                                                {keys, values}, 
                                                &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}