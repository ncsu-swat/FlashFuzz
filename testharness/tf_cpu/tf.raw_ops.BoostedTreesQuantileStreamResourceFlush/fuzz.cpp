#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
            dtype = tensorflow::DT_INT64;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType resource_dtype = tensorflow::DT_RESOURCE;
        uint8_t resource_rank = 0;
        std::vector<int64_t> resource_shape = {};
        
        tensorflow::Tensor quantile_stream_resource_handle(resource_dtype, tensorflow::TensorShape(resource_shape));
        
        tensorflow::DataType num_buckets_dtype = tensorflow::DT_INT64;
        uint8_t num_buckets_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        std::vector<int64_t> num_buckets_shape = parseShape(data, offset, size, num_buckets_rank);
        
        tensorflow::Tensor num_buckets_tensor(num_buckets_dtype, tensorflow::TensorShape(num_buckets_shape));
        fillTensorWithDataByType(num_buckets_tensor, num_buckets_dtype, data, offset, size);
        
        bool generate_quantiles = false;
        if (offset < size) {
            generate_quantiles = (data[offset++] % 2) == 1;
        }

        auto quantile_stream_resource_handle_op = tensorflow::ops::Placeholder(root, resource_dtype);
        auto num_buckets_op = tensorflow::ops::Placeholder(root, num_buckets_dtype);

        // Use raw_ops directly instead of BoostedTreesQuantileStreamResourceFlush
        auto flush_op = tensorflow::ops::Operation(
            root.WithOpName("BoostedTreesQuantileStreamResourceFlush"),
            "BoostedTreesQuantileStreamResourceFlush",
            {quantile_stream_resource_handle_op, num_buckets_op},
            {{"generate_quantiles", generate_quantiles}}
        );

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
            {quantile_stream_resource_handle_op.node()->name(), quantile_stream_resource_handle},
            {num_buckets_op.node()->name(), num_buckets_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, {}, {flush_op.node()->name()}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
