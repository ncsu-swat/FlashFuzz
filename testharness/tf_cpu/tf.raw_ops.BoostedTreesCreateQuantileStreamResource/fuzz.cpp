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
    switch (selector % 3) {  
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
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
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor quantile_stream_resource_handle(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        if (offset >= size) return 0;
        uint8_t epsilon_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        std::vector<int64_t> epsilon_shape = parseShape(data, offset, size, epsilon_rank);
        tensorflow::Tensor epsilon_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(epsilon_shape));
        fillTensorWithDataByType(epsilon_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        if (offset >= size) return 0;
        uint8_t num_streams_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        std::vector<int64_t> num_streams_shape = parseShape(data, offset, size, num_streams_rank);
        tensorflow::Tensor num_streams_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(num_streams_shape));
        fillTensorWithDataByType(num_streams_tensor, tensorflow::DT_INT64, data, offset, size);

        int64_t max_elements = 1099511627776;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&max_elements, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            max_elements = std::abs(max_elements) % 1000000 + 1;
        }

        auto quantile_stream_resource_input = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto epsilon_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto num_streams_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);

        // Use raw ops instead of the missing boosted_trees_ops.h
        std::vector<tensorflow::Output> op_outputs;
        tensorflow::Status status = tensorflow::ops::internal::CreateQuantileStreamResource(
            root.WithOpName("BoostedTreesCreateQuantileStreamResource"),
            quantile_stream_resource_input,
            epsilon_input,
            num_streams_input,
            max_elements,
            &op_outputs
        );

        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
            {quantile_stream_resource_input.node()->name(), quantile_stream_resource_handle},
            {epsilon_input.node()->name(), epsilon_tensor},
            {num_streams_input.node()->name(), num_streams_tensor}
        };

        status = session.Run(feed_dict, {}, {"BoostedTreesCreateQuantileStreamResource"}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}