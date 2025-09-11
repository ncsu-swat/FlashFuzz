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
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType group_key_dtype = tensorflow::DT_INT32;
        uint8_t group_key_rank = 0;
        std::vector<int64_t> group_key_shape = {};
        tensorflow::TensorShape group_key_tensor_shape(group_key_shape);
        tensorflow::Tensor group_key_tensor(group_key_dtype, group_key_tensor_shape);
        fillTensorWithDataByType(group_key_tensor, group_key_dtype, data, offset, size);
        
        tensorflow::DataType rank_dtype = tensorflow::DT_INT32;
        uint8_t rank_rank = 0;
        std::vector<int64_t> rank_shape = {};
        tensorflow::TensorShape rank_tensor_shape(rank_shape);
        tensorflow::Tensor rank_tensor(rank_dtype, rank_tensor_shape);
        fillTensorWithDataByType(rank_tensor, rank_dtype, data, offset, size);
        
        tensorflow::DataType group_size_dtype = tensorflow::DT_INT32;
        uint8_t group_size_rank = 0;
        std::vector<int64_t> group_size_shape = {};
        tensorflow::TensorShape group_size_tensor_shape(group_size_shape);
        tensorflow::Tensor group_size_tensor(group_size_dtype, group_size_tensor_shape);
        fillTensorWithDataByType(group_size_tensor, group_size_dtype, data, offset, size);

        auto group_key_input = tensorflow::ops::Const(root, group_key_tensor);
        auto rank_input = tensorflow::ops::Const(root, rank_tensor);
        auto group_size_input = tensorflow::ops::Const(root, group_size_tensor);

        std::string communication_hint = "auto";
        float timeout_seconds = 0.0f;
        
        if (offset < size) {
            uint8_t hint_selector = data[offset++];
            switch (hint_selector % 3) {
                case 0: communication_hint = "auto"; break;
                case 1: communication_hint = "nccl"; break;
                case 2: communication_hint = "ring"; break;
            }
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&timeout_seconds, data + offset, sizeof(float));
            offset += sizeof(float);
            if (timeout_seconds < 0.0f || timeout_seconds > 3600.0f) {
                timeout_seconds = 0.0f;
            }
        }

        // Use raw_ops directly since we don't have collective_ops.h
        auto attrs = tensorflow::ops::Raw::CollectiveInitializeCommunicator::Attrs()
            .CommunicationHint(communication_hint)
            .TimeoutSeconds(timeout_seconds);
            
        auto collective_init = tensorflow::ops::Raw::CollectiveInitializeCommunicator(
            root, 
            group_key_input, 
            rank_input, 
            group_size_input,
            attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({collective_init}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
