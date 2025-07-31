#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
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
    switch (selector % 1) {  
        case 0:
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
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_batch_tensors = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::Output> batch_inputs;
        
        for (uint8_t i = 0; i < num_batch_tensors; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            uint8_t rank = 1;
            
            std::vector<int64_t> shape;
            if (offset + sizeof(int64_t) <= size) {
                int64_t dim_val;
                std::memcpy(&dim_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                        static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) %
                                            static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));
                shape.push_back(dim_val);
            } else {
                shape.push_back(1);
            }
            
            tensorflow::TensorShape tensor_shape(shape);
            tensorflow::Tensor tensor(dtype, tensor_shape);
            
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto placeholder = tensorflow::ops::Placeholder(root, dtype);
            batch_inputs.push_back(placeholder);
        }
        
        if (batch_inputs.empty()) {
            tensorflow::TensorShape tensor_shape({1});
            tensorflow::Tensor tensor(tensorflow::DT_INT32, tensor_shape);
            tensor.flat<int32_t>()(0) = 0;
            auto placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
            batch_inputs.push_back(placeholder);
        }
        
        std::string mode_override_str = "unspecified";
        if (offset < size) {
            uint8_t mode_selector = data[offset++] % 4;
            switch (mode_selector) {
                case 0: mode_override_str = "unspecified"; break;
                case 1: mode_override_str = "inference"; break;
                case 2: mode_override_str = "training"; break;
                case 3: mode_override_str = "backward_pass_only"; break;
            }
        }
        
        tensorflow::TensorShape mode_shape({});
        tensorflow::Tensor mode_tensor(tensorflow::DT_STRING, mode_shape);
        mode_tensor.scalar<tensorflow::tstring>()() = mode_override_str;
        auto mode_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        int device_ordinal = -1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&device_ordinal, data + offset, sizeof(int));
            offset += sizeof(int);
            device_ordinal = device_ordinal % 8;
        }
        
        // Use raw_ops directly instead of the missing tpu_ops.h
        auto enqueue_op = tensorflow::ops::Operation(
            root.WithOpName("EnqueueTPUEmbeddingIntegerBatch"),
            "EnqueueTPUEmbeddingIntegerBatch",
            batch_inputs,
            {mode_placeholder},
            {{"device_ordinal", device_ordinal}}
        );

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        
        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            tensorflow::TensorShape tensor_shape({1});
            tensorflow::Tensor tensor(tensorflow::DT_INT32, tensor_shape);
            tensor.flat<int32_t>()(0) = static_cast<int32_t>(i);
            feed_dict.push_back({batch_inputs[i].node()->name(), tensor});
        }
        
        feed_dict.push_back({mode_placeholder.node()->name(), mode_tensor});
        
        tensorflow::Status status = session.Run(feed_dict, {}, {enqueue_op.node()->name()}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}