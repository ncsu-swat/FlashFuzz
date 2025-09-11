#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <unordered_map>

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
    switch (selector % 11) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
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
        uint8_t cond_rank = parseRank(data[offset++]);
        std::vector<int64_t> cond_shape = parseShape(data, offset, size, cond_rank);
        tensorflow::TensorShape cond_tensor_shape(cond_shape);
        tensorflow::Tensor cond_tensor(tensorflow::DT_BOOL, cond_tensor_shape);
        fillTensorWithDataByType(cond_tensor, tensorflow::DT_BOOL, data, offset, size);
        
        auto cond_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        
        uint8_t num_inputs = (data[offset++] % 3) + 1;
        std::vector<tensorflow::Output> input_placeholders;
        std::vector<tensorflow::Tensor> input_tensors;
        std::vector<tensorflow::DataType> input_types;
        
        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType input_dtype = parseDataType(data[offset++]);
            uint8_t input_rank = parseRank(data[offset++]);
            std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
            
            tensorflow::TensorShape input_tensor_shape(input_shape);
            tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
            fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
            
            auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
            input_placeholders.push_back(input_placeholder);
            input_tensors.push_back(input_tensor);
            input_types.push_back(input_dtype);
        }
        
        if (input_placeholders.empty()) {
            return 0;
        }
        
        auto then_func = [&](const std::vector<tensorflow::Output>& inputs) -> std::vector<tensorflow::Output> {
            std::vector<tensorflow::Output> outputs;
            for (const auto& input : inputs) {
                outputs.push_back(tensorflow::ops::Identity(root, input));
            }
            return outputs;
        };
        
        auto else_func = [&](const std::vector<tensorflow::Output>& inputs) -> std::vector<tensorflow::Output> {
            std::vector<tensorflow::Output> outputs;
            for (const auto& input : inputs) {
                outputs.push_back(tensorflow::ops::Identity(root, input));
            }
            return outputs;
        };
        
        std::vector<tensorflow::Output> then_outputs = then_func(input_placeholders);
        std::vector<tensorflow::Output> else_outputs = else_func(input_placeholders);
        
        tensorflow::ClientSession session(root);
        
        // Convert feed_dict to the correct format for ClientSession::Run
        std::unordered_map<tensorflow::Output, tensorflow::Input::Initializer> feed_dict;
        feed_dict.emplace(cond_placeholder, cond_tensor);
        
        for (size_t i = 0; i < input_placeholders.size(); ++i) {
            feed_dict.emplace(input_placeholders[i], input_tensors[i]);
        }
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, then_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }
        
        status = session.Run(feed_dict, else_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
