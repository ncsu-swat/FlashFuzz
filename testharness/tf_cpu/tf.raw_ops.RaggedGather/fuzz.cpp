#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include <iostream>
#include <vector>
#include <cstring>
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

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
    return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t num_splits = (data[offset++] % 3) + 1;
        
        if (offset >= size) return 0;
        tensorflow::DataType splits_dtype = parseIndicesDataType(data[offset++]);
        
        if (offset >= size) return 0;
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t values_rank = parseRank(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t indices_rank = parseRank(data[offset++]);
        
        if (offset >= size) return 0;
        int output_ragged_rank = (data[offset++] % 5);

        std::vector<tensorflow::Output> params_nested_splits;
        std::vector<tensorflow::Tensor> splits_tensors;
        
        for (int i = 0; i < num_splits; i++) {
            if (offset >= size) return 0;
            uint8_t splits_rank = parseRank(data[offset++]);
            if (splits_rank == 0) splits_rank = 1;
            
            std::vector<int64_t> splits_shape = parseShape(data, offset, size, splits_rank);
            
            tensorflow::TensorShape splits_tensor_shape;
            for (auto dim : splits_shape) {
                splits_tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor splits_tensor(splits_dtype, splits_tensor_shape);
            fillTensorWithDataByType(splits_tensor, splits_dtype, data, offset, size);
            
            if (splits_dtype == tensorflow::DT_INT32) {
                auto flat = splits_tensor.flat<int32_t>();
                for (int j = 0; j < flat.size(); j++) {
                    flat(j) = std::abs(flat(j)) % 100;
                    if (j > 0 && flat(j) < flat(j-1)) {
                        flat(j) = flat(j-1);
                    }
                }
            } else {
                auto flat = splits_tensor.flat<int64_t>();
                for (int j = 0; j < flat.size(); j++) {
                    flat(j) = std::abs(flat(j)) % 100;
                    if (j > 0 && flat(j) < flat(j-1)) {
                        flat(j) = flat(j-1);
                    }
                }
            }
            
            splits_tensors.push_back(splits_tensor);
            
            std::string splits_name = "params_nested_splits_" + std::to_string(i);
            params_nested_splits.push_back(tensorflow::ops::Placeholder(root.WithOpName(splits_name), splits_dtype));
        }

        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        tensorflow::TensorShape values_tensor_shape;
        for (auto dim : values_shape) {
            values_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor values_tensor(values_dtype, values_tensor_shape);
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);

        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::TensorShape indices_tensor_shape;
        for (auto dim : indices_shape) {
            indices_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size(); i++) {
                flat(i) = std::abs(flat(i)) % 10;
            }
        } else {
            auto flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < flat.size(); i++) {
                flat(i) = std::abs(flat(i)) % 10;
            }
        }

        auto params_dense_values = tensorflow::ops::Placeholder(root.WithOpName("params_dense_values"), values_dtype);
        auto indices = tensorflow::ops::Placeholder(root.WithOpName("indices"), indices_dtype);

        // Use raw_ops.RaggedGather through the Scope API
        std::vector<tensorflow::Input> nested_splits_inputs;
        for (const auto& split : params_nested_splits) {
            nested_splits_inputs.push_back(split);
        }

        auto ragged_gather_op = tensorflow::ops::RaggedGather(
            root.WithOpName("ragged_gather"),
            nested_splits_inputs,
            params_dense_values,
            indices,
            output_ragged_rank);

        tensorflow::ClientSession session(root);
        
        // Create feed dictionary using unordered_map
        std::unordered_map<tensorflow::Output, tensorflow::Input::Initializer> feed_dict;
        for (int i = 0; i < params_nested_splits.size(); i++) {
            feed_dict.insert({params_nested_splits[i], splits_tensors[i]});
        }
        feed_dict.insert({params_dense_values, values_tensor});
        feed_dict.insert({indices, indices_tensor});

        std::vector<tensorflow::Output> fetch_outputs;
        for (int i = 0; i < output_ragged_rank + 1; i++) {
            fetch_outputs.push_back(ragged_gather_op.output_nested_splits[i]);
        }
        fetch_outputs.push_back(ragged_gather_op.output_dense_values);

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, fetch_outputs, &outputs);
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return 0;
    } 

    return 0;
}
