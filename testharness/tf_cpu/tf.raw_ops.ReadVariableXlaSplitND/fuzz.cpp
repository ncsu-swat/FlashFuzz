#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
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
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_SPLITS 4

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 12) {  
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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
            break;
        case 11:
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::vector<int> parseNumSplits(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    std::vector<int> num_splits;
    num_splits.reserve(rank);
    
    for (uint8_t i = 0; i < rank; ++i) {
        if (offset < total_size) {
            uint8_t split_val = data[offset++];
            int splits = 1 + (split_val % MAX_NUM_SPLITS);
            num_splits.push_back(splits);
        } else {
            num_splits.push_back(2);
        }
    }
    
    return num_splits;
}

std::vector<int> parsePaddings(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    std::vector<int> paddings;
    paddings.reserve(rank);
    
    for (uint8_t i = 0; i < rank; ++i) {
        if (offset < total_size) {
            uint8_t pad_val = data[offset++];
            int padding = pad_val % 4;
            paddings.push_back(padding);
        } else {
            paddings.push_back(0);
        }
    }
    
    return paddings;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        std::vector<int> num_splits = parseNumSplits(data, offset, size, rank);
        std::vector<int> paddings = parsePaddings(data, offset, size, rank);
        
        for (size_t i = 0; i < shape.size(); ++i) {
            int padding = (i < paddings.size()) ? paddings[i] : 0;
            int splits = (i < num_splits.size()) ? num_splits[i] : 2;
            int64_t padded_dim = shape[i] + padding;
            if (padded_dim % splits != 0) {
                shape[i] = ((padded_dim / splits) + 1) * splits - padding;
                if (shape[i] <= 0) shape[i] = splits;
            }
        }
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        auto var = tensorflow::ops::VarHandleOp(root, dtype, tensor_shape);
        auto assign = tensorflow::ops::AssignVariableOp(root, var, input_tensor);
        
        int total_splits = 1;
        for (int splits : num_splits) {
            total_splits *= splits;
        }
        
        // Convert num_splits to a tensor
        tensorflow::Tensor num_splits_tensor(tensorflow::DT_INT32, {static_cast<int64_t>(num_splits.size())});
        auto num_splits_flat = num_splits_tensor.flat<int32_t>();
        for (size_t i = 0; i < num_splits.size(); ++i) {
            num_splits_flat(i) = num_splits[i];
        }
        
        // Convert paddings to a tensor
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, {static_cast<int64_t>(paddings.size())});
        auto paddings_flat = paddings_tensor.flat<int32_t>();
        for (size_t i = 0; i < paddings.size(); ++i) {
            paddings_flat(i) = paddings[i];
        }
        
        // Use raw_ops directly
        auto read_split = tensorflow::ops::_Internal::ReadVariableXlaSplitND(
            root, var, num_splits_tensor, paddings_tensor, dtype);
        
        tensorflow::ClientSession session(root);
        
        // Run the assign operation first
        TF_CHECK_OK(session.Run({assign}, nullptr));
        
        // Then run the read_split operation
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session.Run({read_split.output}, &outputs));

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
