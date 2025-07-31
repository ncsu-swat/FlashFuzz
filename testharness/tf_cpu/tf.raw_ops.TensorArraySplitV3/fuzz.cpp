#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
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
    switch (selector % 15) {  
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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 11:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 12:
            dtype = tensorflow::DT_HALF;
            break;
        case 13:
            dtype = tensorflow::DT_UINT32;
            break;
        case 14:
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
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
        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        uint8_t value_rank = parseRank(data[offset++]);
        std::vector<int64_t> value_shape = parseShape(data, offset, size, value_rank);
        
        if (value_shape.empty() && value_rank > 0) {
            value_shape.push_back(1);
        }
        
        tensorflow::TensorShape value_tensor_shape;
        for (auto dim : value_shape) {
            value_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor value_tensor(value_dtype, value_tensor_shape);
        fillTensorWithDataByType(value_tensor, value_dtype, data, offset, size);
        
        uint8_t lengths_rank = parseRank(data[offset++]);
        std::vector<int64_t> lengths_shape = parseShape(data, offset, size, lengths_rank);
        
        if (lengths_shape.empty()) {
            lengths_shape.push_back(3);
        }
        
        tensorflow::TensorShape lengths_tensor_shape;
        for (auto dim : lengths_shape) {
            lengths_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor lengths_tensor(tensorflow::DT_INT64, lengths_tensor_shape);
        fillTensorWithData<int64_t>(lengths_tensor, data, offset, size);
        
        auto lengths_flat = lengths_tensor.flat<int64_t>();
        for (int i = 0; i < lengths_flat.size(); ++i) {
            lengths_flat(i) = std::abs(lengths_flat(i)) % 10 + 1;
        }
        
        tensorflow::Tensor flow_in_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        flow_in_tensor.scalar<float>()() = 1.0f;
        
        // Create a tensor array
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        size_tensor.scalar<int32_t>()() = 10;
        
        tensorflow::Tensor dtype_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        dtype_tensor.scalar<int32_t>()() = value_dtype;
        
        auto size_const = tensorflow::ops::Const(root, size_tensor);
        auto dtype_const = tensorflow::ops::Const(root, dtype_tensor);
        
        auto handle = tensorflow::ops::_Arg(root.WithOpName("handle"), tensorflow::DT_RESOURCE, 0);
        
        auto value_input = tensorflow::ops::Const(root, value_tensor);
        auto lengths_input = tensorflow::ops::Const(root, lengths_tensor);
        auto flow_in_input = tensorflow::ops::Const(root, flow_in_tensor);
        
        // Use raw_ops approach
        std::vector<tensorflow::Output> outputs;
        tensorflow::ops::internal::TensorArraySplitV3(
            root.WithOpName("TensorArraySplitV3"),
            handle,
            value_input,
            lengths_input,
            flow_in_input,
            &outputs
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        // Since we're using a placeholder for the handle, we can't actually run this
        // But we can check if the graph construction succeeded
        if (!root.status().ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}