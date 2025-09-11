#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/variant.h"
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
    switch (selector % 21) {  
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
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
        case 17:
            dtype = tensorflow::DT_UINT32;
            break;
        case 18:
            dtype = tensorflow::DT_UINT64;
            break;
        case 19:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 20:
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
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10 + 1;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset] % 128);
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "";
                }
            }
            break;
        }
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
        if (offset >= size) return 0;
        uint8_t handles_rank = parseRank(data[offset++]);
        
        if (offset >= size) return 0;
        std::vector<int64_t> handles_shape = parseShape(data, offset, size, handles_rank);
        
        tensorflow::TensorShape handles_tensor_shape;
        for (auto dim : handles_shape) {
            handles_tensor_shape.AddDim(dim);
        }
        
        // Create a tensor list first
        tensorflow::DataType element_dtype = parseDataType(data[offset++]);
        if (offset >= size) return 0;
        
        // Create a tensor for the element shape
        uint8_t element_shape_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        
        std::vector<int64_t> element_shape_dims = parseShape(data, offset, size, element_shape_rank);
        tensorflow::TensorShape element_shape_tensor_shape;
        element_shape_tensor_shape.AddDim(element_shape_dims.size());
        
        tensorflow::Tensor element_shape_tensor(tensorflow::DT_INT32, element_shape_tensor_shape);
        auto element_shape_flat = element_shape_tensor.flat<int32_t>();
        for (int i = 0; i < element_shape_dims.size(); i++) {
            element_shape_flat(i) = static_cast<int32_t>(element_shape_dims[i]);
        }
        
        // Create tensor for tensor_list_reserve
        tensorflow::Tensor num_elements_tensor(tensorflow::DT_INT32, {});
        num_elements_tensor.scalar<int32_t>()() = 0;  // Start with empty list
        
        // Create tensor lists using TensorListReserve
        auto element_shape = tensorflow::ops::Const(root.WithOpName("element_shape"), element_shape_tensor);
        auto num_elements = tensorflow::ops::Const(root.WithOpName("num_elements"), num_elements_tensor);
        
        auto tensor_list = tensorflow::ops::TensorListReserve(
            root.WithOpName("tensor_list_reserve"), 
            element_shape, 
            num_elements, 
            tensorflow::ops::TensorListReserve::ElementDtype(element_dtype));
        
        // Create batch of tensor lists
        tensorflow::Tensor batch_size_tensor(tensorflow::DT_INT32, {});
        batch_size_tensor.scalar<int32_t>()() = handles_tensor_shape.dim_size(0);
        
        auto batch_size = tensorflow::ops::Const(root.WithOpName("batch_size"), batch_size_tensor);
        auto tensor_list_stack = tensorflow::ops::TensorListStack(
            root.WithOpName("tensor_list_stack"),
            tensor_list,
            batch_size,
            tensorflow::ops::TensorListStack::ElementDtype(tensorflow::DT_VARIANT));
        
        // Create tensor to push
        if (offset >= size) return 0;
        tensorflow::DataType tensor_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t tensor_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> tensor_shape = parseShape(data, offset, size, tensor_rank);
        
        tensorflow::TensorShape tensor_tensor_shape;
        for (auto dim : tensor_shape) {
            tensor_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor tensor_tensor(tensor_dtype, tensor_tensor_shape);
        fillTensorWithDataByType(tensor_tensor, tensor_dtype, data, offset, size);

        // Create placeholders for inputs
        auto input_handles = tensorflow::ops::Placeholder(root.WithOpName("input_handles"), tensorflow::DT_VARIANT);
        auto tensor = tensorflow::ops::Placeholder(root.WithOpName("tensor"), tensor_dtype);

        // Create TensorListPushBackBatch op
        auto result = tensorflow::ops::Raw::TensorListPushBackBatch(
            root.WithOpName("tensor_list_push_back_batch"),
            input_handles,
            tensor);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_handles, tensor_list_stack.output}, {tensor, tensor_tensor}}, 
            {result}, 
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
