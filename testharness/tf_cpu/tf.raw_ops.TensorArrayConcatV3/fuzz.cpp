#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

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
    switch (selector % 17) {  
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
            dtype = tensorflow::DT_HALF;
            break;
        case 12:
            dtype = tensorflow::DT_UINT32;
            break;
        case 13:
            dtype = tensorflow::DT_UINT64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 15:
            dtype = tensorflow::DT_STRING;
            break;
        case 16:
            dtype = tensorflow::DT_FLOAT;
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
        case tensorflow::DT_STRING:
            {
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
            }
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
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        if (dtype == tensorflow::DT_STRING) {
            return 0;
        }

        int32_t array_size = (data[offset++] % 5) + 1;
        
        tensorflow::TensorShape element_shape_except0;
        uint8_t element_rank = parseRank(data[offset++]);
        if (element_rank > 0) {
            std::vector<int64_t> element_dims = parseShape(data, offset, size, element_rank);
            for (auto dim : element_dims) {
                element_shape_except0.AddDim(dim);
            }
        }

        // Create a tensor array
        auto size_tensor = tensorflow::ops::Const(root, tensorflow::Tensor(array_size));
        auto tensor_array = tensorflow::ops::DataFlowOps::TensorArray(root, size_tensor, dtype);
        
        auto flow = tensorflow::ops::Const(root, 0.0f);
        
        tensorflow::Output current_flow = flow;
        
        for (int32_t i = 0; i < array_size; ++i) {
            if (offset >= size) break;
            
            int64_t first_dim = (data[offset++] % 5) + 1;
            
            tensorflow::TensorShape write_shape;
            write_shape.AddDim(first_dim);
            for (int j = 0; j < element_shape_except0.dims(); ++j) {
                write_shape.AddDim(element_shape_except0.dim_size(j));
            }
            
            tensorflow::Tensor write_tensor(dtype, write_shape);
            fillTensorWithDataByType(write_tensor, dtype, data, offset, size);
            
            auto write_value = tensorflow::ops::Const(root, write_tensor);
            auto index = tensorflow::ops::Const(root, i);
            auto write_op = tensorflow::ops::DataFlowOps::TensorArrayWrite(
                root, tensor_array.handle, index, write_value, current_flow);
            current_flow = write_op;
        }

        // Create a vector of int64 for element_shape_except0
        std::vector<int64_t> element_shape_vec;
        for (int i = 0; i < element_shape_except0.dims(); ++i) {
            element_shape_vec.push_back(element_shape_except0.dim_size(i));
        }
        auto element_shape_tensor = tensorflow::ops::Const(root, tensorflow::Tensor(
            tensorflow::TensorShape({static_cast<int64_t>(element_shape_vec.size())}), element_shape_vec));

        // Use TensorArrayConcat instead of TensorArrayConcatV3
        auto concat_result = tensorflow::ops::DataFlowOps::TensorArrayConcat(
            root, 
            tensor_array.handle, 
            current_flow,
            element_shape_tensor
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({concat_result.value, concat_result.lengths}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
