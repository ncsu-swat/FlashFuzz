#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MIN_NUM_TENSORS 2
#define MAX_NUM_TENSORS 5

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 13) {
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
            dtype = tensorflow::DT_HALF;
            break;
        case 11:
            dtype = tensorflow::DT_UINT32;
            break;
        case 12:
            dtype = tensorflow::DT_UINT64;
            break;
        default:
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
        
        uint8_t num_tensors_byte = data[offset++];
        int num_tensors = MIN_NUM_TENSORS + (num_tensors_byte % (MAX_NUM_TENSORS - MIN_NUM_TENSORS + 1));
        
        uint8_t rank = parseRank(data[offset++]);
        
        int32_t axis_value = 0;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&axis_value, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            axis_value = axis_value % rank;
            if (axis_value < 0) axis_value += rank;
        }
        
        std::vector<tensorflow::Output> input_tensors;
        std::vector<int64_t> base_shape = parseShape(data, offset, size, rank);
        
        for (int i = 0; i < num_tensors; ++i) {
            std::vector<int64_t> tensor_shape = base_shape;
            
            if (offset < size && rank > 0) {
                uint8_t dim_modifier = data[offset++];
                int64_t new_dim = 1 + (dim_modifier % 5);
                if (axis_value < tensor_shape.size()) {
                    tensor_shape[axis_value] = new_dim;
                }
            }
            
            tensorflow::TensorShape shape;
            for (int64_t dim : tensor_shape) {
                shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            std::string tensor_name = "input_tensor_" + std::to_string(i);
            auto placeholder = tensorflow::ops::Placeholder(root.WithOpName(tensor_name), dtype);
            input_tensors.push_back(placeholder);
        }
        
        tensorflow::Tensor axis_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        axis_tensor.scalar<int32_t>()() = axis_value;
        auto axis_placeholder = tensorflow::ops::Placeholder(root.WithOpName("axis"), tensorflow::DT_INT32);
        
        auto concat_op = tensorflow::ops::Concat(root.WithOpName("concat"), input_tensors, axis_placeholder);
        
        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        for (int i = 0; i < num_tensors; ++i) {
            std::vector<int64_t> tensor_shape = base_shape;
            if (axis_value < tensor_shape.size()) {
                tensor_shape[axis_value] = 1 + (i % 5);
            }
            
            tensorflow::TensorShape shape;
            for (int64_t dim : tensor_shape) {
                shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            std::string tensor_name = "input_tensor_" + std::to_string(i);
            feed_dict.push_back({tensor_name, tensor});
        }
        feed_dict.push_back({"axis", axis_tensor});
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, {concat_op}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
