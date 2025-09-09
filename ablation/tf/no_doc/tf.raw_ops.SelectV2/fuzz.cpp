#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT16;
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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        uint8_t condition_rank = parseRank(data[offset++]);
        std::vector<int64_t> condition_shape = parseShape(data, offset, size, condition_rank);
        
        uint8_t then_dtype_selector = data[offset++];
        tensorflow::DataType then_dtype = parseDataType(then_dtype_selector);
        uint8_t then_rank = parseRank(data[offset++]);
        std::vector<int64_t> then_shape = parseShape(data, offset, size, then_rank);
        
        uint8_t else_dtype_selector = data[offset++];
        tensorflow::DataType else_dtype = parseDataType(else_dtype_selector);
        uint8_t else_rank = parseRank(data[offset++]);
        std::vector<int64_t> else_shape = parseShape(data, offset, size, else_rank);

        tensorflow::TensorShape condition_tensor_shape(condition_shape);
        tensorflow::TensorShape then_tensor_shape(then_shape);
        tensorflow::TensorShape else_tensor_shape(else_shape);

        tensorflow::Tensor condition_tensor(tensorflow::DT_BOOL, condition_tensor_shape);
        tensorflow::Tensor then_tensor(then_dtype, then_tensor_shape);
        tensorflow::Tensor else_tensor(else_dtype, else_tensor_shape);

        fillTensorWithData<bool>(condition_tensor, data, offset, size);
        fillTensorWithDataByType(then_tensor, then_dtype, data, offset, size);
        fillTensorWithDataByType(else_tensor, else_dtype, data, offset, size);

        std::cout << "Condition tensor shape: ";
        for (int i = 0; i < condition_tensor.dims(); ++i) {
            std::cout << condition_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Then tensor shape: ";
        for (int i = 0; i < then_tensor.dims(); ++i) {
            std::cout << then_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Else tensor shape: ";
        for (int i = 0; i < else_tensor.dims(); ++i) {
            std::cout << else_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto condition_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        auto then_placeholder = tensorflow::ops::Placeholder(root, then_dtype);
        auto else_placeholder = tensorflow::ops::Placeholder(root, else_dtype);
        
        auto select_op = tensorflow::ops::SelectV2(root, condition_placeholder, then_placeholder, else_placeholder);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{condition_placeholder, condition_tensor}, 
                                                 {then_placeholder, then_tensor}, 
                                                 {else_placeholder, else_tensor}}, 
                                                {select_op}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "SelectV2 operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "SelectV2 operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}