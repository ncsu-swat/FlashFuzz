#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_SPARSE_ELEMENTS 20

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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_HALF;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
            break;
        case 11:
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        int64_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= dim;
        }
        
        if (offset >= size) return 0;
        
        uint8_t num_a_elements_byte = data[offset++];
        int64_t num_a_elements = std::min(static_cast<int64_t>(num_a_elements_byte % MAX_SPARSE_ELEMENTS + 1), total_elements);
        
        uint8_t num_b_elements_byte = data[offset++];
        int64_t num_b_elements = std::min(static_cast<int64_t>(num_b_elements_byte % MAX_SPARSE_ELEMENTS + 1), total_elements);
        
        tensorflow::TensorShape a_indices_shape({num_a_elements, static_cast<int64_t>(rank)});
        tensorflow::Tensor a_indices(tensorflow::DT_INT64, a_indices_shape);
        fillTensorWithData<int64_t>(a_indices, data, offset, size);
        
        auto a_indices_matrix = a_indices.matrix<int64_t>();
        for (int64_t i = 0; i < num_a_elements; ++i) {
            for (int64_t j = 0; j < rank; ++j) {
                int64_t val = a_indices_matrix(i, j);
                if (val < 0) val = -val;
                a_indices_matrix(i, j) = val % shape[j];
            }
        }
        
        tensorflow::TensorShape a_values_shape({num_a_elements});
        tensorflow::Tensor a_values(value_dtype, a_values_shape);
        fillTensorWithDataByType(a_values, value_dtype, data, offset, size);
        
        tensorflow::TensorShape a_shape_tensor_shape({static_cast<int64_t>(rank)});
        tensorflow::Tensor a_shape_tensor(tensorflow::DT_INT64, a_shape_tensor_shape);
        auto a_shape_flat = a_shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < shape.size(); ++i) {
            a_shape_flat(i) = shape[i];
        }
        
        tensorflow::TensorShape b_indices_shape({num_b_elements, static_cast<int64_t>(rank)});
        tensorflow::Tensor b_indices(tensorflow::DT_INT64, b_indices_shape);
        fillTensorWithData<int64_t>(b_indices, data, offset, size);
        
        auto b_indices_matrix = b_indices.matrix<int64_t>();
        for (int64_t i = 0; i < num_b_elements; ++i) {
            for (int64_t j = 0; j < rank; ++j) {
                int64_t val = b_indices_matrix(i, j);
                if (val < 0) val = -val;
                b_indices_matrix(i, j) = val % shape[j];
            }
        }
        
        tensorflow::TensorShape b_values_shape({num_b_elements});
        tensorflow::Tensor b_values(value_dtype, b_values_shape);
        fillTensorWithDataByType(b_values, value_dtype, data, offset, size);
        
        tensorflow::TensorShape b_shape_tensor_shape({static_cast<int64_t>(rank)});
        tensorflow::Tensor b_shape_tensor(tensorflow::DT_INT64, b_shape_tensor_shape);
        auto b_shape_flat = b_shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < shape.size(); ++i) {
            b_shape_flat(i) = shape[i];
        }
        
        auto a_indices_op = tensorflow::ops::Const(root, a_indices);
        auto a_values_op = tensorflow::ops::Const(root, a_values);
        auto a_shape_op = tensorflow::ops::Const(root, a_shape_tensor);
        auto b_indices_op = tensorflow::ops::Const(root, b_indices);
        auto b_values_op = tensorflow::ops::Const(root, b_values);
        auto b_shape_op = tensorflow::ops::Const(root, b_shape_tensor);
        
        auto sparse_sparse_maximum = tensorflow::ops::SparseSparseMaximum(
            root, a_indices_op, a_values_op, a_shape_op,
            b_indices_op, b_values_op, b_shape_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_sparse_maximum.output_indices, sparse_sparse_maximum.output_values}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}