#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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
#define MAX_SPARSE_ELEMENTS 20

namespace tf_fuzzer_utils {
    void logError(const std::string& msg, const uint8_t* data, size_t size) {
        std::cerr << msg << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 16) {
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
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
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
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        int64_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= dim;
        }
        
        uint8_t num_a_elements_byte = data[offset++];
        uint8_t num_b_elements_byte = data[offset++];
        
        int64_t num_a_elements = std::min(static_cast<int64_t>(num_a_elements_byte % MAX_SPARSE_ELEMENTS + 1), total_elements);
        int64_t num_b_elements = std::min(static_cast<int64_t>(num_b_elements_byte % MAX_SPARSE_ELEMENTS + 1), total_elements);
        
        tensorflow::Tensor a_indices(tensorflow::DT_INT64, tensorflow::TensorShape({num_a_elements, rank}));
        tensorflow::Tensor a_values(dtype, tensorflow::TensorShape({num_a_elements}));
        tensorflow::Tensor a_shape(tensorflow::DT_INT64, tensorflow::TensorShape({rank}));
        
        tensorflow::Tensor b_indices(tensorflow::DT_INT64, tensorflow::TensorShape({num_b_elements, rank}));
        tensorflow::Tensor b_values(dtype, tensorflow::TensorShape({num_b_elements}));
        tensorflow::Tensor b_shape(tensorflow::DT_INT64, tensorflow::TensorShape({rank}));
        
        auto a_shape_flat = a_shape.flat<int64_t>();
        auto b_shape_flat = b_shape.flat<int64_t>();
        for (int i = 0; i < rank; ++i) {
            a_shape_flat(i) = shape[i];
            b_shape_flat(i) = shape[i];
        }
        
        auto a_indices_matrix = a_indices.matrix<int64_t>();
        for (int64_t i = 0; i < num_a_elements; ++i) {
            for (int j = 0; j < rank; ++j) {
                if (offset < size) {
                    int64_t idx = data[offset++] % shape[j];
                    a_indices_matrix(i, j) = idx;
                } else {
                    a_indices_matrix(i, j) = 0;
                }
            }
        }
        
        auto b_indices_matrix = b_indices.matrix<int64_t>();
        for (int64_t i = 0; i < num_b_elements; ++i) {
            for (int j = 0; j < rank; ++j) {
                if (offset < size) {
                    int64_t idx = data[offset++] % shape[j];
                    b_indices_matrix(i, j) = idx;
                } else {
                    b_indices_matrix(i, j) = 0;
                }
            }
        }
        
        fillTensorWithDataByType(a_values, dtype, data, offset, size);
        fillTensorWithDataByType(b_values, dtype, data, offset, size);
        
        auto a_indices_input = tensorflow::ops::Const(root, a_indices);
        auto a_values_input = tensorflow::ops::Const(root, a_values);
        auto a_shape_input = tensorflow::ops::Const(root, a_shape);
        auto b_indices_input = tensorflow::ops::Const(root, b_indices);
        auto b_values_input = tensorflow::ops::Const(root, b_values);
        auto b_shape_input = tensorflow::ops::Const(root, b_shape);
        
        auto sparse_sparse_minimum = tensorflow::ops::SparseSparseMinimum(
            root, a_indices_input, a_values_input, a_shape_input,
            b_indices_input, b_values_input, b_shape_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_sparse_minimum.output_indices, sparse_sparse_minimum.output_values}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}