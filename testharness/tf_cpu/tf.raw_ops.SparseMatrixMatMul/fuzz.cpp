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

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 8) {
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
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 5:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 6:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 7:
            dtype = tensorflow::DT_HALF;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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

tensorflow::Tensor createCSRSparseMatrix(const uint8_t* data, size_t& offset, size_t total_size, tensorflow::DataType dtype) {
    if (offset >= total_size) {
        return tensorflow::Tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
    }
    
    int64_t batch_size = 1;
    int64_t rows = 3;
    int64_t cols = 3;
    int64_t nnz = 2;
    
    if (offset + sizeof(int64_t) * 4 <= total_size) {
        std::memcpy(&batch_size, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&rows, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&cols, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&nnz, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        batch_size = std::abs(batch_size) % 3 + 1;
        rows = std::abs(rows) % 5 + 2;
        cols = std::abs(cols) % 5 + 2;
        nnz = std::abs(nnz) % (rows * cols) + 1;
    }
    
    tensorflow::Tensor sparse_matrix(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
    return sparse_matrix;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        tensorflow::Tensor sparse_matrix_a = createCSRSparseMatrix(data, offset, size, dtype);
        
        uint8_t rank_b = parseRank(data[offset++]);
        std::vector<int64_t> shape_b = parseShape(data, offset, size, rank_b);
        
        tensorflow::TensorShape tensor_shape_b;
        for (int64_t dim : shape_b) {
            tensor_shape_b.AddDim(dim);
        }
        
        tensorflow::Tensor tensor_b(dtype, tensor_shape_b);
        fillTensorWithDataByType(tensor_b, dtype, data, offset, size);
        
        bool transpose_a = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool transpose_b = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool adjoint_a = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool adjoint_b = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool transpose_output = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool conjugate_output = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        auto input_a = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        auto input_b = tensorflow::ops::Placeholder(root, dtype);
        
        // Use raw_ops namespace for SparseMatrixMatMul
        auto sparse_matmul = tensorflow::ops::Raw(
            root.WithOpName("SparseMatrixMatMul"),
            "SparseMatrixMatMul",
            {input_a.output, input_b.output},
            {
                {"transpose_a", transpose_a},
                {"transpose_b", transpose_b},
                {"adjoint_a", adjoint_a},
                {"adjoint_b", adjoint_b},
                {"transpose_output", transpose_output},
                {"conjugate_output", conjugate_output}
            }
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_a, sparse_matrix_a}, {input_b, tensor_b}},
            {sparse_matmul},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}