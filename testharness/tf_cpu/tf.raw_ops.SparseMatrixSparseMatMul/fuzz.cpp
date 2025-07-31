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
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 3:
            dtype = tensorflow::DT_COMPLEX128;
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

tensorflow::Output createCSRSparseMatrix(tensorflow::Scope& scope, 
                                        const std::vector<int64_t>& dense_shape,
                                        tensorflow::DataType dtype,
                                        const uint8_t* data, size_t& offset, size_t total_size,
                                        const std::string& name_prefix) {
    
    int64_t rows = dense_shape[dense_shape.size() - 2];
    int64_t cols = dense_shape[dense_shape.size() - 1];
    
    int64_t batch_size = 1;
    for (size_t i = 0; i < dense_shape.size() - 2; ++i) {
        batch_size *= dense_shape[i];
    }
    
    std::vector<int64_t> batch_pointers_shape = {batch_size + 1};
    tensorflow::Tensor batch_pointers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(batch_pointers_shape));
    auto batch_pointers_flat = batch_pointers_tensor.flat<int32_t>();
    for (int64_t i = 0; i <= batch_size; ++i) {
        batch_pointers_flat(i) = static_cast<int32_t>(i * rows);
    }
    
    std::vector<int64_t> row_pointers_shape = {batch_size * (rows + 1)};
    tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(row_pointers_shape));
    auto row_pointers_flat = row_pointers_tensor.flat<int32_t>();
    
    int32_t nnz_per_batch = std::min(static_cast<int32_t>(rows * cols / 4), static_cast<int32_t>(10));
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t r = 0; r <= rows; ++r) {
            row_pointers_flat(b * (rows + 1) + r) = static_cast<int32_t>(b * nnz_per_batch + (r * nnz_per_batch) / rows);
        }
    }
    
    int32_t total_nnz = batch_size * nnz_per_batch;
    std::vector<int64_t> col_indices_shape = {total_nnz};
    tensorflow::Tensor col_indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(col_indices_shape));
    auto col_indices_flat = col_indices_tensor.flat<int32_t>();
    for (int32_t i = 0; i < total_nnz; ++i) {
        if (offset < total_size) {
            col_indices_flat(i) = static_cast<int32_t>(data[offset] % cols);
            offset++;
        } else {
            col_indices_flat(i) = 0;
        }
    }
    
    std::vector<int64_t> values_shape = {total_nnz};
    tensorflow::Tensor values_tensor(dtype, tensorflow::TensorShape(values_shape));
    fillTensorWithDataByType(values_tensor, dtype, data, offset, total_size);
    
    tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(dense_shape.size())}));
    auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
    for (size_t i = 0; i < dense_shape.size(); ++i) {
        dense_shape_flat(i) = dense_shape[i];
    }
    
    auto batch_pointers_op = tensorflow::ops::Const(scope.WithOpName(name_prefix + "_batch_pointers"), batch_pointers_tensor);
    auto row_pointers_op = tensorflow::ops::Const(scope.WithOpName(name_prefix + "_row_pointers"), row_pointers_tensor);
    auto col_indices_op = tensorflow::ops::Const(scope.WithOpName(name_prefix + "_col_indices"), col_indices_tensor);
    auto values_op = tensorflow::ops::Const(scope.WithOpName(name_prefix + "_values"), values_tensor);
    auto dense_shape_op = tensorflow::ops::Const(scope.WithOpName(name_prefix + "_dense_shape"), dense_shape_tensor);
    
    return tensorflow::ops::CSRSparseMatrix(
        scope.WithOpName(name_prefix + "_csr_matrix"),
        dense_shape_op,
        batch_pointers_op,
        row_pointers_op,
        col_indices_op,
        values_op);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t rank_a = parseRank(data[offset++]);
        uint8_t rank_b = parseRank(data[offset++]);
        
        if (rank_a != rank_b) {
            rank_b = rank_a;
        }
        
        std::vector<int64_t> shape_a = parseShape(data, offset, size, rank_a);
        std::vector<int64_t> shape_b = parseShape(data, offset, size, rank_b);
        
        if (shape_a.size() >= 2 && shape_b.size() >= 2) {
            shape_b[shape_b.size() - 2] = shape_a[shape_a.size() - 1];
        }
        
        bool transpose_a = (data[offset % size] & 1) != 0; offset++;
        bool transpose_b = (data[offset % size] & 1) != 0; offset++;
        bool adjoint_a = (data[offset % size] & 1) != 0; offset++;
        bool adjoint_b = (data[offset % size] & 1) != 0; offset++;
        
        if (transpose_a && adjoint_a) {
            adjoint_a = false;
        }
        if (transpose_b && adjoint_b) {
            adjoint_b = false;
        }
        
        auto csr_a = createCSRSparseMatrix(root, shape_a, dtype, data, offset, size, "matrix_a");
        auto csr_b = createCSRSparseMatrix(root, shape_b, dtype, data, offset, size, "matrix_b");
        
        auto result = tensorflow::ops::Raw(
            root.WithOpName("sparse_matmul"),
            "SparseMatrixSparseMatMul",
            {csr_a, csr_b},
            {{"type", dtype},
             {"transpose_a", transpose_a},
             {"transpose_b", transpose_b},
             {"adjoint_a", adjoint_a},
             {"adjoint_b", adjoint_b}}
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}