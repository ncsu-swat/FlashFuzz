#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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

tensorflow::Output createCSRSparseMatrix(tensorflow::Scope& scope, tensorflow::DataType dtype, 
                                        const uint8_t* data, size_t& offset, size_t total_size) {
    int64_t num_rows = 3;
    int64_t num_cols = 3;
    int64_t nnz = 2;
    
    tensorflow::TensorShape dense_shape({2});
    tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, dense_shape);
    auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
    dense_shape_flat(0) = num_rows;
    dense_shape_flat(1) = num_cols;
    
    tensorflow::TensorShape batch_pointers_shape({2});
    tensorflow::Tensor batch_pointers_tensor(tensorflow::DT_INT32, batch_pointers_shape);
    auto batch_pointers_flat = batch_pointers_tensor.flat<int32_t>();
    batch_pointers_flat(0) = 0;
    batch_pointers_flat(1) = nnz;
    
    tensorflow::TensorShape row_pointers_shape({num_rows + 1});
    tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT32, row_pointers_shape);
    auto row_pointers_flat = row_pointers_tensor.flat<int32_t>();
    row_pointers_flat(0) = 0;
    row_pointers_flat(1) = 1;
    row_pointers_flat(2) = 2;
    row_pointers_flat(3) = 2;
    
    tensorflow::TensorShape col_indices_shape({nnz});
    tensorflow::Tensor col_indices_tensor(tensorflow::DT_INT32, col_indices_shape);
    auto col_indices_flat = col_indices_tensor.flat<int32_t>();
    col_indices_flat(0) = 0;
    col_indices_flat(1) = 1;
    
    tensorflow::TensorShape values_shape({nnz});
    tensorflow::Tensor values_tensor(dtype, values_shape);
    fillTensorWithDataByType(values_tensor, dtype, data, offset, total_size);
    
    auto dense_shape_op = tensorflow::ops::Const(scope, dense_shape_tensor);
    auto batch_pointers_op = tensorflow::ops::Const(scope, batch_pointers_tensor);
    auto row_pointers_op = tensorflow::ops::Const(scope, row_pointers_tensor);
    auto col_indices_op = tensorflow::ops::Const(scope, col_indices_tensor);
    auto values_op = tensorflow::ops::Const(scope, values_tensor);
    
    return tensorflow::ops::CSRSparseMatrix(scope, dense_shape_op, batch_pointers_op, 
                                           row_pointers_op, col_indices_op, values_op);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        tensorflow::TensorShape alpha_shape({});
        tensorflow::Tensor alpha_tensor(dtype, alpha_shape);
        fillTensorWithDataByType(alpha_tensor, dtype, data, offset, size);
        auto alpha_op = tensorflow::ops::Const(root, alpha_tensor);
        
        tensorflow::TensorShape beta_shape({});
        tensorflow::Tensor beta_tensor(dtype, beta_shape);
        fillTensorWithDataByType(beta_tensor, dtype, data, offset, size);
        auto beta_op = tensorflow::ops::Const(root, beta_tensor);
        
        int64_t num_rows = 2;
        int64_t num_cols = 2;
        int64_t nnz = 1;
        
        tensorflow::TensorShape dense_shape({2});
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, dense_shape);
        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        dense_shape_flat(0) = num_rows;
        dense_shape_flat(1) = num_cols;
        
        tensorflow::TensorShape batch_pointers_shape({2});
        tensorflow::Tensor batch_pointers_tensor(tensorflow::DT_INT32, batch_pointers_shape);
        auto batch_pointers_flat = batch_pointers_tensor.flat<int32_t>();
        batch_pointers_flat(0) = 0;
        batch_pointers_flat(1) = nnz;
        
        tensorflow::TensorShape row_pointers_shape({num_rows + 1});
        tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT32, row_pointers_shape);
        auto row_pointers_flat = row_pointers_tensor.flat<int32_t>();
        row_pointers_flat(0) = 0;
        row_pointers_flat(1) = 1;
        row_pointers_flat(2) = 1;
        
        tensorflow::TensorShape col_indices_shape({nnz});
        tensorflow::Tensor col_indices_tensor(tensorflow::DT_INT32, col_indices_shape);
        auto col_indices_flat = col_indices_tensor.flat<int32_t>();
        col_indices_flat(0) = 0;
        
        tensorflow::TensorShape values_shape({nnz});
        tensorflow::Tensor values_tensor_a(dtype, values_shape);
        tensorflow::Tensor values_tensor_b(dtype, values_shape);
        fillTensorWithDataByType(values_tensor_a, dtype, data, offset, size);
        fillTensorWithDataByType(values_tensor_b, dtype, data, offset, size);
        
        auto dense_shape_op = tensorflow::ops::Const(root, dense_shape_tensor);
        auto batch_pointers_op = tensorflow::ops::Const(root, batch_pointers_tensor);
        auto row_pointers_op = tensorflow::ops::Const(root, row_pointers_tensor);
        auto col_indices_op = tensorflow::ops::Const(root, col_indices_tensor);
        auto values_op_a = tensorflow::ops::Const(root, values_tensor_a);
        auto values_op_b = tensorflow::ops::Const(root, values_tensor_b);
        
        auto sparse_matrix_a = tensorflow::ops::CSRSparseMatrix(root, dense_shape_op, 
                                                              batch_pointers_op, 
                                                              row_pointers_op, 
                                                              col_indices_op, 
                                                              values_op_a);
        
        auto sparse_matrix_b = tensorflow::ops::CSRSparseMatrix(root, dense_shape_op, 
                                                              batch_pointers_op, 
                                                              row_pointers_op, 
                                                              col_indices_op, 
                                                              values_op_b);
        
        auto result = tensorflow::ops::Raw(root.WithOpName("SparseMatrixAdd"),
                                          "SparseMatrixAdd",
                                          {sparse_matrix_a, sparse_matrix_b, alpha_op, beta_op},
                                          {tensorflow::DT_VARIANT});

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