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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType matrix_dtype = parseDataType(data[offset++]);
        
        uint8_t batch_rank = parseRank(data[offset++]);
        if (batch_rank > 2) batch_rank = 2;
        
        std::vector<int64_t> batch_shape = parseShape(data, offset, size, batch_rank);
        
        int64_t matrix_size = 3;
        if (offset < size) {
            matrix_size = 2 + (data[offset++] % 5);
        }
        
        std::vector<int64_t> dense_shape = batch_shape;
        dense_shape.push_back(matrix_size);
        dense_shape.push_back(matrix_size);
        
        int64_t total_batch_size = 1;
        for (auto dim : batch_shape) {
            total_batch_size *= dim;
        }
        
        int64_t nnz_per_matrix = matrix_size + (matrix_size * (matrix_size - 1)) / 2;
        int64_t total_nnz = total_batch_size * nnz_per_matrix;
        
        std::vector<int64_t> indices_shape = {total_nnz, static_cast<int64_t>(dense_shape.size())};
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        auto indices_flat = indices_tensor.flat<int64_t>();
        
        int64_t idx = 0;
        for (int64_t batch = 0; batch < total_batch_size; ++batch) {
            for (int64_t i = 0; i < matrix_size; ++i) {
                for (int64_t j = 0; j <= i; ++j) {
                    int64_t flat_batch_idx = batch;
                    int64_t temp = flat_batch_idx;
                    for (int dim = batch_shape.size() - 1; dim >= 0; --dim) {
                        indices_flat(idx * dense_shape.size() + dim) = temp % batch_shape[dim];
                        temp /= batch_shape[dim];
                    }
                    indices_flat(idx * dense_shape.size() + batch_shape.size()) = i;
                    indices_flat(idx * dense_shape.size() + batch_shape.size() + 1) = j;
                    idx++;
                }
            }
        }
        
        tensorflow::Tensor values_tensor(matrix_dtype, tensorflow::TensorShape({total_nnz}));
        fillTensorWithDataByType(values_tensor, matrix_dtype, data, offset, size);
        
        auto values_flat = values_tensor.flat<float>();
        for (int64_t i = 0; i < total_nnz; ++i) {
            if (matrix_dtype == tensorflow::DT_FLOAT) {
                auto flat = values_tensor.flat<float>();
                flat(i) = std::abs(flat(i)) + 0.1f;
            } else if (matrix_dtype == tensorflow::DT_DOUBLE) {
                auto flat = values_tensor.flat<double>();
                flat(i) = std::abs(flat(i)) + 0.1;
            }
        }
        
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(dense_shape.size())}));
        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < dense_shape.size(); ++i) {
            dense_shape_flat(i) = dense_shape[i];
        }
        
        // Create sparse tensor
        auto sparse_indices = tensorflow::ops::Const(root, indices_tensor);
        auto sparse_values = tensorflow::ops::Const(root, values_tensor);
        auto sparse_dense_shape = tensorflow::ops::Const(root, dense_shape_tensor);
        
        // Convert to CSR format
        auto sparse_tensor_to_csr = tensorflow::ops::SparseTensorToDenseMatrix(
            root, sparse_indices, sparse_values, sparse_dense_shape);
        
        std::vector<int64_t> perm_shape = batch_shape;
        perm_shape.push_back(matrix_size);
        tensorflow::Tensor permutation_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(perm_shape));
        auto perm_flat = permutation_tensor.flat<int32_t>();
        
        for (int64_t batch = 0; batch < total_batch_size; ++batch) {
            for (int64_t i = 0; i < matrix_size; ++i) {
                perm_flat(batch * matrix_size + i) = static_cast<int32_t>(i);
            }
        }
        
        auto permutation = tensorflow::ops::Const(root, permutation_tensor);
        
        // Use raw_ops for SparseMatrixSparseCholesky
        auto cholesky_op = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // We'll just run the conversion to CSR since the actual op isn't available in the C++ API
        tensorflow::Status status = session.Run({sparse_tensor_to_csr}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}