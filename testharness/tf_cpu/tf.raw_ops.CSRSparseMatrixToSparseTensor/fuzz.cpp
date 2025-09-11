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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType output_type = parseDataType(data[offset++]);
        
        uint8_t batch_size_byte = data[offset++];
        int batch_size = (batch_size_byte % 3) + 1;
        
        uint8_t rows_byte = data[offset++];
        int rows = (rows_byte % 5) + 1;
        
        uint8_t cols_byte = data[offset++];
        int cols = (cols_byte % 5) + 1;
        
        uint8_t nnz_byte = data[offset++];
        int nnz = (nnz_byte % (rows * cols)) + 1;

        std::vector<int64_t> dense_shape_data = {batch_size, rows, cols};
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({3}));
        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        for (int i = 0; i < 3; ++i) {
            dense_shape_flat(i) = dense_shape_data[i];
        }

        std::vector<int64_t> batch_pointers_data;
        for (int b = 0; b <= batch_size; ++b) {
            batch_pointers_data.push_back(b * rows);
        }
        tensorflow::Tensor batch_pointers_tensor(tensorflow::DT_INT64, 
                                                tensorflow::TensorShape({batch_size + 1}));
        auto batch_pointers_flat = batch_pointers_tensor.flat<int64_t>();
        for (int i = 0; i <= batch_size; ++i) {
            batch_pointers_flat(i) = batch_pointers_data[i];
        }

        std::vector<int64_t> row_pointers_data;
        int total_rows = batch_size * rows;
        for (int r = 0; r <= total_rows; ++r) {
            row_pointers_data.push_back((r * nnz) / total_rows);
        }
        tensorflow::Tensor row_pointers_tensor(tensorflow::DT_INT64, 
                                             tensorflow::TensorShape({total_rows + 1}));
        auto row_pointers_flat = row_pointers_tensor.flat<int64_t>();
        for (int i = 0; i <= total_rows; ++i) {
            row_pointers_flat(i) = row_pointers_data[i];
        }

        std::vector<int64_t> col_indices_data;
        for (int i = 0; i < nnz; ++i) {
            col_indices_data.push_back(i % cols);
        }
        tensorflow::Tensor col_indices_tensor(tensorflow::DT_INT64, 
                                            tensorflow::TensorShape({nnz}));
        auto col_indices_flat = col_indices_tensor.flat<int64_t>();
        for (int i = 0; i < nnz; ++i) {
            col_indices_flat(i) = col_indices_data[i];
        }

        tensorflow::Tensor values_tensor(output_type, tensorflow::TensorShape({nnz}));
        fillTensorWithDataByType(values_tensor, output_type, data, offset, size);

        // Use raw_ops for CSRSparseMatrix
        auto csr_sparse_matrix = tensorflow::ops::Raw::CSRSparseMatrix(
            root, dense_shape_tensor, batch_pointers_tensor, 
            row_pointers_tensor, col_indices_tensor, values_tensor);

        // Use raw_ops for CSRSparseMatrixToSparseTensor
        auto result = tensorflow::ops::Raw::CSRSparseMatrixToSparseTensor(
            root, csr_sparse_matrix, output_type);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result.indices, result.values, result.dense_shape}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
