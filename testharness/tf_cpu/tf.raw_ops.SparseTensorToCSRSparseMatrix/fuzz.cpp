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
        std::cerr << message << std::endl;
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
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        uint8_t indices_rank = 2;
        uint8_t dense_shape_rank = 1;
        
        if (offset >= size) return 0;
        uint8_t num_entries_byte = data[offset++];
        int64_t num_entries = 1 + (num_entries_byte % 5);
        
        if (offset >= size) return 0;
        uint8_t num_dims_byte = data[offset++];
        int64_t num_dims = 2 + (num_dims_byte % 3);
        
        std::vector<int64_t> indices_shape = {num_entries, num_dims};
        std::vector<int64_t> values_shape = {num_entries};
        std::vector<int64_t> dense_shape_shape = {num_dims};
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(dense_shape_shape));
        
        fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        fillTensorWithData<int64_t>(dense_shape_tensor, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int64_t>();
        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        
        for (int i = 0; i < dense_shape_flat.size(); ++i) {
            if (dense_shape_flat(i) <= 0) {
                dense_shape_flat(i) = 1;
            }
        }
        
        for (int i = 0; i < indices_flat.size(); ++i) {
            int64_t dim_idx = i % num_dims;
            int64_t max_val = dense_shape_flat(dim_idx);
            if (indices_flat(i) < 0 || indices_flat(i) >= max_val) {
                indices_flat(i) = std::abs(indices_flat(i)) % max_val;
            }
        }
        
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto dense_shape_input = tensorflow::ops::Const(root, dense_shape_tensor);
        
        // Use raw_ops namespace for SparseTensorToCSRSparseMatrix
        auto sparse_to_csr = tensorflow::ops::Raw::SparseTensorToCSRSparseMatrix(
            root, indices_input, values_input, dense_shape_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_to_csr}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
