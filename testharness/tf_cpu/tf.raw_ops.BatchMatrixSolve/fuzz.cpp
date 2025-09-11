#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/linalg_ops.h"
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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
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
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t matrix_rank = parseRank(data[offset++]);
        if (matrix_rank < 3) matrix_rank = 3;
        
        std::vector<int64_t> matrix_shape = parseShape(data, offset, size, matrix_rank);
        if (matrix_shape.size() < 3) {
            matrix_shape = {2, 3, 3};
        }
        
        int64_t batch_size = matrix_shape[0];
        int64_t n = matrix_shape[matrix_shape.size() - 1];
        int64_t m = matrix_shape[matrix_shape.size() - 2];
        
        matrix_shape[matrix_shape.size() - 1] = n;
        matrix_shape[matrix_shape.size() - 2] = n;
        
        std::vector<int64_t> rhs_shape = matrix_shape;
        rhs_shape[rhs_shape.size() - 1] = m;
        
        tensorflow::TensorShape matrix_tensor_shape;
        for (int64_t dim : matrix_shape) {
            matrix_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape rhs_tensor_shape;
        for (int64_t dim : rhs_shape) {
            rhs_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor matrix_tensor(dtype, matrix_tensor_shape);
        tensorflow::Tensor rhs_tensor(dtype, rhs_tensor_shape);
        
        fillTensorWithDataByType(matrix_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(rhs_tensor, dtype, data, offset, size);
        
        bool adjoint = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        auto matrix_input = tensorflow::ops::Const(root, matrix_tensor);
        auto rhs_input = tensorflow::ops::Const(root, rhs_tensor);
        
        auto batch_matrix_solve = tensorflow::ops::MatrixSolve(
            root, matrix_input, rhs_input,
            tensorflow::ops::MatrixSolve::Adjoint(adjoint)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({batch_matrix_solve}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
