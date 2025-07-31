#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include <iostream>
#include <cstring>
#include <vector>

#define MAX_RANK 2
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
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
        tensorflow::DataType dtype_a = parseDataType(data[offset++]);
        tensorflow::DataType dtype_b = parseDataType(data[offset++]);
        
        uint8_t rank_a = parseRank(data[offset++]);
        uint8_t rank_b = parseRank(data[offset++]);
        
        std::vector<int64_t> shape_a = parseShape(data, offset, size, rank_a);
        std::vector<int64_t> shape_b = parseShape(data, offset, size, rank_b);
        
        if (shape_a.size() != 2 || shape_b.size() != 2) {
            return 0;
        }
        
        bool transpose_a = (data[offset % size] % 2) == 1;
        offset++;
        bool transpose_b = (data[offset % size] % 2) == 1;
        offset++;
        bool a_is_sparse = (data[offset % size] % 2) == 1;
        offset++;
        bool b_is_sparse = (data[offset % size] % 2) == 1;
        offset++;
        
        int64_t inner_dim_a = transpose_a ? shape_a[0] : shape_a[1];
        int64_t outer_dim_b = transpose_b ? shape_b[1] : shape_b[0];
        
        if (inner_dim_a != outer_dim_b) {
            if (transpose_a) {
                shape_a[0] = outer_dim_b;
            } else {
                shape_a[1] = outer_dim_b;
            }
        }

        tensorflow::TensorShape tensor_shape_a(shape_a);
        tensorflow::TensorShape tensor_shape_b(shape_b);

        tensorflow::Tensor tensor_a(dtype_a, tensor_shape_a);
        tensorflow::Tensor tensor_b(dtype_b, tensor_shape_b);

        fillTensorWithDataByType(tensor_a, dtype_a, data, offset, size);
        fillTensorWithDataByType(tensor_b, dtype_b, data, offset, size);

        auto input_a = tensorflow::ops::Const(root, tensor_a);
        auto input_b = tensorflow::ops::Const(root, tensor_b);

        auto sparse_matmul = tensorflow::ops::SparseMatMul(root, input_a, input_b,
            tensorflow::ops::SparseMatMul::TransposeA(transpose_a)
            .TransposeB(transpose_b)
            .AIsSparse(a_is_sparse)
            .BIsSparse(b_is_sparse));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_matmul}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}