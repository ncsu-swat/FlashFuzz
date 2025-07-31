#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NNZ 100

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        uint8_t sp_rank = parseRank(data[offset++]);
        std::vector<int64_t> sp_shape_vec = parseShape(data, offset, size, sp_rank);
        
        if (offset >= size) return 0;
        
        int64_t total_elements = 1;
        for (int64_t dim : sp_shape_vec) {
            total_elements *= dim;
        }
        
        uint8_t nnz_selector = data[offset++];
        if (offset >= size) return 0;
        
        int64_t nnz = 1 + (nnz_selector % std::min(static_cast<int64_t>(MAX_NNZ), total_elements));
        
        tensorflow::Tensor sp_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({nnz, sp_rank}));
        fillTensorWithData<int64_t>(sp_indices_tensor, data, offset, size);
        
        auto indices_matrix = sp_indices_tensor.matrix<int64_t>();
        for (int64_t i = 0; i < nnz; ++i) {
            for (int64_t j = 0; j < sp_rank; ++j) {
                int64_t max_val = sp_shape_vec[j] - 1;
                if (max_val > 0) {
                    indices_matrix(i, j) = std::abs(indices_matrix(i, j)) % (max_val + 1);
                } else {
                    indices_matrix(i, j) = 0;
                }
            }
        }
        
        tensorflow::Tensor sp_values_tensor(values_dtype, tensorflow::TensorShape({nnz}));
        fillTensorWithDataByType(sp_values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor sp_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({sp_rank}));
        auto shape_flat = sp_shape_tensor.flat<int64_t>();
        for (int i = 0; i < sp_rank; ++i) {
            shape_flat(i) = sp_shape_vec[i];
        }

        auto sp_indices = tensorflow::ops::Const(root, sp_indices_tensor);
        auto sp_values = tensorflow::ops::Const(root, sp_values_tensor);
        auto sp_shape = tensorflow::ops::Const(root, sp_shape_tensor);

        auto sparse_softmax_op = tensorflow::ops::SparseSoftmax(root, sp_indices, sp_values, sp_shape);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_softmax_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}