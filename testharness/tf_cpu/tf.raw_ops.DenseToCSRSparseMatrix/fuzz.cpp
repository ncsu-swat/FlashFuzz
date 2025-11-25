#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/graph/node_builder.h"
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
        tensorflow::DataType dense_dtype = parseDataType(data[offset++]);
        
        // DenseToCSRSparseMatrix supports rank 2 or 3 inputs.
        uint8_t dense_rank = static_cast<uint8_t>(2 + (data[offset++] % 2));
        std::vector<int64_t> dense_shape = parseShape(data, offset, size, dense_rank);
        
        tensorflow::TensorShape dense_tensor_shape;
        for (int64_t dim : dense_shape) {
            dense_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor dense_tensor(dense_dtype, dense_tensor_shape);
        fillTensorWithDataByType(dense_tensor, dense_dtype, data, offset, size);
        
        int64_t max_indices = std::max<int64_t>(dense_tensor.NumElements(), 1);
        int64_t num_indices = 1;
        if (offset < size) {
            num_indices = static_cast<int64_t>(data[offset++]) % max_indices;
            if (num_indices == 0) {
                num_indices = 1;
            }
        }
        
        tensorflow::Tensor indices_tensor(
            tensorflow::DT_INT64,
            tensorflow::TensorShape({num_indices, static_cast<int64_t>(dense_rank)}));
        auto indices_matrix = indices_tensor.matrix<int64_t>();
        for (int64_t i = 0; i < num_indices; ++i) {
            for (int64_t j = 0; j < dense_rank; ++j) {
                int64_t value = 0;
                if (offset + sizeof(int64_t) <= size) {
                    std::memcpy(&value, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else if (offset < size) {
                    value = data[offset++];
                }
                int64_t dim_size = dense_shape[static_cast<size_t>(j)];
                indices_matrix(i, j) = std::abs(value) % dim_size;
            }
        }
        
        auto dense_input = tensorflow::ops::Placeholder(root, dense_dtype);
        auto indices = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        tensorflow::NodeBuilder builder(root.GetUniqueNameForOp("DenseToCSRSparseMatrix"),
                                        "DenseToCSRSparseMatrix");
        builder.Input(dense_input.node());
        builder.Input(indices.node());
        builder.Attr("T", dense_dtype);
        
        tensorflow::Node* dense_to_csr_node = nullptr;
        tensorflow::Status build_status = builder.Finalize(root.graph(), &dense_to_csr_node);
        if (!build_status.ok()) {
            return -1;
        }
        tensorflow::Output result(dense_to_csr_node);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{dense_input, dense_tensor},
                                                 {indices, indices_tensor}},
                                                {result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
