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

    if (rank >= 2) {
        shape[rank-1] = shape[rank-2];
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
        
        uint8_t rank_l = parseRank(data[offset++]);
        std::vector<int64_t> shape_l = parseShape(data, offset, size, rank_l);
        
        uint8_t rank_grad = parseRank(data[offset++]);
        std::vector<int64_t> shape_grad = parseShape(data, offset, size, rank_grad);
        
        if (shape_l.size() != shape_grad.size()) {
            shape_grad = shape_l;
        }
        
        tensorflow::TensorShape tensor_shape_l(shape_l);
        tensorflow::Tensor tensor_l(dtype, tensor_shape_l);
        fillTensorWithDataByType(tensor_l, dtype, data, offset, size);
        
        tensorflow::TensorShape tensor_shape_grad(shape_grad);
        tensorflow::Tensor tensor_grad(dtype, tensor_shape_grad);
        fillTensorWithDataByType(tensor_grad, dtype, data, offset, size);

        auto l_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root, dtype);

        auto l_node_out = tensorflow::ops::AsNodeOut(root, l_placeholder);
        auto grad_node_out = tensorflow::ops::AsNodeOut(root, grad_placeholder);

        // Build the deprecated BatchCholeskyGrad op directly so the harness hits the raw op keyword.
        tensorflow::Node* batch_cholesky_grad_node = nullptr;
        auto builder = tensorflow::NodeBuilder(root.GetUniqueNameForOp("BatchCholeskyGrad"), "BatchCholeskyGrad")
                           .Input(l_node_out)
                           .Input(grad_node_out);
        root.UpdateStatus(builder.Finalize(root.graph(), &batch_cholesky_grad_node));
        if (!root.ok() || batch_cholesky_grad_node == nullptr) {
            return -1;
        }
        tensorflow::Output batch_cholesky_grad(batch_cholesky_grad_node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{l_placeholder, tensor_l}, {grad_placeholder, tensor_grad}}, 
                                               {batch_cholesky_grad}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
