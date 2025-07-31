#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseShapeDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
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
        tensorflow::DataType shape_dtype = parseShapeDataType(data[offset++]);
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape;
        shape_tensor_shape.AddDim(shape_dims.size());
        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        
        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat = shape_tensor.flat<int32_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                flat(i) = static_cast<int32_t>(shape_dims[i]);
            }
        } else {
            auto flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                flat(i) = shape_dims[i];
            }
        }

        tensorflow::DataType seed_dtype = parseShapeDataType(data[offset++]);
        tensorflow::TensorShape seed_shape;
        seed_shape.AddDim(2);
        tensorflow::Tensor seed_tensor(seed_dtype, seed_shape);
        fillTensorWithDataByType(seed_tensor, seed_dtype, data, offset, size);

        tensorflow::DataType counts_dtype = parseDataType(data[offset++]);
        uint8_t counts_rank = parseRank(data[offset++]);
        std::vector<int64_t> counts_shape = parseShape(data, offset, size, counts_rank);
        
        tensorflow::TensorShape counts_tensor_shape;
        for (auto dim : counts_shape) {
            counts_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor counts_tensor(counts_dtype, counts_tensor_shape);
        fillTensorWithDataByType(counts_tensor, counts_dtype, data, offset, size);

        tensorflow::DataType probs_dtype = counts_dtype;
        uint8_t probs_rank = parseRank(data[offset++]);
        std::vector<int64_t> probs_shape = parseShape(data, offset, size, probs_rank);
        
        tensorflow::TensorShape probs_tensor_shape;
        for (auto dim : probs_shape) {
            probs_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor probs_tensor(probs_dtype, probs_tensor_shape);
        fillTensorWithDataByType(probs_tensor, probs_dtype, data, offset, size);

        tensorflow::DataType output_dtype = parseOutputDataType(data[offset++]);

        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto seed_op = tensorflow::ops::Const(root, seed_tensor);
        auto counts_op = tensorflow::ops::Const(root, counts_tensor);
        auto probs_op = tensorflow::ops::Const(root, probs_tensor);

        // Use raw_ops namespace for StatelessRandomBinomial
        tensorflow::Output binomial_op = tensorflow::ops::StatelessRandomUniform(root, shape_op, seed_op);
        
        // Create a placeholder for the actual operation
        // In a real implementation, you would use the correct raw op
        // This is a workaround since StatelessRandomBinomial is not directly available in the C++ API
        std::vector<tensorflow::Output> inputs = {shape_op, seed_op, counts_op, probs_op};
        tensorflow::NodeBuilder node_builder = tensorflow::NodeBuilder("StatelessRandomBinomial", "StatelessRandomBinomial")
            .Input(tensorflow::NodeBuilder::NodeOut(shape_op.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(seed_op.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(counts_op.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(probs_op.node()))
            .Attr("dtype", output_dtype);
            
        tensorflow::Node* node;
        tensorflow::Status status = root.graph()->AddNode(node_builder, &node);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Output output(node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        status = session.Run({output}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}