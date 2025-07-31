#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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

tensorflow::DataType parseValuesDataType(uint8_t selector) {
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

tensorflow::DataType parseWeightsDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType values_dtype = parseValuesDataType(data[offset++]);
        tensorflow::DataType weights_dtype = parseWeightsDataType(data[offset++]);
        
        uint8_t values_rank = parseRank(data[offset++]);
        uint8_t weights_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        std::vector<int64_t> weights_shape = parseShape(data, offset, size, weights_rank);
        
        bool binary_output = (data[offset++] % 2) == 1;
        
        int minlength = -1;
        int maxlength = -1;
        if (offset < size) {
            minlength = static_cast<int>(data[offset++] % 100) - 1;
        }
        if (offset < size) {
            maxlength = static_cast<int>(data[offset++] % 100) - 1;
        }
        
        tensorflow::TensorShape values_tensor_shape;
        for (int64_t dim : values_shape) {
            values_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape weights_tensor_shape;
        for (int64_t dim : weights_shape) {
            weights_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor values_tensor(values_dtype, values_tensor_shape);
        tensorflow::Tensor weights_tensor(weights_dtype, weights_tensor_shape);
        
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        fillTensorWithDataByType(weights_tensor, weights_dtype, data, offset, size);
        
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto weights_input = tensorflow::ops::Const(root, weights_tensor);
        
        // Create the operation using raw_ops
        tensorflow::OutputList outputs;
        tensorflow::NodeBuilder node_builder = 
            tensorflow::NodeBuilder("DenseCountSparseOutput", "DenseCountSparseOutput")
                .Input(values_input.node())
                .Input(weights_input.node())
                .Attr("binary_output", binary_output);
        
        if (minlength >= 0) {
            node_builder.Attr("minlength", minlength);
        }
        
        if (maxlength >= 0) {
            node_builder.Attr("maxlength", maxlength);
        }
        
        tensorflow::Node* node;
        TF_CHECK_OK(node_builder.Finalize(root.graph(), &node));
        
        for (int i = 0; i < 3; ++i) {
            outputs.push_back(tensorflow::Output(node, i));
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        tensorflow::Status status = session.Run({outputs[0], outputs[1], outputs[2]}, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}