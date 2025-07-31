#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataTypeForIndices(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_INT32;
    }
}

tensorflow::DataType parseDataTypeForWeights(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_FLOAT;
        case 1:
            return tensorflow::DT_DOUBLE;
        default:
            return tensorflow::DT_FLOAT;
    }
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_tables = (data[offset++] % 3) + 1;
        
        tensorflow::DataType indices_dtype = parseDataTypeForIndices(data[offset++]);
        tensorflow::DataType weights_dtype = parseDataTypeForWeights(data[offset++]);
        
        std::vector<tensorflow::Input> sample_indices_list;
        std::vector<tensorflow::Input> embedding_indices_list;
        std::vector<tensorflow::Input> aggregation_weights_list;
        
        for (uint8_t i = 0; i < num_tables; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = 1;
            std::vector<int64_t> shape = {static_cast<int64_t>((data[offset++] % 10) + 1)};
            
            tensorflow::TensorShape tensor_shape;
            for (auto dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor sample_indices_tensor(indices_dtype, tensor_shape);
            fillTensorWithDataByType(sample_indices_tensor, indices_dtype, data, offset, size);
            auto sample_indices_input = tensorflow::ops::Const(root, sample_indices_tensor);
            sample_indices_list.push_back(sample_indices_input);
            
            tensorflow::Tensor embedding_indices_tensor(indices_dtype, tensor_shape);
            fillTensorWithDataByType(embedding_indices_tensor, indices_dtype, data, offset, size);
            auto embedding_indices_input = tensorflow::ops::Const(root, embedding_indices_tensor);
            embedding_indices_list.push_back(embedding_indices_input);
            
            tensorflow::Tensor aggregation_weights_tensor(weights_dtype, tensor_shape);
            fillTensorWithDataByType(aggregation_weights_tensor, weights_dtype, data, offset, size);
            auto aggregation_weights_input = tensorflow::ops::Const(root, aggregation_weights_tensor);
            aggregation_weights_list.push_back(aggregation_weights_input);
        }
        
        std::string mode_override_str = "unspecified";
        tensorflow::Tensor mode_override_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        mode_override_tensor.scalar<tensorflow::tstring>()() = mode_override_str;
        auto mode_override_input = tensorflow::ops::Const(root, mode_override_tensor);
        
        // Use raw_ops instead of ops::EnqueueTPUEmbeddingSparseBatch
        auto enqueue_op = tensorflow::ops::Raw(
            root.WithOpName("EnqueueTPUEmbeddingSparseBatch"),
            "EnqueueTPUEmbeddingSparseBatch",
            sample_indices_list,
            embedding_indices_list,
            aggregation_weights_list,
            {mode_override_input},
            tensorflow::ops::Raw::Attrs()
                .SetAttr("device_ordinal", -1)
                .SetAttr("combiners", std::vector<std::string>())
        );

        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({enqueue_op}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}