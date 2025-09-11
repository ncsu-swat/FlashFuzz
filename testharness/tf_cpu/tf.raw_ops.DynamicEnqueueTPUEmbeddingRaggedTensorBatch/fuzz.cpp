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
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector, bool is_int_type) {
    tensorflow::DataType dtype;
    if (is_int_type) {
        switch (selector % 2) {
            case 0:
                dtype = tensorflow::DT_INT32;
                break;
            case 1:
                dtype = tensorflow::DT_INT64;
                break;
        }
    } else {
        switch (selector % 2) {
            case 0:
                dtype = tensorflow::DT_FLOAT;
                break;
            case 1:
                dtype = tensorflow::DT_DOUBLE;
                break;
        }
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_tables = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Output> sample_splits;
        std::vector<tensorflow::Output> embedding_indices;
        std::vector<tensorflow::Output> aggregation_weights;
        std::vector<int> table_ids;
        
        tensorflow::DataType int_dtype = parseDataType(data[offset++], true);
        tensorflow::DataType float_dtype = parseDataType(data[offset++], false);
        
        for (uint8_t i = 0; i < num_tables; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::Tensor sample_split_tensor(int_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(sample_split_tensor, int_dtype, data, offset, size);
            auto sample_split_op = tensorflow::ops::Const(root, sample_split_tensor);
            sample_splits.push_back(sample_split_op);
            
            tensorflow::Tensor embedding_indices_tensor(int_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(embedding_indices_tensor, int_dtype, data, offset, size);
            auto embedding_indices_op = tensorflow::ops::Const(root, embedding_indices_tensor);
            embedding_indices.push_back(embedding_indices_op);
            
            tensorflow::Tensor aggregation_weights_tensor(float_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(aggregation_weights_tensor, float_dtype, data, offset, size);
            auto aggregation_weights_op = tensorflow::ops::Const(root, aggregation_weights_tensor);
            aggregation_weights.push_back(aggregation_weights_op);
            
            table_ids.push_back(i);
        }
        
        if (sample_splits.empty()) return 0;
        
        tensorflow::Tensor mode_override_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        mode_override_tensor.scalar<tensorflow::tstring>()() = "inference";
        auto mode_override_op = tensorflow::ops::Const(root, mode_override_tensor);
        
        tensorflow::Tensor device_ordinal_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        device_ordinal_tensor.scalar<int32_t>()() = 0;
        auto device_ordinal_op = tensorflow::ops::Const(root, device_ordinal_tensor);
        
        // Use raw_ops directly instead of including tpu_ops.h
        std::vector<tensorflow::DataType> sample_splits_dtypes(sample_splits.size(), int_dtype);
        std::vector<tensorflow::DataType> embedding_indices_dtypes(embedding_indices.size(), int_dtype);
        std::vector<tensorflow::DataType> aggregation_weights_dtypes(aggregation_weights.size(), float_dtype);
        
        std::vector<std::string> combiners;
        std::vector<int64_t> max_sequence_lengths;
        std::vector<int64_t> num_features;
        
        tensorflow::NodeDef node_def;
        node_def.set_op("DynamicEnqueueTPUEmbeddingRaggedTensorBatch");
        node_def.set_name(root.UniqueName("DynamicEnqueueTPUEmbeddingRaggedTensorBatch"));
        
        tensorflow::NodeDefBuilder builder(node_def.name(), node_def.op());
        builder.Input(tensorflow::NodeDefBuilder::NodeOut("sample_splits", 0, sample_splits_dtypes))
               .Input(tensorflow::NodeDefBuilder::NodeOut("embedding_indices", 0, embedding_indices_dtypes))
               .Input(tensorflow::NodeDefBuilder::NodeOut("aggregation_weights", 0, aggregation_weights_dtypes))
               .Input(tensorflow::NodeDefBuilder::NodeOut("mode_override", 0, tensorflow::DT_STRING))
               .Input(tensorflow::NodeDefBuilder::NodeOut("device_ordinal", 0, tensorflow::DT_INT32))
               .Attr("table_ids", table_ids)
               .Attr("combiners", combiners)
               .Attr("max_sequence_lengths", max_sequence_lengths)
               .Attr("num_features", num_features);
        
        tensorflow::Status status;
        tensorflow::Node* node;
        status = builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        status = session.Run({}, {}, {node->name()}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
