#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_handle.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    switch (selector % 2) {  
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Output tree_ensemble_handle = tensorflow::ops::Placeholder(
            root.WithOpName("tree_ensemble_handle"), tensorflow::DT_RESOURCE);

        if (offset >= size) return 0;
        uint8_t num_features_byte = data[offset++];
        int num_features = (num_features_byte % 5) + 1;

        std::vector<tensorflow::Output> bucketized_features;
        
        for (int i = 0; i < num_features; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (rank > 1) rank = 1;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                if (dim <= 0) dim = 1;
                if (dim > 100) dim = 100;
                tensor_shape.AddDim(dim);
            }

            tensorflow::Tensor feature_tensor(tensorflow::DT_INT32, tensor_shape);
            fillTensorWithDataByType(feature_tensor, tensorflow::DT_INT32, data, offset, size);
            
            auto feature_flat = feature_tensor.flat<int32_t>();
            for (int j = 0; j < feature_flat.size(); ++j) {
                if (feature_flat(j) < 0) feature_flat(j) = 0;
                if (feature_flat(j) > 1000) feature_flat(j) = feature_flat(j) % 1000;
            }

            tensorflow::Output feature_placeholder = tensorflow::ops::Placeholder(
                root.WithOpName("bucketized_feature_" + std::to_string(i)), tensorflow::DT_INT32);
            bucketized_features.push_back(feature_placeholder);
        }

        if (bucketized_features.empty()) {
            tensorflow::TensorShape default_shape;
            default_shape.AddDim(1);
            tensorflow::Tensor default_tensor(tensorflow::DT_INT32, default_shape);
            default_tensor.flat<int32_t>()(0) = 0;
            
            tensorflow::Output default_feature = tensorflow::ops::Placeholder(
                root.WithOpName("default_bucketized_feature"), tensorflow::DT_INT32);
            bucketized_features.push_back(default_feature);
        }

        if (offset >= size) return 0;
        int32_t logits_dimension = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&logits_dimension, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            if (logits_dimension <= 0) logits_dimension = 1;
            if (logits_dimension > 10) logits_dimension = logits_dimension % 10 + 1;
        }

        // Use raw_ops API instead of ops::BoostedTreesPredict
        auto predict_op = tensorflow::ops::internal::BoostedTreesPredict(
            root.WithOpName("boosted_trees_predict"),
            tree_ensemble_handle,
            bucketized_features,
            logits_dimension
        );

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        
        tensorflow::TensorShape handle_shape;
        tensorflow::Tensor handle_tensor(tensorflow::DT_RESOURCE, handle_shape);
        feed_dict.push_back({"tree_ensemble_handle", handle_tensor});
        
        for (size_t i = 0; i < bucketized_features.size(); ++i) {
            tensorflow::TensorShape feature_shape;
            feature_shape.AddDim(1);
            tensorflow::Tensor feature_tensor(tensorflow::DT_INT32, feature_shape);
            feature_tensor.flat<int32_t>()(0) = 0;
            feed_dict.push_back({"bucketized_feature_" + std::to_string(i), feature_tensor});
        }

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, {predict_op.logits}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}