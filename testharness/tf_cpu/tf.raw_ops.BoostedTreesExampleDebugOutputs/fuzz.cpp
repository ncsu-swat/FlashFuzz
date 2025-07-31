#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_handle.h"
#include <iostream>
#include <vector>
#include <cstring>
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
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::ResourceHandle tree_ensemble_handle;
        tree_ensemble_handle.set_device("/cpu:0");
        tree_ensemble_handle.set_container("test_container");
        tree_ensemble_handle.set_name("test_tree_ensemble");
        tree_ensemble_handle.set_hash_code(12345);
        tree_ensemble_handle.set_maybe_type_name("BoostedTreesEnsembleResource");

        auto tree_ensemble_tensor = tensorflow::Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        tree_ensemble_tensor.scalar<tensorflow::ResourceHandle>()() = tree_ensemble_handle;

        auto tree_ensemble_input = tensorflow::ops::Const(root, tree_ensemble_tensor);

        if (offset >= size) return 0;
        uint8_t num_features_byte = data[offset++];
        int num_features = (num_features_byte % 5) + 1;

        std::vector<tensorflow::Output> bucketized_features;
        
        for (int i = 0; i < num_features; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (rank == 0) rank = 1;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (auto dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor feature_tensor(tensorflow::DT_INT32, tensor_shape);
            fillTensorWithDataByType(feature_tensor, tensorflow::DT_INT32, data, offset, size);
            
            auto feature_input = tensorflow::ops::Const(root, feature_tensor);
            bucketized_features.push_back(feature_input);
        }

        if (bucketized_features.empty()) {
            tensorflow::Tensor default_feature(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
            default_feature.flat<int32_t>()(0) = 0;
            auto default_input = tensorflow::ops::Const(root, default_feature);
            bucketized_features.push_back(default_input);
        }

        if (offset >= size) return 0;
        int32_t logits_dimension = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&logits_dimension, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            logits_dimension = std::abs(logits_dimension) % 10 + 1;
        }

        std::cout << "Tree ensemble handle: " << tree_ensemble_handle.DebugString() << std::endl;
        std::cout << "Number of bucketized features: " << bucketized_features.size() << std::endl;
        std::cout << "Logits dimension: " << logits_dimension << std::endl;

        // Use raw_ops directly instead of the missing header
        auto debug_outputs = tensorflow::ops::Operation(
            root.WithOpName("BoostedTreesExampleDebugOutputs"),
            "BoostedTreesExampleDebugOutputs",
            {tree_ensemble_input, bucketized_features},
            {{"logits_dimension", logits_dimension}}
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({debug_outputs.output(0)}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}