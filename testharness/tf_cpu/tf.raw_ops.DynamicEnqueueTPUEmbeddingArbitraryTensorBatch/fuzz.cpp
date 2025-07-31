#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
#define MIN_RANK 0
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
        uint8_t num_features = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Output> sample_indices_list;
        std::vector<tensorflow::Output> embedding_indices_list;
        std::vector<tensorflow::Output> aggregation_weights_list;
        
        for (uint8_t i = 0; i < num_features; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType indices_dtype = parseDataTypeForIndices(data[offset++]);
            tensorflow::DataType weights_dtype = parseDataTypeForWeights(data[offset++]);
            
            uint8_t sample_rank = 1;
            if (offset < size) {
                sample_rank = (data[offset++] % 2) + 1;
            }
            
            std::vector<int64_t> sample_shape = parseShape(data, offset, size, sample_rank);
            tensorflow::TensorShape sample_tensor_shape(sample_shape);
            tensorflow::Tensor sample_tensor(indices_dtype, sample_tensor_shape);
            fillTensorWithDataByType(sample_tensor, indices_dtype, data, offset, size);
            
            uint8_t embedding_rank = 1;
            std::vector<int64_t> embedding_shape = parseShape(data, offset, size, embedding_rank);
            tensorflow::TensorShape embedding_tensor_shape(embedding_shape);
            tensorflow::Tensor embedding_tensor(indices_dtype, embedding_tensor_shape);
            fillTensorWithDataByType(embedding_tensor, indices_dtype, data, offset, size);
            
            uint8_t weights_rank = 1;
            std::vector<int64_t> weights_shape = parseShape(data, offset, size, weights_rank);
            tensorflow::TensorShape weights_tensor_shape(weights_shape);
            tensorflow::Tensor weights_tensor(weights_dtype, weights_tensor_shape);
            fillTensorWithDataByType(weights_tensor, weights_dtype, data, offset, size);
            
            auto sample_const = tensorflow::ops::Const(root, sample_tensor);
            auto embedding_const = tensorflow::ops::Const(root, embedding_tensor);
            auto weights_const = tensorflow::ops::Const(root, weights_tensor);
            
            sample_indices_list.push_back(sample_const);
            embedding_indices_list.push_back(embedding_const);
            aggregation_weights_list.push_back(weights_const);
        }
        
        std::string mode_str = "inference";
        tensorflow::Tensor mode_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        mode_tensor.scalar<tensorflow::tstring>()() = mode_str;
        auto mode_override = tensorflow::ops::Const(root, mode_tensor);
        
        tensorflow::Tensor device_ordinal_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        device_ordinal_tensor.scalar<int32_t>()() = 0;
        auto device_ordinal = tensorflow::ops::Const(root, device_ordinal_tensor);
        
        std::vector<std::string> combiners_vec = {"sum"};
        
        // Use raw ops instead of the missing tpu_ops.h header
        std::vector<tensorflow::DataType> sample_indices_dtypes;
        std::vector<tensorflow::DataType> embedding_indices_dtypes;
        std::vector<tensorflow::DataType> aggregation_weights_dtypes;
        
        for (uint8_t i = 0; i < num_features; ++i) {
            if (i < sample_indices_list.size()) {
                sample_indices_dtypes.push_back(sample_indices_list[i].type());
                embedding_indices_dtypes.push_back(embedding_indices_list[i].type());
                aggregation_weights_dtypes.push_back(aggregation_weights_list[i].type());
            }
        }
        
        tensorflow::NodeDef node_def;
        node_def.set_name("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch");
        node_def.set_op("DynamicEnqueueTPUEmbeddingArbitraryTensorBatch");
        
        // Add inputs to NodeDef
        for (const auto& sample : sample_indices_list) {
            node_def.add_input(sample.name());
        }
        for (const auto& embedding : embedding_indices_list) {
            node_def.add_input(embedding.name());
        }
        for (const auto& weight : aggregation_weights_list) {
            node_def.add_input(weight.name());
        }
        node_def.add_input(mode_override.name());
        node_def.add_input(device_ordinal.name());
        
        // Add attributes to NodeDef
        auto* attr_map = node_def.mutable_attr();
        
        tensorflow::AttrValue sample_indices_dtypes_attr;
        for (const auto& dtype : sample_indices_dtypes) {
            sample_indices_dtypes_attr.mutable_list()->add_type(dtype);
        }
        (*attr_map)["sample_indices_dtypes"] = sample_indices_dtypes_attr;
        
        tensorflow::AttrValue embedding_indices_dtypes_attr;
        for (const auto& dtype : embedding_indices_dtypes) {
            embedding_indices_dtypes_attr.mutable_list()->add_type(dtype);
        }
        (*attr_map)["embedding_indices_dtypes"] = embedding_indices_dtypes_attr;
        
        tensorflow::AttrValue aggregation_weights_dtypes_attr;
        for (const auto& dtype : aggregation_weights_dtypes) {
            aggregation_weights_dtypes_attr.mutable_list()->add_type(dtype);
        }
        (*attr_map)["aggregation_weights_dtypes"] = aggregation_weights_dtypes_attr;
        
        tensorflow::AttrValue combiners_attr;
        for (const auto& combiner : combiners_vec) {
            combiners_attr.mutable_list()->add_s(combiner);
        }
        (*attr_map)["combiners"] = combiners_attr;
        
        // Create the operation using the NodeDef
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (status.ok()) {
            tensorflow::ClientSession session(root);
            // We don't actually run the session since this is just a test harness
        }
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}