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
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset,
                                                total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset,
                                                 total_size);
      break;
    default:
      break;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto tree_ensemble_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        
        uint8_t cached_tree_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> cached_tree_ids_shape = parseShape(data, offset, size, cached_tree_ids_rank);
        tensorflow::Tensor cached_tree_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(cached_tree_ids_shape));
        fillTensorWithDataByType(cached_tree_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto cached_tree_ids = tensorflow::ops::Const(root, cached_tree_ids_tensor);
        
        uint8_t cached_node_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> cached_node_ids_shape = parseShape(data, offset, size, cached_node_ids_rank);
        tensorflow::Tensor cached_node_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(cached_node_ids_shape));
        fillTensorWithDataByType(cached_node_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto cached_node_ids = tensorflow::ops::Const(root, cached_node_ids_tensor);
        
        uint8_t num_features = (offset < size) ? (data[offset++] % 5 + 1) : 1;
        
        std::vector<tensorflow::Output> bucketized_features_list;
        for (uint8_t i = 0; i < num_features; ++i) {
            uint8_t feature_rank = parseRank(data[offset++]);
            std::vector<int64_t> feature_shape = parseShape(data, offset, size, feature_rank);
            tensorflow::Tensor feature_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(feature_tensor, tensorflow::DT_INT32, data, offset, size);
            auto feature_const = tensorflow::ops::Const(root, feature_tensor);
            bucketized_features_list.push_back(feature_const);
        }
        
        int logits_dimension = 1;
        if (offset < size) {
            logits_dimension = static_cast<int>(data[offset++] % 10 + 1);
        }
        
        // Use raw_ops directly since we don't have boosted_trees_ops.h
        tensorflow::NodeDef node_def;
        node_def.set_op("BoostedTreesTrainingPredict");
        node_def.set_name(root.UniqueName("BoostedTreesTrainingPredict"));
        
        // Add inputs to NodeDef
        tensorflow::NodeDefBuilder builder(node_def.name(), node_def.op());
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(tree_ensemble_handle.node()->name(), 0, tensorflow::DT_RESOURCE));
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(cached_tree_ids.node()->name(), 0, tensorflow::DT_INT32));
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(cached_node_ids.node()->name(), 0, tensorflow::DT_INT32));
        
        // Add bucketized features
        std::vector<tensorflow::NodeDefBuilder::NodeOut> feature_inputs;
        for (const auto& feature : bucketized_features_list) {
            feature_inputs.push_back(tensorflow::NodeDefBuilder::NodeOut(feature.node()->name(), 0, tensorflow::DT_INT32));
        }
        builder.Input(feature_inputs);
        
        // Add attributes
        builder.Attr("logits_dimension", logits_dimension);
        
        // Finalize the NodeDef
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        // Create the operation
        tensorflow::Operation operation;
        status = root.graph()->AddNode(node_def, &operation);
        if (!status.ok()) {
            return -1;
        }
        
        // Create outputs
        tensorflow::Output partial_logits(operation, 0);
        tensorflow::Output tree_ids(operation, 1);
        tensorflow::Output node_ids(operation, 2);
        
        tensorflow::ClientSession session(root);
        
        tensorflow::Tensor resource_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feeds = {
            {tree_ensemble_handle.node()->name(), resource_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run(feeds, {partial_logits, tree_ids, node_ids}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
