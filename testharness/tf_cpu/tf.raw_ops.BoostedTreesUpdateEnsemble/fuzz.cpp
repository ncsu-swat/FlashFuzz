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
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
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
    if (size < 50) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto tree_ensemble_handle = tensorflow::ops::VarHandleOp(root, tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        uint8_t feature_ids_rank = parseRank(data[offset++]);
        auto feature_ids_shape = parseShape(data, offset, size, feature_ids_rank);
        tensorflow::Tensor feature_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_ids_shape));
        fillTensorWithDataByType(feature_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto feature_ids = tensorflow::ops::Const(root, feature_ids_tensor);

        uint8_t num_lists = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::Output> node_ids_list;
        std::vector<tensorflow::Output> gains_list;
        std::vector<tensorflow::Output> thresholds_list;
        std::vector<tensorflow::Output> left_node_contribs_list;
        std::vector<tensorflow::Output> right_node_contribs_list;

        for (uint8_t i = 0; i < num_lists; ++i) {
            uint8_t node_ids_rank = parseRank(data[offset++]);
            auto node_ids_shape = parseShape(data, offset, size, node_ids_rank);
            tensorflow::Tensor node_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(node_ids_shape));
            fillTensorWithDataByType(node_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            node_ids_list.push_back(tensorflow::ops::Const(root, node_ids_tensor));

            uint8_t gains_rank = parseRank(data[offset++]);
            auto gains_shape = parseShape(data, offset, size, gains_rank);
            tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gains_shape));
            fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
            gains_list.push_back(tensorflow::ops::Const(root, gains_tensor));

            uint8_t thresholds_rank = parseRank(data[offset++]);
            auto thresholds_shape = parseShape(data, offset, size, thresholds_rank);
            tensorflow::Tensor thresholds_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(thresholds_shape));
            fillTensorWithDataByType(thresholds_tensor, tensorflow::DT_INT32, data, offset, size);
            thresholds_list.push_back(tensorflow::ops::Const(root, thresholds_tensor));

            uint8_t left_contribs_rank = parseRank(data[offset++]);
            auto left_contribs_shape = parseShape(data, offset, size, left_contribs_rank);
            tensorflow::Tensor left_contribs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(left_contribs_shape));
            fillTensorWithDataByType(left_contribs_tensor, tensorflow::DT_FLOAT, data, offset, size);
            left_node_contribs_list.push_back(tensorflow::ops::Const(root, left_contribs_tensor));

            uint8_t right_contribs_rank = parseRank(data[offset++]);
            auto right_contribs_shape = parseShape(data, offset, size, right_contribs_rank);
            tensorflow::Tensor right_contribs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(right_contribs_shape));
            fillTensorWithDataByType(right_contribs_tensor, tensorflow::DT_FLOAT, data, offset, size);
            right_node_contribs_list.push_back(tensorflow::ops::Const(root, right_contribs_tensor));
        }

        int32_t max_depth_val = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_depth_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_depth_val = std::abs(max_depth_val) % 10 + 1;
        }
        auto max_depth = tensorflow::ops::Const(root, max_depth_val);

        float learning_rate_val = 0.1f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&learning_rate_val, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(learning_rate_val) || std::isinf(learning_rate_val)) {
                learning_rate_val = 0.1f;
            }
        }
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_val);

        int pruning_mode_val = 0;
        if (offset < size) {
            pruning_mode_val = data[offset++] % 3;
        }

        auto update_op = tensorflow::ops::Raw(
            root.WithOpName("BoostedTreesUpdateEnsemble"),
            "BoostedTreesUpdateEnsemble",
            {tree_ensemble_handle.output, feature_ids, 
             tensorflow::InputList(node_ids_list), 
             tensorflow::InputList(gains_list), 
             tensorflow::InputList(thresholds_list), 
             tensorflow::InputList(left_node_contribs_list), 
             tensorflow::InputList(right_node_contribs_list), 
             max_depth, learning_rate},
            {tensorflow::DT_RESOURCE},
            tensorflow::ops::Raw::Attrs()
                .Set("pruning_mode", pruning_mode_val)
                .Set("num_features", feature_ids_tensor.dim_size(0))
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({update_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        return -1;
    }

    return 0;
}