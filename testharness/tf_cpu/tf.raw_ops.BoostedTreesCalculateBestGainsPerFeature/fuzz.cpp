#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor node_id_range_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto node_id_range_flat = node_id_range_tensor.flat<int32_t>();
        if (offset + 2 * sizeof(int32_t) <= size) {
            int32_t start, end;
            std::memcpy(&start, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            std::memcpy(&end, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            start = std::abs(start) % 10;
            end = start + 1 + (std::abs(end) % 5);
            
            node_id_range_flat(0) = start;
            node_id_range_flat(1) = end;
        } else {
            node_id_range_flat(0) = 0;
            node_id_range_flat(1) = 1;
        }

        uint8_t num_features = 1;
        if (offset < size) {
            num_features = 1 + (data[offset] % 3);
            offset++;
        }

        std::vector<tensorflow::Output> stats_summary_list;
        for (uint8_t i = 0; i < num_features; ++i) {
            uint8_t rank = 3;
            std::vector<int64_t> shape = {5, 10, 2};
            
            tensorflow::Tensor stats_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(stats_tensor, tensorflow::DT_FLOAT, data, offset, size);
            
            auto stats_input = tensorflow::ops::Const(root, stats_tensor);
            stats_summary_list.push_back(stats_input);
        }

        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto l1_flat = l1_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float l1_val;
            std::memcpy(&l1_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l1_flat(0) = std::abs(l1_val);
        } else {
            l1_flat(0) = 0.1f;
        }

        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto l2_flat = l2_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float l2_val;
            std::memcpy(&l2_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l2_flat(0) = std::abs(l2_val);
        } else {
            l2_flat(0) = 0.1f;
        }

        tensorflow::Tensor tree_complexity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto tree_complexity_flat = tree_complexity_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float tree_complexity_val;
            std::memcpy(&tree_complexity_val, data + offset, sizeof(float));
            offset += sizeof(float);
            tree_complexity_flat(0) = std::abs(tree_complexity_val);
        } else {
            tree_complexity_flat(0) = 0.1f;
        }

        tensorflow::Tensor min_node_weight_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto min_node_weight_flat = min_node_weight_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float min_node_weight_val;
            std::memcpy(&min_node_weight_val, data + offset, sizeof(float));
            offset += sizeof(float);
            min_node_weight_flat(0) = std::abs(min_node_weight_val);
        } else {
            min_node_weight_flat(0) = 1.0f;
        }

        int max_splits = 5;

        auto node_id_range_input = tensorflow::ops::Const(root, node_id_range_tensor);
        auto l1_input = tensorflow::ops::Const(root, l1_tensor);
        auto l2_input = tensorflow::ops::Const(root, l2_tensor);
        auto tree_complexity_input = tensorflow::ops::Const(root, tree_complexity_tensor);
        auto min_node_weight_input = tensorflow::ops::Const(root, min_node_weight_tensor);

        // Use raw_ops directly instead of the missing boosted_trees_ops.h
        std::vector<tensorflow::Output> outputs;
        tensorflow::ops::BoostedTreesCalculateBestGainsPerFeature(
            root.WithOpName("BoostedTreesCalculateBestGainsPerFeature"),
            node_id_range_input,
            stats_summary_list,
            l1_input,
            l2_input,
            tree_complexity_input,
            min_node_weight_input,
            tensorflow::ops::BoostedTreesCalculateBestGainsPerFeature::Attrs().MaxSplits(max_splits),
            &outputs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        tensorflow::Status status = session.Run({outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]}, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
