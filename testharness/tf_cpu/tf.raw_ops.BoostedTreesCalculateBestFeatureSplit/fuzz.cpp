#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor node_id_range_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto node_id_range_flat = node_id_range_tensor.flat<int32_t>();
        if (offset + 8 <= size) {
            int32_t start, end;
            std::memcpy(&start, data + offset, 4);
            offset += 4;
            std::memcpy(&end, data + offset, 4);
            offset += 4;
            start = std::abs(start) % 10;
            end = start + (std::abs(end) % 5) + 1;
            node_id_range_flat(0) = start;
            node_id_range_flat(1) = end;
        } else {
            node_id_range_flat(0) = 0;
            node_id_range_flat(1) = 2;
        }

        uint8_t stats_rank = parseRank(data[offset % size]);
        offset++;
        if (stats_rank < 4) stats_rank = 4;
        std::vector<int64_t> stats_shape = parseShape(data, offset, size, stats_rank);
        if (stats_shape.size() < 4) {
            stats_shape = {2, 1, 3, 2};
        }
        tensorflow::Tensor stats_summary_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(stats_shape));
        fillTensorWithDataByType(stats_summary_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + 4 <= size) {
            float l1_val;
            std::memcpy(&l1_val, data + offset, 4);
            offset += 4;
            l1_tensor.scalar<float>()() = std::abs(l1_val);
        } else {
            l1_tensor.scalar<float>()() = 0.1f;
        }

        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + 4 <= size) {
            float l2_val;
            std::memcpy(&l2_val, data + offset, 4);
            offset += 4;
            l2_tensor.scalar<float>()() = std::abs(l2_val);
        } else {
            l2_tensor.scalar<float>()() = 0.1f;
        }

        tensorflow::Tensor tree_complexity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + 4 <= size) {
            float tree_complexity_val;
            std::memcpy(&tree_complexity_val, data + offset, 4);
            offset += 4;
            tree_complexity_tensor.scalar<float>()() = std::abs(tree_complexity_val);
        } else {
            tree_complexity_tensor.scalar<float>()() = 0.0f;
        }

        tensorflow::Tensor min_node_weight_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset + 4 <= size) {
            float min_node_weight_val;
            std::memcpy(&min_node_weight_val, data + offset, 4);
            offset += 4;
            min_node_weight_tensor.scalar<float>()() = std::abs(min_node_weight_val);
        } else {
            min_node_weight_tensor.scalar<float>()() = 1.0f;
        }

        int logits_dimension = 1;
        if (offset < size) {
            logits_dimension = (data[offset] % 5) + 1;
            offset++;
        }

        std::string split_type = "inequality";
        if (offset < size && data[offset] % 2 == 1) {
            split_type = "equality";
        }

        auto node_id_range_op = tensorflow::ops::Const(root, node_id_range_tensor);
        auto stats_summary_op = tensorflow::ops::Const(root, stats_summary_tensor);
        auto l1_op = tensorflow::ops::Const(root, l1_tensor);
        auto l2_op = tensorflow::ops::Const(root, l2_tensor);
        auto tree_complexity_op = tensorflow::ops::Const(root, tree_complexity_tensor);
        auto min_node_weight_op = tensorflow::ops::Const(root, min_node_weight_tensor);

        auto boosted_trees_op = tensorflow::ops::Raw(
            root.WithOpName("BoostedTreesCalculateBestFeatureSplit"),
            {node_id_range_op, stats_summary_op, l1_op, l2_op, tree_complexity_op, min_node_weight_op},
            {tensorflow::DT_INT32, tensorflow::DT_FLOAT, tensorflow::DT_INT32, tensorflow::DT_FLOAT, 
             tensorflow::DT_FLOAT, tensorflow::DT_FLOAT, tensorflow::DT_BOOL},
            tensorflow::ops::Raw::Attrs()
                .SetAttr("logits_dimension", logits_dimension)
                .SetAttr("split_type", split_type)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({boosted_trees_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}