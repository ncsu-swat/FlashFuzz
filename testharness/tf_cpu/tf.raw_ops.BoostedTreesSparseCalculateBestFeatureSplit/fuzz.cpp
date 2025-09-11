#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t node_id_range_rank = parseRank(data[offset++]);
        if (node_id_range_rank != 1) node_id_range_rank = 1;
        std::vector<int64_t> node_id_range_shape = {2};
        tensorflow::Tensor node_id_range_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(node_id_range_shape));
        fillTensorWithDataByType(node_id_range_tensor, tensorflow::DT_INT32, data, offset, size);
        auto node_id_range = tensorflow::ops::Const(root, node_id_range_tensor);

        uint8_t stats_summary_indices_rank = parseRank(data[offset++]);
        if (stats_summary_indices_rank != 2) stats_summary_indices_rank = 2;
        std::vector<int64_t> stats_summary_indices_shape = parseShape(data, offset, size, stats_summary_indices_rank);
        if (stats_summary_indices_shape.size() >= 2) {
            stats_summary_indices_shape[1] = 4;
        }
        tensorflow::Tensor stats_summary_indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(stats_summary_indices_shape));
        fillTensorWithDataByType(stats_summary_indices_tensor, tensorflow::DT_INT32, data, offset, size);
        auto stats_summary_indices = tensorflow::ops::Const(root, stats_summary_indices_tensor);

        uint8_t stats_summary_values_rank = parseRank(data[offset++]);
        if (stats_summary_values_rank != 1) stats_summary_values_rank = 1;
        std::vector<int64_t> stats_summary_values_shape = parseShape(data, offset, size, stats_summary_values_rank);
        tensorflow::Tensor stats_summary_values_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(stats_summary_values_shape));
        fillTensorWithDataByType(stats_summary_values_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto stats_summary_values = tensorflow::ops::Const(root, stats_summary_values_tensor);

        uint8_t stats_summary_shape_rank = parseRank(data[offset++]);
        if (stats_summary_shape_rank != 1) stats_summary_shape_rank = 1;
        std::vector<int64_t> stats_summary_shape_shape = {4};
        tensorflow::Tensor stats_summary_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(stats_summary_shape_shape));
        fillTensorWithDataByType(stats_summary_shape_tensor, tensorflow::DT_INT32, data, offset, size);
        auto stats_summary_shape = tensorflow::ops::Const(root, stats_summary_shape_tensor);

        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l1_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto l1 = tensorflow::ops::Const(root, l1_tensor);

        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l2_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto l2 = tensorflow::ops::Const(root, l2_tensor);

        tensorflow::Tensor tree_complexity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(tree_complexity_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto tree_complexity = tensorflow::ops::Const(root, tree_complexity_tensor);

        tensorflow::Tensor min_node_weight_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(min_node_weight_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto min_node_weight = tensorflow::ops::Const(root, min_node_weight_tensor);

        int logits_dimension = 1;
        if (offset < size) {
            logits_dimension = std::max(1, static_cast<int>(data[offset++] % 10 + 1));
        }

        // Use raw_ops directly instead of BoostedTreesOps
        auto op_attrs = tensorflow::ops::internal::BoostedTreesSparseCalculateBestFeatureSplit::Attrs().LogitsDimension(logits_dimension);
        auto boosted_trees_op = tensorflow::ops::internal::BoostedTreesSparseCalculateBestFeatureSplit(
            root,
            node_id_range,
            stats_summary_indices,
            stats_summary_values,
            stats_summary_shape,
            l1,
            l2,
            tree_complexity,
            min_node_weight,
            op_attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({boosted_trees_op.node_ids, 
                                                boosted_trees_op.gains,
                                                boosted_trees_op.feature_dimensions,
                                                boosted_trees_op.thresholds,
                                                boosted_trees_op.left_node_contribs,
                                                boosted_trees_op.right_node_contribs,
                                                boosted_trees_op.split_with_default_directions}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
