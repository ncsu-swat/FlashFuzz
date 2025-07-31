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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_STRING;
            break;
        case 3:
            dtype = tensorflow::DT_RESOURCE;
            break;
        default:
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
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10 + 1;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset] % 26 + 'a');
                        offset++;
                    }
                    flat(i) = tensorflow::tstring(str);
                } else {
                    flat(i) = tensorflow::tstring("default");
                }
            }
            break;
        }
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 100) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto tree_ensemble_handle = tensorflow::ops::VarHandleOp(root, tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        uint8_t num_features = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Input> feature_ids_list;
        std::vector<tensorflow::Input> dimension_ids_list;
        std::vector<tensorflow::Input> node_ids_list;
        std::vector<tensorflow::Input> gains_list;
        std::vector<tensorflow::Input> thresholds_list;
        std::vector<tensorflow::Input> left_node_contribs_list;
        std::vector<tensorflow::Input> right_node_contribs_list;
        std::vector<tensorflow::Input> split_types_list;

        for (uint8_t i = 0; i < num_features; ++i) {
            if (offset >= size - 50) break;
            
            uint8_t feature_rank = 1;
            std::vector<int64_t> feature_shape = {(data[offset++] % 5) + 1};
            tensorflow::Tensor feature_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(feature_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto feature_ids_const = tensorflow::ops::Const(root, feature_ids_tensor);
            feature_ids_list.push_back(feature_ids_const);

            tensorflow::Tensor dimension_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(dimension_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto dimension_ids_const = tensorflow::ops::Const(root, dimension_ids_tensor);
            dimension_ids_list.push_back(dimension_ids_const);

            tensorflow::Tensor node_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(node_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto node_ids_const = tensorflow::ops::Const(root, node_ids_tensor);
            node_ids_list.push_back(node_ids_const);

            tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto gains_const = tensorflow::ops::Const(root, gains_tensor);
            gains_list.push_back(gains_const);

            tensorflow::Tensor thresholds_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(thresholds_tensor, tensorflow::DT_INT32, data, offset, size);
            auto thresholds_const = tensorflow::ops::Const(root, thresholds_tensor);
            thresholds_list.push_back(thresholds_const);

            std::vector<int64_t> contrib_shape = {feature_shape[0], (data[offset++] % 3) + 1};
            tensorflow::Tensor left_contribs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(contrib_shape));
            fillTensorWithDataByType(left_contribs_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto left_contribs_const = tensorflow::ops::Const(root, left_contribs_tensor);
            left_node_contribs_list.push_back(left_contribs_const);

            tensorflow::Tensor right_contribs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(contrib_shape));
            fillTensorWithDataByType(right_contribs_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto right_contribs_const = tensorflow::ops::Const(root, right_contribs_tensor);
            right_node_contribs_list.push_back(right_contribs_const);

            tensorflow::Tensor split_types_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(feature_shape));
            fillTensorWithDataByType(split_types_tensor, tensorflow::DT_STRING, data, offset, size);
            auto split_types_const = tensorflow::ops::Const(root, split_types_tensor);
            split_types_list.push_back(split_types_const);
        }

        int32_t max_depth_val = (offset < size) ? static_cast<int32_t>(data[offset++] % 10 + 1) : 5;
        auto max_depth = tensorflow::ops::Const(root, max_depth_val);

        float learning_rate_val = (offset < size) ? static_cast<float>(data[offset++]) / 255.0f : 0.1f;
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_val);

        int32_t pruning_mode_val = (offset < size) ? static_cast<int32_t>(data[offset++] % 3) : 0;
        auto pruning_mode = tensorflow::ops::Const(root, pruning_mode_val);

        tensorflow::Operation update_op = tensorflow::Operation();
        tensorflow::Status status = tensorflow::ops::BoostedTreesUpdateEnsembleV2(
            root.WithOpName("BoostedTreesUpdateEnsembleV2"),
            tree_ensemble_handle,
            feature_ids_list,
            dimension_ids_list,
            node_ids_list,
            gains_list,
            thresholds_list,
            left_node_contribs_list,
            right_node_contribs_list,
            split_types_list,
            max_depth,
            learning_rate,
            pruning_mode,
            tensorflow::ops::BoostedTreesUpdateEnsembleV2::Attrs().LogitsDimension(1),
            &update_op);

        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        status = session.Run({update_op}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}