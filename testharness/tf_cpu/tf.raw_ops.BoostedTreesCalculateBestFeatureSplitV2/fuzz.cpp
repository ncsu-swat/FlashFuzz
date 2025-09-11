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
    switch (selector % 23) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
            dtype = tensorflow::DT_UINT64;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                     size_t& offset, size_t total_size) {
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
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
            end = start + (std::abs(end) % 10) + 1;
            node_id_range_flat(0) = start;
            node_id_range_flat(1) = end;
        } else {
            node_id_range_flat(0) = 0;
            node_id_range_flat(1) = 2;
        }

        uint8_t num_features = 1;
        if (offset < size) {
            num_features = (data[offset] % 3) + 1;
            offset++;
        }

        std::vector<tensorflow::Output> stats_summaries_list;
        for (uint8_t f = 0; f < num_features; ++f) {
            tensorflow::TensorShape stats_shape({2, 1, 3, 2});
            tensorflow::Tensor stats_tensor(tensorflow::DT_FLOAT, stats_shape);
            fillTensorWithDataByType(stats_tensor, tensorflow::DT_FLOAT, data, offset, size);
            
            auto stats_input = tensorflow::ops::Const(root, stats_tensor);
            stats_summaries_list.push_back(stats_input);
        }

        tensorflow::Tensor split_types_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({static_cast<int64_t>(num_features)}));
        fillStringTensor(split_types_tensor, data, offset, size);

        tensorflow::Tensor candidate_feature_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_features)}));
        auto feature_ids_flat = candidate_feature_ids_tensor.flat<int32_t>();
        for (int i = 0; i < num_features; ++i) {
            if (offset + 4 <= size) {
                int32_t id;
                std::memcpy(&id, data + offset, 4);
                offset += 4;
                feature_ids_flat(i) = std::abs(id) % 100;
            } else {
                feature_ids_flat(i) = i;
            }
        }

        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor tree_complexity_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_node_weight_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));

        if (offset + 16 <= size) {
            float l1, l2, tree_comp, min_weight;
            std::memcpy(&l1, data + offset, 4); offset += 4;
            std::memcpy(&l2, data + offset, 4); offset += 4;
            std::memcpy(&tree_comp, data + offset, 4); offset += 4;
            std::memcpy(&min_weight, data + offset, 4); offset += 4;
            
            l1_tensor.scalar<float>()() = std::abs(l1);
            l2_tensor.scalar<float>()() = std::abs(l2);
            tree_complexity_tensor.scalar<float>()() = std::abs(tree_comp);
            min_node_weight_tensor.scalar<float>()() = std::abs(min_weight);
        } else {
            l1_tensor.scalar<float>()() = 0.1f;
            l2_tensor.scalar<float>()() = 0.1f;
            tree_complexity_tensor.scalar<float>()() = 0.1f;
            min_node_weight_tensor.scalar<float>()() = 1.0f;
        }

        int64_t logits_dimension = 1;
        if (offset < size) {
            logits_dimension = (data[offset] % 5) + 1;
            offset++;
        }

        auto node_id_range_input = tensorflow::ops::Const(root, node_id_range_tensor);
        auto split_types_input = tensorflow::ops::Const(root, split_types_tensor);
        auto candidate_feature_ids_input = tensorflow::ops::Const(root, candidate_feature_ids_tensor);
        auto l1_input = tensorflow::ops::Const(root, l1_tensor);
        auto l2_input = tensorflow::ops::Const(root, l2_tensor);
        auto tree_complexity_input = tensorflow::ops::Const(root, tree_complexity_tensor);
        auto min_node_weight_input = tensorflow::ops::Const(root, min_node_weight_tensor);

        // Use raw_ops namespace to access BoostedTreesCalculateBestFeatureSplitV2
        auto boosted_trees_op = tensorflow::ops::BoostedTreesCalculateBestFeatureSplitV2(
            root.WithOpName("BoostedTreesCalculateBestFeatureSplitV2"),
            node_id_range_input,
            stats_summaries_list,
            split_types_input,
            candidate_feature_ids_input,
            l1_input,
            l2_input,
            tree_complexity_input,
            min_node_weight_input,
            logits_dimension
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            boosted_trees_op.node_ids,
            boosted_trees_op.gains,
            boosted_trees_op.feature_ids,
            boosted_trees_op.feature_dimensions,
            boosted_trees_op.thresholds,
            boosted_trees_op.left_node_contribs,
            boosted_trees_op.right_node_contribs,
            boosted_trees_op.split_with_default_directions
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
