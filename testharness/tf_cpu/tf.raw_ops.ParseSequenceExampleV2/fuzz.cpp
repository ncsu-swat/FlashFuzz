#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
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
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_STRING;
            break;
    }
    return dtype;
}

tensorflow::DataType parseRaggedSplitType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
            uint8_t str_len = data[offset] % 20;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset]);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("");
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
        uint8_t serialized_rank = parseRank(data[offset++]);
        std::vector<int64_t> serialized_shape = parseShape(data, offset, size, serialized_rank);
        tensorflow::Tensor serialized_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(serialized_shape));
        fillStringTensor(serialized_tensor, data, offset, size);

        uint8_t debug_name_rank = parseRank(data[offset++]);
        std::vector<int64_t> debug_name_shape = parseShape(data, offset, size, debug_name_rank);
        tensorflow::Tensor debug_name_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(debug_name_shape));
        fillStringTensor(debug_name_tensor, data, offset, size);

        uint8_t context_sparse_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> context_sparse_keys_shape = parseShape(data, offset, size, context_sparse_keys_rank);
        tensorflow::Tensor context_sparse_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(context_sparse_keys_shape));
        fillStringTensor(context_sparse_keys_tensor, data, offset, size);

        uint8_t context_dense_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> context_dense_keys_shape = parseShape(data, offset, size, context_dense_keys_rank);
        tensorflow::Tensor context_dense_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(context_dense_keys_shape));
        fillStringTensor(context_dense_keys_tensor, data, offset, size);

        uint8_t context_ragged_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> context_ragged_keys_shape = parseShape(data, offset, size, context_ragged_keys_rank);
        tensorflow::Tensor context_ragged_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(context_ragged_keys_shape));
        fillStringTensor(context_ragged_keys_tensor, data, offset, size);

        uint8_t feature_list_sparse_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> feature_list_sparse_keys_shape = parseShape(data, offset, size, feature_list_sparse_keys_rank);
        tensorflow::Tensor feature_list_sparse_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(feature_list_sparse_keys_shape));
        fillStringTensor(feature_list_sparse_keys_tensor, data, offset, size);

        uint8_t feature_list_dense_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> feature_list_dense_keys_shape = parseShape(data, offset, size, feature_list_dense_keys_rank);
        tensorflow::Tensor feature_list_dense_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(feature_list_dense_keys_shape));
        fillStringTensor(feature_list_dense_keys_tensor, data, offset, size);

        uint8_t feature_list_ragged_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> feature_list_ragged_keys_shape = parseShape(data, offset, size, feature_list_ragged_keys_rank);
        tensorflow::Tensor feature_list_ragged_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(feature_list_ragged_keys_shape));
        fillStringTensor(feature_list_ragged_keys_tensor, data, offset, size);

        uint8_t feature_list_dense_missing_rank = parseRank(data[offset++]);
        std::vector<int64_t> feature_list_dense_missing_shape = parseShape(data, offset, size, feature_list_dense_missing_rank);
        tensorflow::Tensor feature_list_dense_missing_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape(feature_list_dense_missing_shape));
        fillTensorWithDataByType(feature_list_dense_missing_tensor, tensorflow::DT_BOOL, data, offset, size);

        tensorflow::DataType context_dense_default_dtype = parseDataType(data[offset++]);
        uint8_t context_dense_default_rank = parseRank(data[offset++]);
        std::vector<int64_t> context_dense_default_shape = parseShape(data, offset, size, context_dense_default_rank);
        tensorflow::Tensor context_dense_default_tensor(context_dense_default_dtype, tensorflow::TensorShape(context_dense_default_shape));
        fillTensorWithDataByType(context_dense_default_tensor, context_dense_default_dtype, data, offset, size);

        auto serialized_input = tensorflow::ops::Const(root, serialized_tensor);
        auto debug_name_input = tensorflow::ops::Const(root, debug_name_tensor);
        auto context_sparse_keys_input = tensorflow::ops::Const(root, context_sparse_keys_tensor);
        auto context_dense_keys_input = tensorflow::ops::Const(root, context_dense_keys_tensor);
        auto context_ragged_keys_input = tensorflow::ops::Const(root, context_ragged_keys_tensor);
        auto feature_list_sparse_keys_input = tensorflow::ops::Const(root, feature_list_sparse_keys_tensor);
        auto feature_list_dense_keys_input = tensorflow::ops::Const(root, feature_list_dense_keys_tensor);
        auto feature_list_ragged_keys_input = tensorflow::ops::Const(root, feature_list_ragged_keys_tensor);
        auto feature_list_dense_missing_input = tensorflow::ops::Const(root, feature_list_dense_missing_tensor);
        auto context_dense_default_input = tensorflow::ops::Const(root, context_dense_default_tensor);

        tensorflow::ops::ParseSequenceExampleV2::Attrs attrs;
        attrs = attrs.NcontextSparse(1);
        attrs = attrs.ContextSparseTypes({tensorflow::DT_FLOAT});
        attrs = attrs.ContextRaggedValueTypes({tensorflow::DT_FLOAT});
        attrs = attrs.ContextRaggedSplitTypes({tensorflow::DT_INT64});
        attrs = attrs.ContextDenseShapes({{1}});
        attrs = attrs.NfeatureListSparse(1);
        attrs = attrs.NfeatureListDense(1);
        attrs = attrs.FeatureListDenseTypes({tensorflow::DT_FLOAT});
        attrs = attrs.FeatureListSparseTypes({tensorflow::DT_FLOAT});
        attrs = attrs.FeatureListRaggedValueTypes({tensorflow::DT_FLOAT});
        attrs = attrs.FeatureListRaggedSplitTypes({tensorflow::DT_INT64});
        attrs = attrs.FeatureListDenseShapes({{1}});

        auto parse_op = tensorflow::ops::ParseSequenceExampleV2(
            root,
            serialized_input,
            debug_name_input,
            context_sparse_keys_input,
            context_dense_keys_input,
            context_ragged_keys_input,
            feature_list_sparse_keys_input,
            feature_list_dense_keys_input,
            feature_list_ragged_keys_input,
            feature_list_dense_missing_input,
            {context_dense_default_input},
            attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({}, {
            parse_op.context_sparse_indices[0],
            parse_op.context_sparse_values[0],
            parse_op.context_sparse_shapes[0],
            parse_op.context_dense_values[0],
            parse_op.feature_list_sparse_indices[0],
            parse_op.feature_list_sparse_values[0],
            parse_op.feature_list_sparse_shapes[0],
            parse_op.feature_list_dense_values[0],
            parse_op.feature_list_dense_lengths[0]
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
