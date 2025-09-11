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
            uint8_t str_len = data[offset] % 32;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor serialized(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(serialized, data, offset, size);

        tensorflow::Tensor feature_list_dense_missing_assumed_empty(tensorflow::DT_STRING, tensorflow::TensorShape({1}));
        fillStringTensor(feature_list_dense_missing_assumed_empty, data, offset, size);

        uint8_t num_context_sparse = (offset < size) ? data[offset++] % 3 : 0;
        uint8_t num_context_dense = (offset < size) ? data[offset++] % 3 : 0;
        uint8_t num_feature_list_sparse = (offset < size) ? data[offset++] % 3 : 0;
        uint8_t num_feature_list_dense = (offset < size) ? data[offset++] % 3 : 0;

        std::vector<tensorflow::Input> context_sparse_keys;
        std::vector<tensorflow::DataType> context_sparse_types;
        for (uint8_t i = 0; i < num_context_sparse; ++i) {
            tensorflow::Tensor key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(key_tensor, data, offset, size);
            context_sparse_keys.push_back(tensorflow::Input(key_tensor));
            
            tensorflow::DataType dtype = parseDataType((offset < size) ? data[offset++] : 0);
            context_sparse_types.push_back(dtype);
        }

        std::vector<tensorflow::Input> context_dense_keys;
        std::vector<tensorflow::Input> context_dense_defaults;
        std::vector<tensorflow::PartialTensorShape> context_dense_shapes;
        for (uint8_t i = 0; i < num_context_dense; ++i) {
            tensorflow::Tensor key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(key_tensor, data, offset, size);
            context_dense_keys.push_back(tensorflow::Input(key_tensor));

            tensorflow::DataType dtype = parseDataType((offset < size) ? data[offset++] : 0);
            uint8_t rank = parseRank((offset < size) ? data[offset++] : 0);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            context_dense_shapes.push_back(tensorflow::PartialTensorShape(tensor_shape));

            tensorflow::Tensor default_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(default_tensor, dtype, data, offset, size);
            context_dense_defaults.push_back(tensorflow::Input(default_tensor));
        }

        std::vector<tensorflow::Input> feature_list_sparse_keys;
        std::vector<tensorflow::DataType> feature_list_sparse_types;
        for (uint8_t i = 0; i < num_feature_list_sparse; ++i) {
            tensorflow::Tensor key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(key_tensor, data, offset, size);
            feature_list_sparse_keys.push_back(tensorflow::Input(key_tensor));
            
            tensorflow::DataType dtype = parseDataType((offset < size) ? data[offset++] : 0);
            feature_list_sparse_types.push_back(dtype);
        }

        std::vector<tensorflow::Input> feature_list_dense_keys;
        std::vector<tensorflow::DataType> feature_list_dense_types;
        std::vector<tensorflow::PartialTensorShape> feature_list_dense_shapes;
        for (uint8_t i = 0; i < num_feature_list_dense; ++i) {
            tensorflow::Tensor key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(key_tensor, data, offset, size);
            feature_list_dense_keys.push_back(tensorflow::Input(key_tensor));

            tensorflow::DataType dtype = parseDataType((offset < size) ? data[offset++] : 0);
            feature_list_dense_types.push_back(dtype);
            
            uint8_t rank = parseRank((offset < size) ? data[offset++] : 0);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            feature_list_dense_shapes.push_back(tensorflow::PartialTensorShape(tensor_shape));
        }

        tensorflow::Tensor debug_name(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(debug_name, data, offset, size);

        auto parse_op = tensorflow::ops::ParseSingleSequenceExample(
            root,
            tensorflow::Input(serialized),
            tensorflow::Input(feature_list_dense_missing_assumed_empty),
            context_sparse_keys,
            context_dense_keys,
            feature_list_sparse_keys,
            feature_list_dense_keys,
            context_dense_defaults,
            tensorflow::Input(debug_name));

        // Set attributes
        if (!context_sparse_types.empty()) {
            parse_op = parse_op.ContextSparseTypes(context_sparse_types);
        }
        if (!feature_list_dense_types.empty()) {
            parse_op = parse_op.FeatureListDenseTypes(feature_list_dense_types);
        }
        if (!context_dense_shapes.empty()) {
            parse_op = parse_op.ContextDenseShapes(context_dense_shapes);
        }
        if (!feature_list_sparse_types.empty()) {
            parse_op = parse_op.FeatureListSparseTypes(feature_list_sparse_types);
        }
        if (!feature_list_dense_shapes.empty()) {
            parse_op = parse_op.FeatureListDenseShapes(feature_list_dense_shapes);
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::vector<tensorflow::Output> fetch_outputs;
        for (const auto& output : parse_op.context_sparse_indices) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.context_sparse_values) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.context_sparse_shapes) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.context_dense_values) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.feature_list_sparse_indices) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.feature_list_sparse_values) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.feature_list_sparse_shapes) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_op.feature_list_dense_values) {
            fetch_outputs.push_back(output);
        }

        tensorflow::Status status = session.Run(fetch_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
