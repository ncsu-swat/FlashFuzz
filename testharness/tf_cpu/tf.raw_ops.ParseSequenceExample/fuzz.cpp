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
#include <string>
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
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

        uint8_t num_context_dense = (offset < size) ? data[offset++] % 3 : 0;
        std::vector<tensorflow::Input> context_dense_defaults;
        
        for (uint8_t i = 0; i < num_context_dense; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::Tensor tensor(dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            context_dense_defaults.push_back(tensorflow::Input(tensor));
        }

        std::vector<tensorflow::tstring> feature_list_dense_missing_assumed_empty;
        uint8_t num_missing = (offset < size) ? data[offset++] % 3 : 0;
        for (uint8_t i = 0; i < num_missing; ++i) {
            feature_list_dense_missing_assumed_empty.push_back("missing_key_" + std::to_string(i));
        }

        std::vector<tensorflow::tstring> context_sparse_keys;
        uint8_t num_context_sparse = (offset < size) ? data[offset++] % 3 : 0;
        for (uint8_t i = 0; i < num_context_sparse; ++i) {
            context_sparse_keys.push_back("context_sparse_" + std::to_string(i));
        }

        std::vector<tensorflow::tstring> context_dense_keys;
        for (uint8_t i = 0; i < num_context_dense; ++i) {
            context_dense_keys.push_back("context_dense_" + std::to_string(i));
        }

        std::vector<tensorflow::tstring> feature_list_sparse_keys;
        uint8_t num_feature_list_sparse = (offset < size) ? data[offset++] % 3 : 0;
        for (uint8_t i = 0; i < num_feature_list_sparse; ++i) {
            feature_list_sparse_keys.push_back("feature_list_sparse_" + std::to_string(i));
        }

        std::vector<tensorflow::tstring> feature_list_dense_keys;
        uint8_t num_feature_list_dense = (offset < size) ? data[offset++] % 3 : 0;
        for (uint8_t i = 0; i < num_feature_list_dense; ++i) {
            feature_list_dense_keys.push_back("feature_list_dense_" + std::to_string(i));
        }

        std::vector<tensorflow::DataType> context_sparse_types;
        for (uint8_t i = 0; i < num_context_sparse; ++i) {
            if (offset < size) {
                context_sparse_types.push_back(parseDataType(data[offset++]));
            } else {
                context_sparse_types.push_back(tensorflow::DT_FLOAT);
            }
        }

        std::vector<tensorflow::DataType> feature_list_dense_types;
        for (uint8_t i = 0; i < num_feature_list_dense; ++i) {
            if (offset < size) {
                feature_list_dense_types.push_back(parseDataType(data[offset++]));
            } else {
                feature_list_dense_types.push_back(tensorflow::DT_FLOAT);
            }
        }

        std::vector<tensorflow::PartialTensorShape> context_dense_shapes;
        for (uint8_t i = 0; i < num_context_dense; ++i) {
            context_dense_shapes.push_back(tensorflow::PartialTensorShape({1}));
        }

        std::vector<tensorflow::DataType> feature_list_sparse_types;
        for (uint8_t i = 0; i < num_feature_list_sparse; ++i) {
            if (offset < size) {
                feature_list_sparse_types.push_back(parseDataType(data[offset++]));
            } else {
                feature_list_sparse_types.push_back(tensorflow::DT_FLOAT);
            }
        }

        std::vector<tensorflow::PartialTensorShape> feature_list_dense_shapes;
        for (uint8_t i = 0; i < num_feature_list_dense; ++i) {
            feature_list_dense_shapes.push_back(tensorflow::PartialTensorShape({1}));
        }

        auto attrs = tensorflow::ops::ParseSequenceExample::Attrs()
            .NcontextSparse(num_context_sparse)
            .NcontextDense(num_context_dense)
            .NfeatureListSparse(num_feature_list_sparse)
            .NfeatureListDense(num_feature_list_dense)
            .ContextSparseTypes(context_sparse_types)
            .FeatureListDenseTypes(feature_list_dense_types)
            .ContextDenseShapes(context_dense_shapes)
            .FeatureListSparseTypes(feature_list_sparse_types)
            .FeatureListDenseShapes(feature_list_dense_shapes);

        auto parse_result = tensorflow::ops::ParseSequenceExample(
            root,
            tensorflow::Input(serialized_tensor),
            tensorflow::Input(debug_name_tensor),
            tensorflow::InputList(context_dense_defaults),
            tensorflow::gtl::ArraySlice<tensorflow::tstring>(feature_list_dense_missing_assumed_empty),
            tensorflow::gtl::ArraySlice<tensorflow::tstring>(context_sparse_keys),
            tensorflow::gtl::ArraySlice<tensorflow::tstring>(context_dense_keys),
            tensorflow::gtl::ArraySlice<tensorflow::tstring>(feature_list_sparse_keys),
            tensorflow::gtl::ArraySlice<tensorflow::tstring>(feature_list_dense_keys),
            attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::vector<tensorflow::Output> fetch_outputs;
        for (const auto& output : parse_result.context_sparse_indices) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_result.context_sparse_values) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_result.context_sparse_shapes) {
            fetch_outputs.push_back(output);
        }
        for (const auto& output : parse_result.context_dense_values) {
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