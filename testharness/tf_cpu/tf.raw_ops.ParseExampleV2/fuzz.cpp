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
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
        uint8_t serialized_rank = parseRank(data[offset++]);
        std::vector<int64_t> serialized_shape = parseShape(data, offset, size, serialized_rank);
        tensorflow::Tensor serialized_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(serialized_shape));
        fillStringTensor(serialized_tensor, data, offset, size);

        uint8_t names_rank = parseRank(data[offset++]);
        std::vector<int64_t> names_shape = parseShape(data, offset, size, names_rank);
        tensorflow::Tensor names_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(names_shape));
        fillStringTensor(names_tensor, data, offset, size);

        uint8_t sparse_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> sparse_keys_shape = parseShape(data, offset, size, sparse_keys_rank);
        tensorflow::Tensor sparse_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(sparse_keys_shape));
        fillStringTensor(sparse_keys_tensor, data, offset, size);

        uint8_t dense_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> dense_keys_shape = parseShape(data, offset, size, dense_keys_rank);
        tensorflow::Tensor dense_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(dense_keys_shape));
        fillStringTensor(dense_keys_tensor, data, offset, size);

        uint8_t ragged_keys_rank = parseRank(data[offset++]);
        std::vector<int64_t> ragged_keys_shape = parseShape(data, offset, size, ragged_keys_rank);
        tensorflow::Tensor ragged_keys_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(ragged_keys_shape));
        fillStringTensor(ragged_keys_tensor, data, offset, size);

        int64_t num_dense = 1;
        if (dense_keys_shape.size() > 0) {
            num_dense = dense_keys_shape[0];
        }
        
        std::vector<tensorflow::Input> dense_defaults;
        std::vector<tensorflow::PartialTensorShape> dense_shapes;
        
        for (int64_t i = 0; i < num_dense && offset < size; ++i) {
            tensorflow::DataType default_dtype = parseDataType(data[offset++]);
            uint8_t default_rank = parseRank(data[offset++]);
            std::vector<int64_t> default_shape = parseShape(data, offset, size, default_rank);
            
            tensorflow::Tensor default_tensor(default_dtype, tensorflow::TensorShape(default_shape));
            fillTensorWithDataByType(default_tensor, default_dtype, data, offset, size);
            dense_defaults.push_back(tensorflow::Input(default_tensor));
            
            dense_shapes.push_back(tensorflow::PartialTensorShape(default_shape));
        }

        int64_t num_sparse = 1;
        if (sparse_keys_shape.size() > 0) {
            num_sparse = sparse_keys_shape[0];
        }

        std::vector<tensorflow::DataType> sparse_types;
        for (int64_t i = 0; i < num_sparse && offset < size; ++i) {
            sparse_types.push_back(parseDataType(data[offset++]));
        }

        int64_t num_ragged = 1;
        if (ragged_keys_shape.size() > 0) {
            num_ragged = ragged_keys_shape[0];
        }

        std::vector<tensorflow::DataType> ragged_value_types;
        std::vector<tensorflow::DataType> ragged_split_types;
        for (int64_t i = 0; i < num_ragged && offset < size; ++i) {
            ragged_value_types.push_back(parseDataType(data[offset++]));
            ragged_split_types.push_back(parseRaggedSplitType(data[offset++]));
        }

        auto parse_op = tensorflow::ops::ParseExampleV2(
            root,
            tensorflow::Input(serialized_tensor),
            tensorflow::Input(names_tensor),
            tensorflow::Input(sparse_keys_tensor),
            tensorflow::Input(dense_keys_tensor),
            tensorflow::Input(ragged_keys_tensor),
            tensorflow::InputList(dense_defaults),
            sparse_types,
            dense_shapes,
            ragged_value_types,
            ragged_split_types,
            num_sparse
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::vector<tensorflow::Output> fetch_outputs;
        for (const auto& sparse_idx : parse_op.sparse_indices) {
            fetch_outputs.push_back(sparse_idx);
        }
        for (const auto& sparse_val : parse_op.sparse_values) {
            fetch_outputs.push_back(sparse_val);
        }
        for (const auto& sparse_shape : parse_op.sparse_shapes) {
            fetch_outputs.push_back(sparse_shape);
        }
        for (const auto& dense_val : parse_op.dense_values) {
            fetch_outputs.push_back(dense_val);
        }
        for (const auto& ragged_val : parse_op.ragged_values) {
            fetch_outputs.push_back(ragged_val);
        }
        for (const auto& ragged_split : parse_op.ragged_row_splits) {
            fetch_outputs.push_back(ragged_split);
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
