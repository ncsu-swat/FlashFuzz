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
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_LIST_SIZE 5

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataTypeInt(uint8_t selector) {
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

tensorflow::DataType parseDataTypeFloat(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
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
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        uint8_t list_size_byte = data[offset++];
        size_t list_size = (list_size_byte % MAX_LIST_SIZE) + 1;

        tensorflow::DataType int_dtype = parseDataTypeInt(data[offset++]);
        tensorflow::DataType float_dtype = parseDataTypeFloat(data[offset++]);

        std::vector<tensorflow::Input> sample_splits;
        std::vector<tensorflow::Input> embedding_indices;
        std::vector<tensorflow::Input> aggregation_weights;
        std::vector<int> table_ids;

        for (size_t i = 0; i < list_size; ++i) {
            if (offset >= size) break;

            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);

            tensorflow::Tensor sample_split_tensor(int_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(sample_split_tensor, int_dtype, data, offset, size);
            sample_splits.push_back(tensorflow::ops::Const(root, sample_split_tensor));

            tensorflow::Tensor embedding_indices_tensor(int_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(embedding_indices_tensor, int_dtype, data, offset, size);
            embedding_indices.push_back(tensorflow::ops::Const(root, embedding_indices_tensor));

            tensorflow::Tensor aggregation_weights_tensor(float_dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(aggregation_weights_tensor, float_dtype, data, offset, size);
            aggregation_weights.push_back(tensorflow::ops::Const(root, aggregation_weights_tensor));

            if (offset < size) {
                int table_id = static_cast<int>(data[offset++] % 10);
                table_ids.push_back(table_id);
            } else {
                table_ids.push_back(0);
            }
        }

        std::string mode_override_str = "unspecified";
        tensorflow::Tensor mode_override_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        mode_override_tensor.scalar<tensorflow::tstring>()() = mode_override_str;
        auto mode_override = tensorflow::ops::Const(root, mode_override_tensor);

        int device_ordinal = -1;
        if (offset < size) {
            device_ordinal = static_cast<int>(data[offset++]) - 1;
        }

        std::vector<std::string> combiners;
        std::vector<int> max_sequence_lengths;
        std::vector<int> num_features;

        for (size_t i = 0; i < list_size; ++i) {
            combiners.push_back("sum");
            max_sequence_lengths.push_back(100);
            num_features.push_back(1);
        }

        // Use raw_ops instead of ops::EnqueueTPUEmbeddingRaggedTensorBatch
        auto enqueue_op = tensorflow::ops::StatefulPartitionedCall(
            root,
            std::vector<tensorflow::Input>(),
            std::vector<tensorflow::DataType>(),
            std::vector<tensorflow::DataType>(),
            tensorflow::NameAttrList{
                "EnqueueTPUEmbeddingRaggedTensorBatch",
                {
                    {"sample_splits", tensorflow::AttrValue()},
                    {"embedding_indices", tensorflow::AttrValue()},
                    {"aggregation_weights", tensorflow::AttrValue()},
                    {"mode_override", tensorflow::AttrValue()},
                    {"table_ids", tensorflow::AttrValue()},
                    {"device_ordinal", tensorflow::AttrValue()},
                    {"combiners", tensorflow::AttrValue()},
                    {"max_sequence_lengths", tensorflow::AttrValue()},
                    {"num_features", tensorflow::AttrValue()}
                }
            }
        );

        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({enqueue_op}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
