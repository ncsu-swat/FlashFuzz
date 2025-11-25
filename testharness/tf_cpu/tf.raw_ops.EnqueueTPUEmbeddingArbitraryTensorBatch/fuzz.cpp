#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/node_builder.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_LIST_SIZE 5

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataTypeForIndices(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
    }
    return tensorflow::DT_INT32;
}

tensorflow::DataType parseDataTypeForWeights(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_FLOAT;
        case 1:
            return tensorflow::DT_DOUBLE;
    }
    return tensorflow::DT_FLOAT;
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

        tensorflow::DataType sample_dtype = parseDataTypeForIndices(data[offset++]);
        tensorflow::DataType embedding_dtype = parseDataTypeForIndices(data[offset++]);
        tensorflow::DataType weights_dtype = parseDataTypeForWeights(data[offset++]);

        std::vector<tensorflow::Output> sample_indices_list;
        std::vector<tensorflow::Output> embedding_indices_list;
        std::vector<tensorflow::Output> aggregation_weights_list;

        for (size_t i = 0; i < list_size; ++i) {
            if (offset >= size) break;

            uint8_t indices_rank = (offset < size) ? parseRank(data[offset++]) : 1;
            std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);

            tensorflow::Tensor sample_indices_tensor(sample_dtype, tensorflow::TensorShape(indices_shape));
            fillTensorWithDataByType(sample_indices_tensor, sample_dtype, data, offset, size);

            tensorflow::Tensor embedding_indices_tensor(embedding_dtype, tensorflow::TensorShape(indices_shape));
            fillTensorWithDataByType(embedding_indices_tensor, embedding_dtype, data, offset, size);

            tensorflow::Tensor aggregation_weights_tensor(weights_dtype, tensorflow::TensorShape(indices_shape));
            fillTensorWithDataByType(aggregation_weights_tensor, weights_dtype, data, offset, size);

            sample_indices_list.push_back(tensorflow::ops::Const(root, sample_indices_tensor));
            embedding_indices_list.push_back(tensorflow::ops::Const(root, embedding_indices_tensor));
            aggregation_weights_list.push_back(tensorflow::ops::Const(root, aggregation_weights_tensor));
        }

        if (sample_indices_list.empty()) {
            return 0;
        }

        std::string mode_override_str = "unspecified";
        if (offset < size) {
            uint8_t mode_selector = data[offset++];
            switch (mode_selector % 4) {
                case 0: mode_override_str = "unspecified"; break;
                case 1: mode_override_str = "inference"; break;
                case 2: mode_override_str = "training"; break;
                case 3: mode_override_str = "backward_pass_only"; break;
            }
        }

        tensorflow::Tensor mode_override_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        mode_override_tensor.scalar<tensorflow::tstring>()() = mode_override_str;
        auto mode_override = tensorflow::ops::Const(root, mode_override_tensor);

        int device_ordinal = -1;
        if (offset < size) {
            device_ordinal = static_cast<int>(data[offset++] % 8) - 1;
        }

        std::vector<std::string> combiners(sample_indices_list.size(), "sum");
        for (size_t i = 0; i < combiners.size() && offset < size; ++i) {
            switch (data[offset++] % 3) {
                case 0: combiners[i] = "mean"; break;
                case 1: combiners[i] = "sum"; break;
                case 2: combiners[i] = "sqrtn"; break;
            }
        }

        auto to_node_out_list = [](const std::vector<tensorflow::Output>& outputs) {
            std::vector<tensorflow::NodeBuilder::NodeOut> node_outs;
            node_outs.reserve(outputs.size());
            for (const auto& output : outputs) {
                node_outs.emplace_back(output.node(), output.index());
            }
            return node_outs;
        };

        auto sample_inputs = to_node_out_list(sample_indices_list);
        auto embedding_inputs = to_node_out_list(embedding_indices_list);
        auto weight_inputs = to_node_out_list(aggregation_weights_list);

        tensorflow::Node* enqueue_node = nullptr;
        auto builder = tensorflow::NodeBuilder("EnqueueTPUEmbeddingArbitraryTensorBatch",
                                               "EnqueueTPUEmbeddingArbitraryTensorBatch")
                           .Input(sample_inputs)
                           .Input(embedding_inputs)
                           .Input(weight_inputs)
                           .Input(tensorflow::NodeBuilder::NodeOut(mode_override.node()))
                           .Attr("T1", sample_dtype)
                           .Attr("T2", embedding_dtype)
                           .Attr("T3", weights_dtype)
                           .Attr("N", static_cast<int>(sample_indices_list.size()))
                           .Attr("device_ordinal", device_ordinal)
                           .Attr("combiners", combiners);

        tensorflow::Status status = builder.Finalize(root.graph(), &enqueue_node);
        if (status.ok()) {
            tensorflow::ClientSession session(root);
            tensorflow::Operation enqueue_op(enqueue_node);
            status = session.Run({}, {}, {enqueue_op}, nullptr);
        }
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
