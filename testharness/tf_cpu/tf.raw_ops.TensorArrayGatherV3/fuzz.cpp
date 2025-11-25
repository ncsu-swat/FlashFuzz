#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t*, size_t) {
    std::cerr << message << std::endl;
}
}  // namespace tf_fuzzer_utils

constexpr int kMaxRank = 4;
constexpr int kMinDim = 1;
constexpr int kMaxDim = 6;

tensorflow::DataType pickDataType(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return tensorflow::DT_FLOAT;
        case 1:
            return tensorflow::DT_INT32;
        default:
            return tensorflow::DT_INT64;
    }
}

uint8_t parseRank(uint8_t byte) {
    return byte % (kMaxRank + 1);
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t size, uint8_t rank) {
    std::vector<int64_t> dims;
    dims.reserve(rank);

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof(int64_t) <= size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            raw_dim = std::abs(raw_dim);
            dims.push_back(kMinDim + static_cast<int64_t>(raw_dim % (kMaxDim - kMinDim + 1)));
        } else {
            dims.push_back(kMinDim);
        }
    }

    return dims;
}

template <typename T>
void fillTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t size) {
    auto flat = tensor.flat<T>();
    const size_t elem_size = sizeof(T);
    for (int i = 0; i < flat.size(); ++i) {
        if (offset + elem_size <= size) {
            T value;
            std::memcpy(&value, data + offset, elem_size);
            offset += elem_size;
            flat(i) = value;
        } else {
            flat(i) = T{};
        }
    }
}

void fillByType(tensorflow::Tensor& tensor, tensorflow::DataType dtype, const uint8_t* data, size_t& offset, size_t size) {
    switch (dtype) {
        case tensorflow::DT_FLOAT:
            fillTensor<float>(tensor, data, offset, size);
            break;
        case tensorflow::DT_INT32:
            fillTensor<int32_t>(tensor, data, offset, size);
            break;
        case tensorflow::DT_INT64:
            fillTensor<int64_t>(tensor, data, offset, size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    size_t offset = 0;
    auto dtype = pickDataType(data[offset++]);
    uint8_t rank = parseRank(data[offset++]);
    std::vector<int64_t> element_dims = parseShape(data, offset, size, rank);

    int32_t array_size = 4;
    if (offset + sizeof(int32_t) <= size) {
        std::memcpy(&array_size, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        array_size = std::max<int32_t>(1, std::abs(array_size % 16));
    }

    int32_t num_indices = 1;
    if (offset < size) {
        num_indices = (data[offset++] % array_size) + 1;
    }

    tensorflow::TensorShape element_shape;
    for (int64_t dim : element_dims) {
        element_shape.AddDim(dim);
    }
    tensorflow::Tensor value_tensor(dtype, element_shape);
    fillByType(value_tensor, dtype, data, offset, size);

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    auto size_const = tensorflow::ops::Const(root.WithOpName("tensor_array_size"), array_size);
    auto value_const = tensorflow::ops::Const(root.WithOpName("tensor_array_value"), value_tensor);

    tensorflow::Node* tensor_array_node = nullptr;
    tensorflow::Status status = tensorflow::NodeBuilder("TensorArrayV3Node", "TensorArrayV3")
                                    .Input(size_const.node())
                                    .Attr("dtype", dtype)
                                    .Attr("element_shape", element_shape)
                                    .Finalize(root.graph(), &tensor_array_node);
    if (!status.ok()) {
        tf_fuzzer_utils::logError("Failed to create TensorArrayV3 node: " + status.ToString(), data, size);
        return 0;
    }

    tensorflow::Output flow_out(tensor_array_node, 1);
    for (int32_t i = 0; i < array_size; ++i) {
        auto index_const = tensorflow::ops::Const(root.WithOpName("tensor_array_index_" + std::to_string(i)), i);

        tensorflow::Node* write_node = nullptr;
        status = tensorflow::NodeBuilder("TensorArrayWriteV3Node_" + std::to_string(i), "TensorArrayWriteV3")
                     .Input(tensorflow::NodeBuilder::NodeOut(tensor_array_node, 0))
                     .Input(index_const.node())
                     .Input(value_const.node())
                     .Input(tensorflow::NodeBuilder::NodeOut(flow_out.node(), flow_out.index()))
                     .Attr("T", dtype)
                     .Finalize(root.graph(), &write_node);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create TensorArrayWriteV3 node: " + status.ToString(), data, size);
            return 0;
        }
        flow_out = tensorflow::Output(write_node, 0);
    }

    tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({num_indices}));
    auto indices_flat = indices_tensor.flat<int32_t>();
    for (int32_t i = 0; i < num_indices; ++i) {
        indices_flat(i) = i % array_size;
    }
    auto indices_const = tensorflow::ops::Const(root.WithOpName("tensor_array_indices"), indices_tensor);

    tensorflow::Node* gather_node = nullptr;
    status = tensorflow::NodeBuilder("TensorArrayGatherV3Node", "TensorArrayGatherV3")
                 .Input(tensorflow::NodeBuilder::NodeOut(tensor_array_node, 0))
                 .Input(indices_const.node())
                 .Input(tensorflow::NodeBuilder::NodeOut(flow_out.node(), flow_out.index()))
                 .Attr("dtype", dtype)
                 .Attr("element_shape", element_shape)
                 .Finalize(root.graph(), &gather_node);
    if (!status.ok()) {
        tf_fuzzer_utils::logError("Failed to create TensorArrayGatherV3 node: " + status.ToString(), data, size);
        return 0;
    }

    tensorflow::ClientSession session(root);
    std::vector<tensorflow::Tensor> outputs;
    status = session.Run({tensorflow::Output(gather_node, 0)}, &outputs);
    if (!status.ok()) {
        tf_fuzzer_utils::logError("Session run failed: " + status.ToString(), data, size);
    }

    return 0;
}
