#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/graph/node_builder.h"
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
    switch (selector % 11) {
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
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
            dtype = tensorflow::DT_UINT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parsePartitionDataType(uint8_t selector) {
    return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        tensorflow::DataType index_dtype = parsePartitionDataType(data[offset++]);
        tensorflow::DataType shape_dtype = index_dtype;

        const int values_size = (data[offset++ % size] % 16) + 1;
        tensorflow::Tensor values_tensor(values_dtype,
                                         tensorflow::TensorShape({values_size}));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);

        tensorflow::Tensor default_value_tensor(values_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(default_value_tensor, values_dtype, data, offset, size);

        const int num_rows = std::max<int>(1, (data[offset++ % size] % 6) + 1);
        std::vector<int64_t> row_splits_vec;
        row_splits_vec.reserve(num_rows + 1);
        row_splits_vec.push_back(0);
        int64_t remaining = values_size;
        for (int i = 1; i < num_rows; ++i) {
            int64_t step = 0;
            if (offset < size) {
                step = static_cast<int64_t>(data[offset++] % (remaining + 1));
            }
            int64_t next_split = std::min<int64_t>(values_size, row_splits_vec.back() + step);
            row_splits_vec.push_back(next_split);
            remaining = values_size - next_split;
        }
        row_splits_vec.push_back(values_size);

        int64_t max_row_len = 0;
        for (int i = 1; i < static_cast<int>(row_splits_vec.size()); ++i) {
            max_row_len = std::max<int64_t>(max_row_len, row_splits_vec[i] - row_splits_vec[i - 1]);
        }
        if (max_row_len <= 0) {
            max_row_len = 1;
        }

        tensorflow::TensorShape shape_tensor_shape({2});
        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat_shape = shape_tensor.flat<int32_t>();
            flat_shape(0) = static_cast<int32_t>(num_rows);
            flat_shape(1) = static_cast<int32_t>(max_row_len);
        } else {
            auto flat_shape = shape_tensor.flat<int64_t>();
            flat_shape(0) = static_cast<int64_t>(num_rows);
            flat_shape(1) = static_cast<int64_t>(max_row_len);
        }

        tensorflow::TensorShape row_splits_shape({static_cast<int64_t>(row_splits_vec.size())});
        tensorflow::Tensor row_splits_tensor(index_dtype, row_splits_shape);
        if (index_dtype == tensorflow::DT_INT32) {
            auto flat_splits = row_splits_tensor.flat<int32_t>();
            for (size_t i = 0; i < row_splits_vec.size(); ++i) {
                flat_splits(i) = static_cast<int32_t>(row_splits_vec[i]);
            }
        } else {
            auto flat_splits = row_splits_tensor.flat<int64_t>();
            for (size_t i = 0; i < row_splits_vec.size(); ++i) {
                flat_splits(i) = row_splits_vec[i];
            }
        }

        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto default_value_input = tensorflow::ops::Const(root, default_value_tensor);
        auto row_splits_input = tensorflow::ops::Const(root, row_splits_tensor);

        std::vector<std::string> row_partition_types = {"ROW_SPLITS"};
        std::vector<tensorflow::NodeBuilder::NodeOut> partition_inputs;
        partition_inputs.emplace_back(row_splits_input.node());

        tensorflow::Node* ragged_node = nullptr;
        auto builder = tensorflow::NodeBuilder(root.GetUniqueNameForOp("RaggedTensorToTensor"),
                                               "RaggedTensorToTensor")
                           .Input(tensorflow::NodeBuilder::NodeOut(shape_input.node()))
                           .Input(tensorflow::NodeBuilder::NodeOut(values_input.node()))
                           .Input(tensorflow::NodeBuilder::NodeOut(default_value_input.node()))
                           .Input(partition_inputs)
                           .Attr("T", values_dtype)
                           .Attr("Tindex", index_dtype)
                           .Attr("Tshape", shape_dtype)
                           .Attr("num_row_partition_tensors",
                                 static_cast<int>(partition_inputs.size()))
                           .Attr("row_partition_types", row_partition_types);

        tensorflow::Status status = builder.Finalize(root.graph(), &ragged_node);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to build RaggedTensorToTensor: " + status.ToString(), data, size);
            return 0;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(ragged_node, 0)}, &outputs);
        if (!status.ok()) {
            return 0;
        }
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
