#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
            break;
    }
    return dtype;
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
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
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides = {1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            int stride_val = static_cast<int>(data[offset] % 3) + 1;
            strides[i] = stride_val;
            offset++;
        }
    }
    return strides;
}

std::string parsePadding(uint8_t byte) {
    return (byte % 2 == 0) ? "SAME" : "VALID";
}

std::vector<int> parseDilations(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> dilations = {1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            int dilation_val = static_cast<int>(data[offset] % 3) + 1;
            dilations[i] = dilation_val;
            offset++;
        }
    }
    return dilations;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseOutputDataType(data[offset++]);

        uint8_t input_rank = 4;
        uint8_t filter_rank = 4;
        uint8_t bias_rank = 1;
        uint8_t scalar_rank = 0;

        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        if (input_shape.size() != 4) {
            input_shape = {1, 8, 8, 3};
        }

        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);
        if (filter_shape.size() != 4) {
            filter_shape = {3, 3, input_shape[3], 16};
        } else {
            filter_shape[2] = input_shape[3];
        }

        std::vector<int64_t> bias_shape = {filter_shape[3]};
        std::vector<int64_t> scalar_shape = {};

        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(filter_dtype, tensorflow::TensorShape(filter_shape));
        tensorflow::Tensor bias_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(bias_shape));
        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scalar_shape));
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scalar_shape));
        tensorflow::Tensor min_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scalar_shape));
        tensorflow::Tensor max_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scalar_shape));

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        fillTensorWithDataByType(bias_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto bias_op = tensorflow::ops::Const(root, bias_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_filter_op = tensorflow::ops::Const(root, min_filter_tensor);
        auto max_filter_op = tensorflow::ops::Const(root, max_filter_tensor);

        std::vector<int> strides = parseStrides(data, offset, size);
        std::string padding = parsePadding(data[offset++]);
        std::vector<int> dilations = parseDilations(data, offset, size);

        tensorflow::Output output;
        tensorflow::Output min_output;
        tensorflow::Output max_output;

        // Use raw_ops directly since QuantizedConv2DWithBiasAndRelu is not in the C++ API
        tensorflow::NodeDef node_def;
        node_def.set_op("QuantizedConv2DWithBiasAndRelu");
        node_def.set_name("QuantizedConv2DWithBiasAndRelu");
        
        // Add inputs
        tensorflow::AddNodeInput("input", &node_def);
        tensorflow::AddNodeInput("filter", &node_def);
        tensorflow::AddNodeInput("bias", &node_def);
        tensorflow::AddNodeInput("min_input", &node_def);
        tensorflow::AddNodeInput("max_input", &node_def);
        tensorflow::AddNodeInput("min_filter", &node_def);
        tensorflow::AddNodeInput("max_filter", &node_def);
        
        // Add attributes
        auto* attr_map = node_def.mutable_attr();
        tensorflow::SetAttrValue(strides, &(*attr_map)["strides"]);
        tensorflow::SetAttrValue(padding, &(*attr_map)["padding"]);
        tensorflow::SetAttrValue(dilations, &(*attr_map)["dilations"]);
        tensorflow::SetAttrValue(out_dtype, &(*attr_map)["out_type"]);
        tensorflow::SetAttrValue(input_dtype, &(*attr_map)["Tinput"]);
        tensorflow::SetAttrValue(filter_dtype, &(*attr_map)["Tfilter"]);
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        output = tensorflow::Output(op, 0);
        min_output = tensorflow::Output(op, 1);
        max_output = tensorflow::Output(op, 2);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({output, min_output, max_output}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
