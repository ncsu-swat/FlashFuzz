#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseInputDataType(uint8_t selector) {
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

tensorflow::DataType parseBiasDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_QINT32;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseInputDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseInputDataType(data[offset++]);
        tensorflow::DataType bias_dtype = parseBiasDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseOutputDataType(data[offset++]);

        uint8_t input_rank = 4;
        uint8_t filter_rank = 4;
        uint8_t bias_rank = 1;

        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);
        std::vector<int64_t> bias_shape = {filter_shape[3]};

        if (input_shape.size() != 4 || filter_shape.size() != 4) {
            return 0;
        }

        if (input_shape[3] != filter_shape[2]) {
            filter_shape[2] = input_shape[3];
        }

        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(filter_dtype, tensorflow::TensorShape(filter_shape));
        tensorflow::Tensor bias_tensor(bias_dtype, tensorflow::TensorShape(bias_shape));

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        fillTensorWithDataByType(bias_tensor, bias_dtype, data, offset, size);

        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));

        fillTensorWithDataByType(min_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_freezed_output_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_freezed_output_tensor, tensorflow::DT_FLOAT, data, offset, size);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto bias_op = tensorflow::ops::Const(root, bias_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_filter_op = tensorflow::ops::Const(root, min_filter_tensor);
        auto max_filter_op = tensorflow::ops::Const(root, max_filter_tensor);
        auto min_freezed_output_op = tensorflow::ops::Const(root, min_freezed_output_tensor);
        auto max_freezed_output_op = tensorflow::ops::Const(root, max_freezed_output_tensor);

        std::vector<int> strides = {1, 1, 1, 1};
        if (offset + 4 <= size) {
            strides[1] = (data[offset++] % 3) + 1;
            strides[2] = (data[offset++] % 3) + 1;
        }

        std::string padding = (data[offset++] % 2 == 0) ? "SAME" : "VALID";

        std::vector<int> dilations = {1, 1, 1, 1};
        if (offset + 2 <= size) {
            dilations[1] = (data[offset++] % 3) + 1;
            dilations[2] = (data[offset++] % 3) + 1;
        }

        tensorflow::Output output;
        tensorflow::Output min_output;
        tensorflow::Output max_output;

        tensorflow::NodeDef node_def;
        node_def.set_op("QuantizedConv2DWithBiasAndRequantize");
        node_def.set_name("QuantizedConv2DWithBiasAndRequantize");
        
        // Add inputs
        node_def.add_input(input_op.node()->name());
        node_def.add_input(filter_op.node()->name());
        node_def.add_input(bias_op.node()->name());
        node_def.add_input(min_input_op.node()->name());
        node_def.add_input(max_input_op.node()->name());
        node_def.add_input(min_filter_op.node()->name());
        node_def.add_input(max_filter_op.node()->name());
        node_def.add_input(min_freezed_output_op.node()->name());
        node_def.add_input(max_freezed_output_op.node()->name());
        
        // Add attributes
        auto attr = node_def.mutable_attr();
        (*attr)["Tinput"].set_type(input_dtype);
        (*attr)["Tfilter"].set_type(filter_dtype);
        (*attr)["Tbias"].set_type(bias_dtype);
        (*attr)["out_type"].set_type(out_dtype);
        
        tensorflow::AttrValue strides_attr;
        for (auto s : strides) {
            strides_attr.mutable_list()->add_i(s);
        }
        (*attr)["strides"] = strides_attr;
        
        (*attr)["padding"].set_s(padding);
        
        tensorflow::AttrValue dilations_attr;
        for (auto d : dilations) {
            dilations_attr.mutable_list()->add_i(d);
        }
        (*attr)["dilations"] = dilations_attr;
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return 0;
        }
        
        output = tensorflow::Output(op, 0);
        min_output = tensorflow::Output(op, 1);
        max_output = tensorflow::Output(op, 2);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({output, min_output, max_output}, &outputs);
        
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return 0;
    }

    return 0;
}
