#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

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

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides = {1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            int stride = (data[offset] % 3) + 1;
            strides[i] = stride;
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
            int dilation = (data[offset] % 3) + 1;
            dilations[i] = dilation;
            offset++;
        }
    }
    return dilations;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 100) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType bias_dtype = parseBiasDataType(data[offset++]);
        tensorflow::DataType summand_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseOutputDataType(data[offset++]);

        uint8_t input_rank = 4;
        uint8_t filter_rank = 4;
        uint8_t bias_rank = 1;
        uint8_t summand_rank = 4;

        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);
        std::vector<int64_t> bias_shape = {filter_shape[3]};
        std::vector<int64_t> summand_shape = input_shape;

        if (input_shape.size() != 4 || filter_shape.size() != 4) {
            return 0;
        }

        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(filter_dtype, tensorflow::TensorShape(filter_shape));
        tensorflow::Tensor bias_tensor(bias_dtype, tensorflow::TensorShape(bias_shape));
        tensorflow::Tensor summand_tensor(summand_dtype, tensorflow::TensorShape(summand_shape));

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        fillTensorWithDataByType(bias_tensor, bias_dtype, data, offset, size);
        fillTensorWithDataByType(summand_tensor, summand_dtype, data, offset, size);

        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_filter_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_summand_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_summand_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));

        fillTensorWithDataByType(min_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_filter_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_freezed_output_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_freezed_output_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_summand_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_summand_tensor, tensorflow::DT_FLOAT, data, offset, size);

        std::vector<int> strides = parseStrides(data, offset, size);
        std::string padding = parsePadding(data[offset++]);
        std::vector<int> dilations = parseDilations(data, offset, size);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto bias_op = tensorflow::ops::Const(root, bias_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_filter_op = tensorflow::ops::Const(root, min_filter_tensor);
        auto max_filter_op = tensorflow::ops::Const(root, max_filter_tensor);
        auto min_freezed_output_op = tensorflow::ops::Const(root, min_freezed_output_tensor);
        auto max_freezed_output_op = tensorflow::ops::Const(root, max_freezed_output_tensor);
        auto summand_op = tensorflow::ops::Const(root, summand_tensor);
        auto min_summand_op = tensorflow::ops::Const(root, min_summand_tensor);
        auto max_summand_op = tensorflow::ops::Const(root, max_summand_tensor);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Output> inputs = {
            input_op, filter_op, bias_op, 
            min_input_op, max_input_op, 
            min_filter_op, max_filter_op,
            min_freezed_output_op, max_freezed_output_op,
            summand_op, min_summand_op, max_summand_op
        };

        std::vector<tensorflow::Tensor> outputs;
        
        // Use raw_ops directly with NodeBuilder
        tensorflow::Node* node;
        tensorflow::Status status;
        
        tensorflow::NodeBuilder node_builder = tensorflow::NodeBuilder("quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize", 
                                                                      "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
            .Input(input_op.node())
            .Input(filter_op.node())
            .Input(bias_op.node())
            .Input(min_input_op.node())
            .Input(max_input_op.node())
            .Input(min_filter_op.node())
            .Input(max_filter_op.node())
            .Input(min_freezed_output_op.node())
            .Input(max_freezed_output_op.node())
            .Input(summand_op.node())
            .Input(min_summand_op.node())
            .Input(max_summand_op.node())
            .Attr("Tinput", input_dtype)
            .Attr("Tfilter", filter_dtype)
            .Attr("Tbias", bias_dtype)
            .Attr("Tsummand", summand_dtype)
            .Attr("out_type", out_dtype)
            .Attr("strides", strides)
            .Attr("padding", padding)
            .Attr("dilations", dilations);

        status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return 0;
        }

        // Create output tensors
        tensorflow::Tensor output_tensor;
        tensorflow::Tensor min_output_tensor;
        tensorflow::Tensor max_output_tensor;

        // Run the session
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_feed;
        std::vector<std::string> output_names = {
            "quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize:0",
            "quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize:1",
            "quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize:2"
        };
        
        status = session.Run(inputs_feed, output_names, {}, &outputs);
        
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        return 0;
    }

    return 0;
}