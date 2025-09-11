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

std::string parsePadding(uint8_t selector) {
    return (selector % 2 == 0) ? "SAME" : "VALID";
}

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides(4, 1);
    for (int i = 0; i < 4; ++i) {
        if (offset + sizeof(int) <= total_size) {
            int stride_val;
            std::memcpy(&stride_val, data + offset, sizeof(int));
            offset += sizeof(int);
            stride_val = std::abs(stride_val) % 5 + 1;
            strides[i] = stride_val;
        }
    }
    return strides;
}

std::vector<int> parseDilations(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> dilations(4, 1);
    for (int i = 0; i < 4; ++i) {
        if (offset + sizeof(int) <= total_size) {
            int dilation_val;
            std::memcpy(&dilation_val, data + offset, sizeof(int));
            offset += sizeof(int);
            dilation_val = std::abs(dilation_val) % 3 + 1;
            dilations[i] = dilation_val;
        }
    }
    return dilations;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseQuantizedDataType(data[offset++]);

        uint8_t input_rank = 4;
        uint8_t filter_rank = 4;

        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);

        if (input_shape.size() != 4 || filter_shape.size() != 4) {
            return 0;
        }

        filter_shape[3] = input_shape[3];

        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);

        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor filter_tensor(filter_dtype, filter_tensor_shape);

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);

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

        std::vector<int> strides = parseStrides(data, offset, size);
        std::vector<int> dilations = parseDilations(data, offset, size);
        std::string padding = parsePadding(data[offset++]);

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_filter_op = tensorflow::ops::Const(root, min_filter_tensor);
        auto max_filter_op = tensorflow::ops::Const(root, max_filter_tensor);
        auto min_freezed_output_op = tensorflow::ops::Const(root, min_freezed_output_tensor);
        auto max_freezed_output_op = tensorflow::ops::Const(root, max_freezed_output_tensor);

        // Use raw_ops namespace for QuantizedConv2DAndRequantize
        auto quantized_conv2d_and_requantize = tensorflow::ops::internal::QuantizedConv2DAndRequantize(
            root,
            input_op,
            filter_op,
            min_input_op,
            max_input_op,
            min_filter_op,
            max_filter_op,
            min_freezed_output_op,
            max_freezed_output_op,
            out_dtype,
            strides,
            padding,
            dilations
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({quantized_conv2d_and_requantize.output,
                                                 quantized_conv2d_and_requantize.min_output,
                                                 quantized_conv2d_and_requantize.max_output}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
