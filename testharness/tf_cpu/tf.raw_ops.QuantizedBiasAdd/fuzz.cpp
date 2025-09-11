#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType bias_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseQuantizedDataType(data[offset++]);

        uint8_t input_rank = parseRank(data[offset++]);
        if (input_rank < 1) input_rank = 2;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        if (input_shape.empty()) {
            input_shape = {2, 3};
        }

        int64_t bias_size = input_shape.back();
        std::vector<int64_t> bias_shape = {bias_size};

        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape bias_tensor_shape(bias_shape);
        tensorflow::TensorShape scalar_shape({});

        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor bias_tensor(bias_dtype, bias_tensor_shape);
        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor min_bias_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor max_bias_tensor(tensorflow::DT_FLOAT, scalar_shape);

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(bias_tensor, bias_dtype, data, offset, size);

        float min_input_val = -1.0f;
        float max_input_val = 1.0f;
        float min_bias_val = -1.0f;
        float max_bias_val = 1.0f;

        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_input_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_input_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_bias_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_bias_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        min_input_tensor.scalar<float>()() = min_input_val;
        max_input_tensor.scalar<float>()() = max_input_val;
        min_bias_tensor.scalar<float>()() = min_bias_val;
        max_bias_tensor.scalar<float>()() = max_bias_val;

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto bias_op = tensorflow::ops::Const(root, bias_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_bias_op = tensorflow::ops::Const(root, min_bias_tensor);
        auto max_bias_op = tensorflow::ops::Const(root, max_bias_tensor);

        tensorflow::Node* quantized_bias_add_node;
        tensorflow::NodeBuilder builder("quantized_bias_add", "QuantizedBiasAdd");
        builder.Input(input_op.node())
               .Input(bias_op.node())
               .Input(min_input_op.node())
               .Input(max_input_op.node())
               .Input(min_bias_op.node())
               .Input(max_bias_op.node())
               .Attr("out_type", out_dtype);

        tensorflow::Status build_status = builder.Finalize(root.graph(), &quantized_bias_add_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Output> fetch_outputs = {
            tensorflow::Output(quantized_bias_add_node, 0),
            tensorflow::Output(quantized_bias_add_node, 1),
            tensorflow::Output(quantized_bias_add_node, 2)
        };
        
        std::vector<tensorflow::Tensor> outputs;
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
