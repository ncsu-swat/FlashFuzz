#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype_a = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType dtype_b = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType dtype_bias = parseBiasDataType(data[offset++]);
        tensorflow::DataType dtype_output = parseOutputDataType(data[offset++]);
        
        uint8_t rank_a = parseRank(data[offset++]);
        uint8_t rank_b = parseRank(data[offset++]);
        uint8_t rank_bias = parseRank(data[offset++]);
        
        if (rank_a < 2) rank_a = 2;
        if (rank_b < 2) rank_b = 2;
        if (rank_bias < 1) rank_bias = 1;
        
        std::vector<int64_t> shape_a = parseShape(data, offset, size, rank_a);
        std::vector<int64_t> shape_b = parseShape(data, offset, size, rank_b);
        std::vector<int64_t> shape_bias = parseShape(data, offset, size, rank_bias);
        
        if (shape_a.size() >= 2 && shape_b.size() >= 2) {
            shape_b[shape_b.size()-2] = shape_a[shape_a.size()-1];
        }
        
        if (shape_bias.size() >= 1 && shape_b.size() >= 1) {
            shape_bias[shape_bias.size()-1] = shape_b[shape_b.size()-1];
        }
        
        tensorflow::Tensor tensor_a(dtype_a, tensorflow::TensorShape(shape_a));
        tensorflow::Tensor tensor_b(dtype_b, tensorflow::TensorShape(shape_b));
        tensorflow::Tensor tensor_bias(dtype_bias, tensorflow::TensorShape(shape_bias));
        
        fillTensorWithDataByType(tensor_a, dtype_a, data, offset, size);
        fillTensorWithDataByType(tensor_b, dtype_b, data, offset, size);
        fillTensorWithDataByType(tensor_bias, dtype_bias, data, offset, size);
        
        tensorflow::Tensor min_a_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_a_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_b_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_b_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_freezed_output_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        fillTensorWithData<float>(min_a_tensor, data, offset, size);
        fillTensorWithData<float>(max_a_tensor, data, offset, size);
        fillTensorWithData<float>(min_b_tensor, data, offset, size);
        fillTensorWithData<float>(max_b_tensor, data, offset, size);
        fillTensorWithData<float>(min_freezed_output_tensor, data, offset, size);
        fillTensorWithData<float>(max_freezed_output_tensor, data, offset, size);
        
        auto a_input = tensorflow::ops::Const(root, tensor_a);
        auto b_input = tensorflow::ops::Const(root, tensor_b);
        auto bias_input = tensorflow::ops::Const(root, tensor_bias);
        auto min_a_input = tensorflow::ops::Const(root, min_a_tensor);
        auto max_a_input = tensorflow::ops::Const(root, max_a_tensor);
        auto min_b_input = tensorflow::ops::Const(root, min_b_tensor);
        auto max_b_input = tensorflow::ops::Const(root, max_b_tensor);
        auto min_freezed_output_input = tensorflow::ops::Const(root, min_freezed_output_tensor);
        auto max_freezed_output_input = tensorflow::ops::Const(root, max_freezed_output_tensor);
        
        bool transpose_a = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool transpose_b = (offset < size) ? (data[offset++] % 2 == 1) : false;
        std::string input_quant_mode = (offset < size && data[offset++] % 2 == 1) ? "SCALED" : "MIN_FIRST";
        
        // Use raw_ops namespace for QuantizedMatMulWithBiasAndRequantize
        auto result = tensorflow::ops::RawOps::QuantizedMatMulWithBiasAndRequantize(
            root, a_input, b_input, bias_input, min_a_input, max_a_input,
            min_b_input, max_b_input, min_freezed_output_input, max_freezed_output_input,
            tensorflow::ops::RawOps::QuantizedMatMulWithBiasAndRequantize::Attrs()
                .Toutput(dtype_output)
                .TransposeA(transpose_a)
                .TransposeB(transpose_b)
                .InputQuantMode(input_quant_mode));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result.output, result.min_output, result.max_output}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}