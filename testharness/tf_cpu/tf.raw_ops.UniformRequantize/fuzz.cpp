#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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

tensorflow::DataType parseInputDataType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_QINT8;
        case 1:
            return tensorflow::DT_QINT32;
        default:
            return tensorflow::DT_QINT8;
    }
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_QINT8;
        case 1:
            return tensorflow::DT_QINT32;
        default:
            return tensorflow::DT_QINT8;
    }
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
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
        tensorflow::DataType input_dtype = parseInputDataType(data[offset++]);
        tensorflow::DataType output_dtype = parseOutputDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        int64_t quantization_axis_size = 1;
        if (input_rank > 0) {
            uint8_t axis_idx = data[offset % size];
            offset++;
            int axis = static_cast<int>(axis_idx % input_rank);
            quantization_axis_size = input_shape[axis];
        }
        
        bool per_tensor = (data[offset % size] % 2 == 0);
        offset++;
        
        int64_t scale_size = per_tensor ? 1 : quantization_axis_size;
        
        tensorflow::TensorShape scale_shape;
        if (per_tensor) {
            scale_shape = tensorflow::TensorShape({});
        } else {
            scale_shape = tensorflow::TensorShape({scale_size});
        }
        
        tensorflow::Tensor input_scales(tensorflow::DT_FLOAT, scale_shape);
        fillTensorWithDataByType(input_scales, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor input_zero_points(tensorflow::DT_INT32, scale_shape);
        fillTensorWithDataByType(input_zero_points, tensorflow::DT_INT32, data, offset, size);
        
        tensorflow::Tensor output_scales(tensorflow::DT_FLOAT, scale_shape);
        fillTensorWithDataByType(output_scales, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor output_zero_points(tensorflow::DT_INT32, scale_shape);
        fillTensorWithDataByType(output_zero_points, tensorflow::DT_INT32, data, offset, size);
        
        int input_quantization_min_val = -128;
        int input_quantization_max_val = 127;
        int output_quantization_min_val = -128;
        int output_quantization_max_val = 127;
        
        if (input_dtype == tensorflow::DT_QINT32) {
            input_quantization_min_val = -2147483648;
            input_quantization_max_val = 2147483647;
        }
        
        if (output_dtype == tensorflow::DT_QINT32) {
            output_quantization_min_val = -2147483648;
            output_quantization_max_val = 2147483647;
        }
        
        int input_quantization_axis = per_tensor ? -1 : 0;
        int output_quantization_axis = per_tensor ? -1 : 0;
        
        auto input_node = tensorflow::ops::Const(root, input_tensor);
        auto input_scales_node = tensorflow::ops::Const(root, input_scales);
        auto input_zero_points_node = tensorflow::ops::Const(root, input_zero_points);
        auto output_scales_node = tensorflow::ops::Const(root, output_scales);
        auto output_zero_points_node = tensorflow::ops::Const(root, output_zero_points);
        
        // Use raw_ops namespace for UniformRequantize
        auto uniform_requantize = tensorflow::ops::UniformQuantizedDequantize(
            root,
            input_node,
            input_scales_node,
            input_zero_points_node,
            tensorflow::DT_FLOAT,
            tensorflow::ops::UniformQuantizedDequantize::Attrs()
                .QuantizationAxis(input_quantization_axis)
                .QuantizationMinVal(input_quantization_min_val)
                .QuantizationMaxVal(input_quantization_max_val)
        );
        
        // Then requantize using UniformQuantize
        auto requantized = tensorflow::ops::UniformQuantize(
            root,
            uniform_requantize,
            output_scales_node,
            output_zero_points_node,
            output_dtype,
            tensorflow::ops::UniformQuantize::Attrs()
                .QuantizationAxis(output_quantization_axis)
                .QuantizationMinVal(output_quantization_min_val)
                .QuantizationMaxVal(output_quantization_max_val)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({requantized}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}