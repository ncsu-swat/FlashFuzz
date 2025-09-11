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
#include <iostream>
#include <cstring>
#include <vector>
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
    switch (selector % 1) {
        case 0:
            return tensorflow::DT_FLOAT;
        default:
            return tensorflow::DT_FLOAT;
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
        
        uint8_t scales_rank = parseRank(data[offset++]);
        std::vector<int64_t> scales_shape = parseShape(data, offset, size, scales_rank);
        
        tensorflow::TensorShape scales_tensor_shape;
        for (int64_t dim : scales_shape) {
            scales_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor scales_tensor(tensorflow::DT_FLOAT, scales_tensor_shape);
        fillTensorWithData<float>(scales_tensor, data, offset, size);
        
        uint8_t zero_points_rank = parseRank(data[offset++]);
        std::vector<int64_t> zero_points_shape = parseShape(data, offset, size, zero_points_rank);
        
        tensorflow::TensorShape zero_points_tensor_shape;
        for (int64_t dim : zero_points_shape) {
            zero_points_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor zero_points_tensor(tensorflow::DT_INT32, zero_points_tensor_shape);
        fillTensorWithData<int32_t>(zero_points_tensor, data, offset, size);
        
        int quantization_min_val = -128;
        int quantization_max_val = 127;
        if (output_dtype == tensorflow::DT_QINT32) {
            quantization_min_val = -2147483648;
            quantization_max_val = 2147483647;
        }
        
        int quantization_axis = -1;
        if (offset < size) {
            quantization_axis = static_cast<int8_t>(data[offset++]);
            if (quantization_axis >= 0 && quantization_axis >= input_rank) {
                quantization_axis = -1;
            }
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto scales_op = tensorflow::ops::Const(root, scales_tensor);
        auto zero_points_op = tensorflow::ops::Const(root, zero_points_tensor);
        
        // Use raw_ops instead of ops namespace for UniformQuantize
        tensorflow::NodeDef node_def;
        node_def.set_op("UniformQuantize");
        node_def.set_name("uniform_quantize");
        node_def.add_input(input_op.node()->name());
        node_def.add_input(scales_op.node()->name());
        node_def.add_input(zero_points_op.node()->name());
        
        auto attr = node_def.mutable_attr();
        (*attr)["Tin"].set_type(input_dtype);
        (*attr)["Tout"].set_type(output_dtype);
        (*attr)["quantization_min_val"].set_i(quantization_min_val);
        (*attr)["quantization_max_val"].set_i(quantization_max_val);
        (*attr)["quantization_axis"].set_i(quantization_axis);
        
        tensorflow::Status status;
        auto uniform_quantize = root.AddNode(node_def, &status);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create UniformQuantize op: " + status.ToString(), data, size);
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({uniform_quantize}, &outputs);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to run UniformQuantize op: " + status.ToString(), data, size);
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
