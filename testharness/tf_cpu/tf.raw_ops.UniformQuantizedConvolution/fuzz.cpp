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
#define MIN_RANK 3
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t lhs_rank = parseRank(data[offset++]);
        uint8_t rhs_rank = lhs_rank;
        
        auto lhs_shape = parseShape(data, offset, size, lhs_rank);
        auto rhs_shape = parseShape(data, offset, size, rhs_rank);
        
        if (lhs_shape.size() < 3 || rhs_shape.size() < 3) return 0;
        
        tensorflow::TensorShape lhs_tensor_shape(lhs_shape);
        tensorflow::TensorShape rhs_tensor_shape(rhs_shape);
        
        tensorflow::Tensor lhs_tensor(tensorflow::DT_QINT8, lhs_tensor_shape);
        tensorflow::Tensor rhs_tensor(tensorflow::DT_QINT8, rhs_tensor_shape);
        
        fillTensorWithDataByType(lhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        fillTensorWithDataByType(rhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        
        tensorflow::Tensor lhs_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor lhs_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor rhs_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor rhs_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor output_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor output_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(lhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(lhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(rhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(rhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(output_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(output_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        auto lhs_input = tensorflow::ops::Const(root, lhs_tensor);
        auto rhs_input = tensorflow::ops::Const(root, rhs_tensor);
        auto lhs_scales_input = tensorflow::ops::Const(root, lhs_scales_tensor);
        auto lhs_zero_points_input = tensorflow::ops::Const(root, lhs_zero_points_tensor);
        auto rhs_scales_input = tensorflow::ops::Const(root, rhs_scales_tensor);
        auto rhs_zero_points_input = tensorflow::ops::Const(root, rhs_zero_points_tensor);
        auto output_scales_input = tensorflow::ops::Const(root, output_scales_tensor);
        auto output_zero_points_input = tensorflow::ops::Const(root, output_zero_points_tensor);
        
        std::vector<tensorflow::ops::internal::NodeOut> inputs = {
            lhs_input, 
            rhs_input, 
            lhs_scales_input, 
            lhs_zero_points_input, 
            rhs_scales_input, 
            rhs_zero_points_input, 
            output_scales_input, 
            output_zero_points_input
        };
        
        tensorflow::NodeBuilder node_builder = 
            tensorflow::NodeBuilder("UniformQuantizedConvolution", "UniformQuantizedConvolution")
                .Input(inputs)
                .Attr("Tin", tensorflow::DT_QINT8)
                .Attr("Tout", tensorflow::DT_QINT32)
                .Attr("padding", "VALID")
                .Attr("lhs_quantization_min_val", -128)
                .Attr("lhs_quantization_max_val", 127)
                .Attr("rhs_quantization_min_val", -128)
                .Attr("rhs_quantization_max_val", 127)
                .Attr("output_quantization_min_val", -2147483648)
                .Attr("output_quantization_max_val", 2147483647)
                .Attr("window_strides", std::vector<int64_t>({1, 1}))
                .Attr("explicit_padding", std::vector<int64_t>({}))
                .Attr("lhs_dilation", std::vector<int64_t>({1, 1}))
                .Attr("rhs_dilation", std::vector<int64_t>({1, 1}))
                .Attr("batch_group_count", 1)
                .Attr("feature_group_count", 1)
                .Attr("dimension_numbers", "")
                .Attr("lhs_quantization_axis", -1)
                .Attr("rhs_quantization_axis", -1)
                .Attr("output_quantization_axis", -1);
        
        tensorflow::Node* node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return -1;
        }
        
        auto result = tensorflow::Output(node, 0);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
