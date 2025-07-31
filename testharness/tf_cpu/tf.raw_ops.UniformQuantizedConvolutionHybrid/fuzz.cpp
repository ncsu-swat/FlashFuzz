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
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#define MAX_RANK 4
#define MIN_RANK 3
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
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

std::string parsePadding(uint8_t byte) {
    switch (byte % 3) {
        case 0: return "SAME";
        case 1: return "VALID";
        case 2: return "EXPLICIT";
        default: return "SAME";
    }
}

std::vector<int64_t> parseIntList(const uint8_t* data, size_t& offset, size_t total_size, size_t max_size) {
    std::vector<int64_t> result;
    if (offset >= total_size) return result;
    
    uint8_t list_size = data[offset++] % (max_size + 1);
    
    for (size_t i = 0; i < list_size && offset + sizeof(int64_t) <= total_size; ++i) {
        int64_t val;
        std::memcpy(&val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        val = std::abs(val) % 10 + 1;
        result.push_back(val);
    }
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t lhs_rank = parseRank(data[offset++]);
        uint8_t rhs_rank = lhs_rank;
        
        auto lhs_shape = parseShape(data, offset, size, lhs_rank);
        auto rhs_shape = parseShape(data, offset, size, rhs_rank);
        
        if (lhs_shape.size() < 3 || rhs_shape.size() < 3) return 0;
        
        tensorflow::Tensor lhs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(lhs_shape));
        fillTensorWithDataByType(lhs_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor rhs_tensor(tensorflow::DT_QINT8, tensorflow::TensorShape(rhs_shape));
        fillTensorWithDataByType(rhs_tensor, tensorflow::DT_QINT8, data, offset, size);
        
        int64_t kernel_output_feature_dim = rhs_shape.back();
        
        tensorflow::Tensor rhs_scales_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({kernel_output_feature_dim}));
        fillTensorWithDataByType(rhs_scales_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor rhs_zero_points_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({kernel_output_feature_dim}));
        fillTensorWithDataByType(rhs_zero_points_tensor, tensorflow::DT_INT32, data, offset, size);
        
        auto lhs_input = tensorflow::ops::Const(root, lhs_tensor);
        auto rhs_input = tensorflow::ops::Const(root, rhs_tensor);
        auto rhs_scales_input = tensorflow::ops::Const(root, rhs_scales_tensor);
        auto rhs_zero_points_input = tensorflow::ops::Const(root, rhs_zero_points_tensor);
        
        std::string padding = parsePadding(data[offset++]);
        
        int32_t rhs_quantization_min_val = -128;
        int32_t rhs_quantization_max_val = 127;
        
        auto window_strides = parseIntList(data, offset, size, lhs_rank - 2);
        auto explicit_padding = parseIntList(data, offset, size, 2 * (lhs_rank - 2));
        auto lhs_dilation = parseIntList(data, offset, size, lhs_rank - 2);
        auto rhs_dilation = parseIntList(data, offset, size, rhs_rank - 2);
        
        int64_t batch_group_count = 1;
        int64_t feature_group_count = 1;
        
        tensorflow::NodeBuilder node_builder(
            root.WithOpName("UniformQuantizedConvolutionHybrid").unique_name(),
            "UniformQuantizedConvolutionHybrid");
        
        node_builder.Input(lhs_input.node());
        node_builder.Input(rhs_input.node());
        node_builder.Input(rhs_scales_input.node());
        node_builder.Input(rhs_zero_points_input.node());
        
        node_builder.Attr("Tin", tensorflow::DT_FLOAT);
        node_builder.Attr("Tout", tensorflow::DT_FLOAT);
        node_builder.Attr("padding", padding);
        node_builder.Attr("window_strides", window_strides);
        if (padding == "EXPLICIT") {
            node_builder.Attr("explicit_padding", explicit_padding);
        }
        node_builder.Attr("lhs_dilation", lhs_dilation);
        node_builder.Attr("rhs_dilation", rhs_dilation);
        node_builder.Attr("batch_group_count", batch_group_count);
        node_builder.Attr("feature_group_count", feature_group_count);
        node_builder.Attr("rhs_quantization_axis", -1);
        node_builder.Attr("rhs_quantization_min_val", rhs_quantization_min_val);
        node_builder.Attr("rhs_quantization_max_val", rhs_quantization_max_val);
        
        tensorflow::Node* output_node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &output_node);
        
        if (!status.ok()) {
            return -1;
        }
        
        auto result = tensorflow::Output(output_node, 0);
        
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