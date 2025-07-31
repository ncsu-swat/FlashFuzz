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
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
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
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = {1, 4, 4, 3};
        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        std::vector<int64_t> size_shape = {2};
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(size_shape));
        auto size_flat = size_tensor.flat<int32_t>();
        if (offset + sizeof(int32_t) <= size) {
            int32_t height;
            std::memcpy(&height, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            size_flat(0) = std::abs(height) % 10 + 1;
        } else {
            size_flat(0) = 2;
        }
        if (offset + sizeof(int32_t) <= size) {
            int32_t width;
            std::memcpy(&width, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            size_flat(1) = std::abs(width) % 10 + 1;
        } else {
            size_flat(1) = 2;
        }
        
        std::vector<int64_t> paddings_shape = {4, 2};
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(paddings_shape));
        auto paddings_flat = paddings_tensor.flat<int32_t>();
        for (int i = 0; i < 8; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t pad_val;
                std::memcpy(&pad_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                paddings_flat(i) = std::abs(pad_val) % 3;
            } else {
                paddings_flat(i) = 0;
            }
        }
        
        std::vector<int64_t> filter_shape = {3, 3, 3, 2};
        tensorflow::Tensor filter_tensor(input_dtype, tensorflow::TensorShape(filter_shape));
        fillTensorWithDataByType(filter_tensor, input_dtype, data, offset, size);
        
        std::string mode = (data[offset % size] % 2 == 0) ? "REFLECT" : "SYMMETRIC";
        offset++;
        
        std::vector<int> strides = {1, 1, 1, 1};
        if (offset + sizeof(int32_t) <= size) {
            int32_t stride_val;
            std::memcpy(&stride_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            strides[1] = std::abs(stride_val) % 3 + 1;
            strides[2] = std::abs(stride_val) % 3 + 1;
        }
        
        std::string padding = (data[offset % size] % 2 == 0) ? "SAME" : "VALID";
        offset++;
        
        bool resize_align_corners = (data[offset % size] % 2 == 0);
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto size_op = tensorflow::ops::Const(root, size_tensor);
        auto paddings_op = tensorflow::ops::Const(root, paddings_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        
        auto fused_op = tensorflow::ops::FusedResizeAndPadConv2D(
            root, input_op, size_op, paddings_op, filter_op,
            mode, strides, padding,
            tensorflow::ops::FusedResizeAndPadConv2D::ResizeAlignCorners(resize_align_corners)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({fused_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}