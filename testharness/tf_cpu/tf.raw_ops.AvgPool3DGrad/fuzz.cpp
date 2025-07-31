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

#define MAX_RANK 5
#define MIN_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType grad_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> orig_input_shape_dims = {5};
        tensorflow::Tensor orig_input_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(orig_input_shape_dims));
        auto orig_input_flat = orig_input_shape_tensor.flat<int32_t>();
        
        std::vector<int64_t> grad_shape(5);
        for (int i = 0; i < 5; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t dim;
                std::memcpy(&dim, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dim = std::abs(dim) % 10 + 1;
                grad_shape[i] = dim;
                orig_input_flat(i) = dim;
            } else {
                grad_shape[i] = 1;
                orig_input_flat(i) = 1;
            }
        }
        
        tensorflow::Tensor grad_tensor(grad_dtype, tensorflow::TensorShape(grad_shape));
        fillTensorWithDataByType(grad_tensor, grad_dtype, data, offset, size);
        
        std::vector<int> ksize = {1, 2, 2, 2, 1};
        std::vector<int> strides = {1, 1, 1, 1, 1};
        
        if (offset < size) {
            uint8_t ksize_selector = data[offset++];
            ksize[1] = (ksize_selector % 3) + 1;
            ksize[2] = (ksize_selector % 3) + 1;
            ksize[3] = (ksize_selector % 3) + 1;
        }
        
        if (offset < size) {
            uint8_t stride_selector = data[offset++];
            strides[1] = (stride_selector % 3) + 1;
            strides[2] = (stride_selector % 3) + 1;
            strides[3] = (stride_selector % 3) + 1;
        }
        
        std::string padding = "VALID";
        if (offset < size) {
            padding = (data[offset++] % 2 == 0) ? "VALID" : "SAME";
        }
        
        std::string data_format = "NDHWC";
        if (offset < size) {
            data_format = (data[offset++] % 2 == 0) ? "NDHWC" : "NCDHW";
        }
        
        auto orig_input_shape_op = tensorflow::ops::Const(root, orig_input_shape_tensor);
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        
        auto avg_pool_3d_grad = tensorflow::ops::AvgPool3DGrad(
            root, orig_input_shape_op, grad_op, ksize, strides, padding,
            tensorflow::ops::AvgPool3DGrad::DataFormat(data_format)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({avg_pool_3d_grad}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}