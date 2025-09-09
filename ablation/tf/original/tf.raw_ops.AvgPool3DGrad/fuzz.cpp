#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 5;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
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
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        tensorflow::DataType grad_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> orig_input_shape_dims = {5};
        tensorflow::Tensor orig_input_shape(tensorflow::DT_INT32, tensorflow::TensorShape(orig_input_shape_dims));
        auto orig_input_flat = orig_input_shape.flat<int32_t>();
        
        if (offset + 5 * sizeof(int32_t) > size) return 0;
        for (int i = 0; i < 5; ++i) {
            int32_t dim;
            std::memcpy(&dim, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            orig_input_flat(i) = std::abs(dim) % 10 + 1;
        }
        
        std::vector<int64_t> grad_shape = {orig_input_flat(0), orig_input_flat(1), orig_input_flat(2), orig_input_flat(3), orig_input_flat(4)};
        tensorflow::Tensor grad(grad_dtype, tensorflow::TensorShape(grad_shape));
        fillTensorWithDataByType(grad, grad_dtype, data, offset, size);
        
        std::vector<int> ksize = {1, 2, 2, 2, 1};
        std::vector<int> strides = {1, 1, 1, 1, 1};
        
        if (offset + 10 * sizeof(int32_t) <= size) {
            for (int i = 1; i < 4; ++i) {
                int32_t k, s;
                std::memcpy(&k, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                std::memcpy(&s, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                ksize[i] = std::abs(k) % 5 + 1;
                strides[i] = std::abs(s) % 3 + 1;
            }
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "SAME" : "VALID";
        std::string data_format = (offset < size && data[offset++] % 2 == 0) ? "NDHWC" : "NCDHW";
        
        std::cout << "orig_input_shape: [";
        for (int i = 0; i < 5; ++i) {
            std::cout << orig_input_flat(i);
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "grad shape: [";
        for (size_t i = 0; i < grad_shape.size(); ++i) {
            std::cout << grad_shape[i];
            if (i < grad_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "ksize: [";
        for (size_t i = 0; i < ksize.size(); ++i) {
            std::cout << ksize[i];
            if (i < ksize.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "strides: [";
        for (size_t i = 0; i < strides.size(); ++i) {
            std::cout << strides[i];
            if (i < strides.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "padding: " << padding << std::endl;
        std::cout << "data_format: " << data_format << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto orig_input_shape_op = tensorflow::ops::Const(root, orig_input_shape);
        auto grad_op = tensorflow::ops::Const(root, grad);
        
        tensorflow::ops::AvgPool3DGrad::Attrs attrs;
        attrs = attrs.DataFormat(data_format);
        
        auto result = tensorflow::ops::AvgPool3DGrad(root, orig_input_shape_op, grad_op, ksize, strides, padding, attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "AvgPool3DGrad executed successfully" << std::endl;
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "AvgPool3DGrad failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}