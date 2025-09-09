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
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
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
    try {
        size_t offset = 0;
        
        if (size < 10) return 0;
        
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = 4;
        uint8_t out_backprop_rank = 4;
        uint8_t filter_sizes_rank = 1;
        
        std::vector<int64_t> input_shape = {1, 8, 8, 3};
        std::vector<int64_t> out_backprop_shape = {1, 6, 6, 16};
        std::vector<int64_t> filter_sizes_shape = {4};
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape out_backprop_tensor_shape(out_backprop_shape);
        tensorflow::TensorShape filter_sizes_tensor_shape(filter_sizes_shape);
        
        tensorflow::Tensor input_tensor(dtype, input_tensor_shape);
        tensorflow::Tensor out_backprop_tensor(dtype, out_backprop_tensor_shape);
        tensorflow::Tensor filter_sizes_tensor(tensorflow::DT_INT32, filter_sizes_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(out_backprop_tensor, dtype, data, offset, size);
        
        auto filter_sizes_flat = filter_sizes_tensor.flat<int32_t>();
        filter_sizes_flat(0) = 3;
        filter_sizes_flat(1) = 3;
        filter_sizes_flat(2) = 3;
        filter_sizes_flat(3) = 16;
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Out backprop tensor shape: ";
        for (int i = 0; i < out_backprop_tensor.dims(); ++i) {
            std::cout << out_backprop_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Filter sizes tensor shape: ";
        for (int i = 0; i < filter_sizes_tensor.dims(); ++i) {
            std::cout << filter_sizes_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto out_backprop_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto filter_sizes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        std::vector<int> strides = {1, 1, 1, 1};
        std::string padding = "VALID";
        
        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root, input_placeholder, filter_sizes_placeholder, out_backprop_placeholder,
            strides, padding);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {out_backprop_placeholder, out_backprop_tensor},
             {filter_sizes_placeholder, filter_sizes_tensor}},
            {conv2d_backprop_filter}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Conv2DBackpropFilter executed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Conv2DBackpropFilter failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}