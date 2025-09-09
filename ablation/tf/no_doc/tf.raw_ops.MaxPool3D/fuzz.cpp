#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>

constexpr uint8_t MIN_RANK = 5;
constexpr uint8_t MAX_RANK = 5;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
            break;
        case 3:
            dtype = tensorflow::DT_BFLOAT16;
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
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Input tensor dtype: " << tensorflow::DataTypeString(dtype) << std::endl;
        
        if (offset + 15 > size) {
            return 0;
        }
        
        std::vector<int32_t> ksize(5);
        std::vector<int32_t> strides(5);
        
        for (int i = 0; i < 5; ++i) {
            ksize[i] = (data[offset++] % 3) + 1;
        }
        
        for (int i = 0; i < 5; ++i) {
            strides[i] = (data[offset++] % 3) + 1;
        }
        
        std::string padding = (data[offset++] % 2 == 0) ? "VALID" : "SAME";
        std::string data_format = "NDHWC";
        
        std::cout << "ksize: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << ksize[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "strides: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << strides[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "padding: " << padding << std::endl;
        std::cout << "data_format: " << data_format << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        auto maxpool3d_op = tensorflow::ops::MaxPool3D(
            root, 
            input_placeholder, 
            ksize, 
            strides, 
            padding,
            tensorflow::ops::MaxPool3D::DataFormat(data_format)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor}}, 
            {maxpool3d_op}, 
            &outputs
        );
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MaxPool3D operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MaxPool3D operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}