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

constexpr uint8_t MIN_RANK = 4;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_HALF;
            break;
        case 5:
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        
        if (offset + 16 > size) {
            return 0;
        }
        
        int32_t ksize_h, ksize_w, stride_h, stride_w;
        std::memcpy(&ksize_h, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        std::memcpy(&ksize_w, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        std::memcpy(&stride_h, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        std::memcpy(&stride_w, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        ksize_h = std::abs(ksize_h) % 5 + 1;
        ksize_w = std::abs(ksize_w) % 5 + 1;
        stride_h = std::abs(stride_h) % 3 + 1;
        stride_w = std::abs(stride_w) % 3 + 1;
        
        std::vector<int> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        
        std::string padding = (offset < size && data[offset] % 2 == 0) ? "VALID" : "SAME";
        
        std::cout << "ksize: [" << ksize[0] << ", " << ksize[1] << ", " << ksize[2] << ", " << ksize[3] << "]" << std::endl;
        std::cout << "strides: [" << strides[0] << ", " << strides[1] << ", " << strides[2] << ", " << strides[3] << "]" << std::endl;
        std::cout << "padding: " << padding << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        auto maxpool_op = tensorflow::ops::MaxPool(root, input_placeholder, ksize, strides, padding);
        
        tensorflow::GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {maxpool_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MaxPool operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MaxPool operation failed: " << status.ToString() << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}