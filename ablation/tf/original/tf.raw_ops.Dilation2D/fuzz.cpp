#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 11) {
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
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_HALF;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
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
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = {1, 3, 3, 1};
        std::vector<int64_t> filter_shape = {2, 2, 1};
        
        if (offset + 4 <= size) {
            input_shape[0] = (data[offset] % 3) + 1;
            input_shape[1] = (data[offset + 1] % 5) + 2;
            input_shape[2] = (data[offset + 2] % 5) + 2;
            input_shape[3] = (data[offset + 3] % 3) + 1;
            offset += 4;
        }
        
        if (offset + 3 <= size) {
            filter_shape[0] = (data[offset] % 3) + 1;
            filter_shape[1] = (data[offset + 1] % 3) + 1;
            filter_shape[2] = input_shape[3];
            offset += 3;
        }

        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);
        
        tensorflow::Tensor input_tensor(dtype, input_tensor_shape);
        tensorflow::Tensor filter_tensor(dtype, filter_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        
        std::vector<int> strides = {1, 1, 1, 1};
        std::vector<int> rates = {1, 1, 1, 1};
        
        if (offset + 2 <= size) {
            strides[1] = (data[offset] % 3) + 1;
            strides[2] = (data[offset + 1] % 3) + 1;
            offset += 2;
        }
        
        if (offset + 2 <= size) {
            rates[1] = (data[offset] % 3) + 1;
            rates[2] = (data[offset + 1] % 3) + 1;
            offset += 2;
        }
        
        std::string padding = "VALID";
        if (offset < size && (data[offset] % 2) == 1) {
            padding = "SAME";
        }
        
        std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] 
                  << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
        std::cout << "Filter shape: [" << filter_shape[0] << ", " << filter_shape[1] 
                  << ", " << filter_shape[2] << "]" << std::endl;
        std::cout << "Strides: [" << strides[0] << ", " << strides[1] 
                  << ", " << strides[2] << ", " << strides[3] << "]" << std::endl;
        std::cout << "Rates: [" << rates[0] << ", " << rates[1] 
                  << ", " << rates[2] << ", " << rates[3] << "]" << std::endl;
        std::cout << "Padding: " << padding << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        auto dilation_op = tensorflow::ops::Dilation2D(
            root, input_placeholder, filter_placeholder, strides, rates, padding);
        
        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to create graph: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {filter_placeholder.node()->name(), filter_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {dilation_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Dilation2D operation completed successfully" << std::endl;
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "Dilation2D operation failed: " << status.ToString() << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}