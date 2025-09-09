#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

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
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = {1, 4, 4, 3};
        std::vector<int64_t> size_shape = {2};
        std::vector<int64_t> paddings_shape = {4, 2};
        std::vector<int64_t> filter_shape = {3, 3, 3, 32};

        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape size_tensor_shape(size_shape);
        tensorflow::TensorShape paddings_tensor_shape(paddings_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);

        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, size_tensor_shape);
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, paddings_tensor_shape);
        tensorflow::Tensor filter_tensor(filter_dtype, filter_tensor_shape);

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        auto size_flat = size_tensor.flat<int32_t>();
        size_flat(0) = 8;
        size_flat(1) = 8;

        auto paddings_flat = paddings_tensor.flat<int32_t>();
        paddings_flat(0) = 1; paddings_flat(1) = 1;
        paddings_flat(2) = 1; paddings_flat(3) = 1;
        paddings_flat(4) = 1; paddings_flat(5) = 1;
        paddings_flat(6) = 0; paddings_flat(7) = 0;

        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);

        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Filter tensor shape: ";
        for (int i = 0; i < filter_tensor.dims(); ++i) {
            std::cout << filter_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto size_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto paddings_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, filter_dtype);

        auto fused_op = tensorflow::ops::FusedResizeAndPadConv2D(
            root,
            input_placeholder,
            size_placeholder,
            paddings_placeholder,
            filter_placeholder,
            "SAME",
            {1, 1, 1, 1}
        );

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
            {size_placeholder.node()->name(), size_tensor},
            {paddings_placeholder.node()->name(), paddings_tensor},
            {filter_placeholder.node()->name(), filter_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {fused_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully. Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}