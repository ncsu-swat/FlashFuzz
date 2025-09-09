#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/nn_ops.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 6;
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t orig_input_rank = parseRank(data[offset++]);
        uint8_t grad_rank = parseRank(data[offset++]);
        uint8_t maxpool_input_rank = parseRank(data[offset++]);

        std::vector<int64_t> orig_input_shape = parseShape(data, offset, size, orig_input_rank);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        std::vector<int64_t> maxpool_input_shape = parseShape(data, offset, size, maxpool_input_rank);

        if (offset >= size) {
            return 0;
        }

        tensorflow::TensorShape orig_input_tensor_shape(orig_input_shape);
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape maxpool_input_tensor_shape(maxpool_input_shape);

        tensorflow::Tensor orig_input_tensor(dtype, orig_input_tensor_shape);
        tensorflow::Tensor grad_tensor(dtype, grad_tensor_shape);
        tensorflow::Tensor maxpool_input_tensor(dtype, maxpool_input_tensor_shape);

        fillTensorWithDataByType(orig_input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(maxpool_input_tensor, dtype, data, offset, size);

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

        std::vector<int32_t> ksize = {1, ksize_h, ksize_w, 1};
        std::vector<int32_t> strides = {1, stride_h, stride_w, 1};

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();

        auto orig_input_placeholder = tensorflow::ops::Placeholder(root.WithOpName("orig_input"), dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root.WithOpName("grad"), dtype);
        auto maxpool_input_placeholder = tensorflow::ops::Placeholder(root.WithOpName("maxpool_input"), dtype);

        auto maxpool_grad_grad = tensorflow::ops::MaxPoolGradGradV2(
            root.WithOpName("maxpool_grad_grad"),
            orig_input_placeholder,
            grad_placeholder,
            maxpool_input_placeholder,
            ksize,
            strides,
            "VALID"
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
            {"orig_input", orig_input_tensor},
            {"grad", grad_tensor},
            {"maxpool_input", maxpool_input_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"maxpool_grad_grad"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MaxPoolGradGradV2 executed successfully" << std::endl;
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "MaxPoolGradGradV2 execution failed: " << status.ToString() << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}