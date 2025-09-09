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
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 8) {
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
            dtype = tensorflow::DT_UINT16;
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

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = 4;
        uint8_t filter_rank = 3;
        
        std::vector<int64_t> input_shape = {1, 10, 10, 3};
        std::vector<int64_t> filter_shape = {3, 3, 3};
        
        if (offset + 4 * sizeof(int64_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int64_t dim_val;
                std::memcpy(&dim_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dim_val = 1 + (std::abs(dim_val) % 10);
                input_shape[i] = dim_val;
            }
        }
        
        if (offset + 3 * sizeof(int64_t) <= size) {
            for (int i = 0; i < 3; ++i) {
                int64_t dim_val;
                std::memcpy(&dim_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dim_val = 1 + (std::abs(dim_val) % 5);
                filter_shape[i] = dim_val;
            }
        }

        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);

        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor filter_tensor(filter_dtype, filter_tensor_shape);

        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);

        std::vector<int32_t> strides = {1, 1, 1, 1};
        std::vector<int32_t> rates = {1, 1, 1, 1};
        
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t stride_val;
                std::memcpy(&stride_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                strides[i] = 1 + (std::abs(stride_val) % 3);
            }
        }
        
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t rate_val;
                std::memcpy(&rate_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                rates[i] = 1 + (std::abs(rate_val) % 3);
            }
        }

        std::string padding = (offset < size && data[offset] % 2 == 0) ? "SAME" : "VALID";

        std::cout << "Input tensor shape: ";
        for (auto dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Filter tensor shape: ";
        for (auto dim : filter_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Strides: ";
        for (auto s : strides) {
            std::cout << s << " ";
        }
        std::cout << std::endl;

        std::cout << "Rates: ";
        for (auto r : rates) {
            std::cout << r << " ";
        }
        std::cout << std::endl;

        std::cout << "Padding: " << padding << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, filter_dtype);
        
        auto dilation2d = tensorflow::ops::Dilation2D(root, input_placeholder, filter_placeholder,
                                                     strides, rates, padding);

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {filter_placeholder.node()->name(), filter_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {dilation2d.node()->name()}, {}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "Dilation2D operation completed successfully" << std::endl;
            std::cout << "Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
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