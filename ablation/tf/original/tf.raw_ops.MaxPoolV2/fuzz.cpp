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
    switch (selector % 11) {
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
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
        case 5:
            dtype = tensorflow::DT_INT64;
            break;
        case 6:
            dtype = tensorflow::DT_UINT8;
            break;
        case 7:
            dtype = tensorflow::DT_INT16;
            break;
        case 8:
            dtype = tensorflow::DT_INT8;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
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
        uint8_t input_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        std::vector<int32_t> ksize_data = {1, 2, 2, 1};
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                ksize_data[i] = std::abs(val) % 5 + 1;
            }
        }
        
        tensorflow::Tensor ksize_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto ksize_flat = ksize_tensor.flat<int32_t>();
        for (int i = 0; i < 4; ++i) {
            ksize_flat(i) = ksize_data[i];
        }
        
        std::vector<int32_t> strides_data = {1, 1, 1, 1};
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                strides_data[i] = std::abs(val) % 5 + 1;
            }
        }
        
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto strides_flat = strides_tensor.flat<int32_t>();
        for (int i = 0; i < 4; ++i) {
            strides_flat(i) = strides_data[i];
        }
        
        std::string padding = "VALID";
        if (offset < size) {
            if (data[offset++] % 2 == 0) {
                padding = "SAME";
            }
        }
        
        std::string data_format = "NHWC";
        if (offset < size) {
            uint8_t format_selector = data[offset++] % 3;
            if (format_selector == 1) {
                data_format = "NCHW";
            } else if (format_selector == 2) {
                data_format = "NCHW_VECT_C";
            }
        }
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Ksize: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << ksize_data[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Strides: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << strides_data[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Padding: " << padding << std::endl;
        std::cout << "Data format: " << data_format << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto ksize_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto strides_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        auto maxpool_op = tensorflow::ops::MaxPoolV2(
            root,
            input_placeholder,
            ksize_placeholder,
            strides_placeholder,
            padding,
            tensorflow::ops::MaxPoolV2::DataFormat(data_format)
        );
        
        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {ksize_placeholder.node()->name(), ksize_tensor},
            {strides_placeholder.node()->name(), strides_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {maxpool_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MaxPoolV2 operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MaxPoolV2 operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}