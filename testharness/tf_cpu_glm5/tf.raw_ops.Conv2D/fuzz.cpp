#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using namespace tensorflow;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 23) {
    case 0: dtype = tensorflow::DT_FLOAT; break;
    case 1: dtype = tensorflow::DT_DOUBLE; break;
    case 2: dtype = tensorflow::DT_INT32; break;
    case 3: dtype = tensorflow::DT_UINT8; break;
    case 4: dtype = tensorflow::DT_INT16; break;
    case 5: dtype = tensorflow::DT_INT8; break;
    case 6: dtype = tensorflow::DT_STRING; break;
    case 7: dtype = tensorflow::DT_COMPLEX64; break;
    case 8: dtype = tensorflow::DT_INT64; break;
    case 9: dtype = tensorflow::DT_BOOL; break;
    case 10: dtype = tensorflow::DT_QINT8; break;
    case 11: dtype = tensorflow::DT_QUINT8; break;
    case 12: dtype = tensorflow::DT_QINT32; break;
    case 13: dtype = tensorflow::DT_BFLOAT16; break;
    case 14: dtype = tensorflow::DT_QINT16; break;
    case 15: dtype = tensorflow::DT_QUINT16; break;
    case 16: dtype = tensorflow::DT_UINT16; break;
    case 17: dtype = tensorflow::DT_COMPLEX128; break;
    case 18: dtype = tensorflow::DT_HALF; break;
    case 19: dtype = tensorflow::DT_UINT32; break;
    case 20: dtype = tensorflow::DT_UINT64; break;
    default: dtype = tensorflow::DT_FLOAT; break;
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
            dim_val = MIN_TENSOR_SHAPE_DIMS_TF + static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) % static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));
            shape.push_back(dim_val);
        } else {
            shape.push_back(1);
        }
    }
    return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor, tensorflow::DataType dtype, const uint8_t* data, size_t& offset, size_t total_size) {
    switch (dtype) {
    case tensorflow::DT_FLOAT: fillTensorWithData<float>(tensor, data, offset, total_size); break;
    case tensorflow::DT_DOUBLE: fillTensorWithData<double>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT32: fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT8: fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT16: fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT8: fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_INT64: fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT32: fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_UINT64: fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
    case tensorflow::DT_BFLOAT16: fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size); break;
    case tensorflow::DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX64: fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size); break;
    case tensorflow::DT_COMPLEX128: fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size); break;
    default: break;
    }
}

void printTensor(const tensorflow::Tensor& tensor, const std::string& name) {
    std::cout << name << ": dtype=" << tensorflow::DataTypeString(tensor.dtype())
              << ", shape=" << tensor.shape().DebugString() << std::endl;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 10) return 0;

        size_t offset = 0;

        // Parse data type (Conv2D supports: half, bfloat16, float32, float64, int32)
        uint8_t dtype_selector = Data[offset++];
        tensorflow::DataType dtype = parseDataType(dtype_selector);

        // Restrict to supported types for Conv2D
        if (dtype != tensorflow::DT_FLOAT && dtype != tensorflow::DT_DOUBLE &&
            dtype != tensorflow::DT_INT32 && dtype != tensorflow::DT_HALF &&
            dtype != tensorflow::DT_BFLOAT16) {
            dtype = tensorflow::DT_FLOAT;
        }

        // Parse input tensor shape (must be 4-D: [batch, in_height, in_width, in_channels])
        uint8_t input_rank = 4; // Conv2D requires 4-D input
        std::vector<int64_t> input_shape = parseShape(Data, offset, Size, input_rank);

        // Ensure minimum dimensions for convolution
        for (int i = 0; i < 4; ++i) {
            if (input_shape[i] < 1) input_shape[i] = 1;
        }

        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, tensorflow::TensorShape(input_shape));
        fillTensorWithDataByType(input_tensor, dtype, Data, offset, Size);

        // Parse filter tensor shape (must be 4-D: [filter_height, filter_width, in_channels, out_channels])
        uint8_t filter_rank = 4; // Conv2D requires 4-D filter
        std::vector<int64_t> filter_shape = parseShape(Data, offset, Size, filter_rank);

        // Ensure filter dimensions are valid
        for (int i = 0; i < 4; ++i) {
            if (filter_shape[i] < 1) filter_shape[i] = 1;
        }

        // Match in_channels between input and filter
        if (input_shape.size() >= 4 && filter_shape.size() >= 4) {
            filter_shape[2] = input_shape[3]; // in_channels must match
        }

        // Create filter tensor
        tensorflow::Tensor filter_tensor(dtype, tensorflow::TensorShape(filter_shape));
        fillTensorWithDataByType(filter_tensor, dtype, Data, offset, Size);

        // Parse strides (length 4)
        std::vector<int> strides = {1, 1, 1, 1};
        for (int i = 0; i < 4; ++i) {
            if (offset + sizeof(int32_t) <= Size) {
                int32_t s;
                std::memcpy(&s, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                strides[i] = std::max(1, std::abs(s) % 4 + 1);
            }
        }
        strides[0] = 1; // batch stride must be 1
        strides[3] = 1; // channels stride must be 1

        // Parse padding
        std::string padding = "VALID";
        if (offset < Size) {
            uint8_t pad_selector = Data[offset++];
            switch (pad_selector % 3) {
                case 0: padding = "VALID"; break;
                case 1: padding = "SAME"; break;
                case 2: padding = "EXPLICIT"; break;
            }
        }

        // Parse explicit_paddings if needed
        std::vector<int> explicit_paddings;
        if (padding == "EXPLICIT") {
            for (int i = 0; i < 8; ++i) {
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t p;
                    std::memcpy(&p, Data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    explicit_paddings.push_back(std::abs(p) % 4);
                } else {
                    explicit_paddings.push_back(0);
                }
            }
        }

        // Parse data_format
        std::string data_format = "NHWC";
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            if (format_selector % 2 == 0) {
                data_format = "NHWC";
            } else {
                data_format = "NCHW";
            }
        }

        // Parse dilations (length 4)
        std::vector<int> dilations = {1, 1, 1, 1};
        for (int i = 0; i < 4; ++i) {
            if (offset + sizeof(int32_t) <= Size) {
                int32_t d;
                std::memcpy(&d, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dilations[i] = std::max(1, std::abs(d) % 3 + 1);
            }
        }
        dilations[0] = 1; // batch dilation must be 1
        dilations[3] = 1; // channels dilation must be 1

        // Print inputs for debugging
        std::cout << "=== Conv2D Fuzz Inputs ===" << std::endl;
        printTensor(input_tensor, "input");
        printTensor(filter_tensor, "filter");
        std::cout << "strides: [" << strides[0] << ", " << strides[1] << ", " << strides[2] << ", " << strides[3] << "]" << std::endl;
        std::cout << "padding: " << padding << std::endl;
        std::cout << "data_format: " << data_format << std::endl;
        std::cout << "dilations: [" << dilations[0] << ", " << dilations[1] << ", " << dilations[2] << ", " << dilations[3] << "]" << std::endl;

        // Build Graph using Scope + ClientSession pattern
        Scope root = Scope::NewRootScope();

        auto input_node = ops::Const(root.WithOpName("input"), input_tensor);
        auto filter_node = ops::Const(root.WithOpName("filter"), filter_tensor);

        auto conv = ops::Conv2D(root.WithOpName("conv"), input_node, filter_node,
                                strides, padding,
                                ops::Conv2D::DataFormat(data_format)
                                    .Dilations(dilations)
                                    .ExplicitPaddings(explicit_paddings));

        // Execute
        ClientSession session(root);
        std::vector<Tensor> outputs;

        Status status = session.Run({conv}, &outputs);
        if (!status.ok()) {
            std::cout << "Conv2D error: " << status.ToString() << std::endl;
        }

        std::cout << "=========================" << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    return 0;
}
