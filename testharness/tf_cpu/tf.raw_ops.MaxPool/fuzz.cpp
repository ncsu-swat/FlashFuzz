#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

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

std::vector<int> parseKsize(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> ksize(4);
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            uint8_t val = data[offset++];
            ksize[i] = (val % 5) + 1;
        } else {
            ksize[i] = 1;
        }
    }
    return ksize;
}

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides(4);
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            uint8_t val = data[offset++];
            strides[i] = (val % 3) + 1;
        } else {
            strides[i] = 1;
        }
    }
    return strides;
}

std::string parsePadding(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "SAME";
        case 1: return "VALID";
        case 2: return "EXPLICIT";
        default: return "SAME";
    }
}

std::string parseDataFormat(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "NHWC";
        case 1: return "NCHW";
        case 2: return "NCHW_VECT_C";
        default: return "NHWC";
    }
}

std::vector<int> parseExplicitPaddings(const uint8_t* data, size_t& offset, size_t total_size, const std::string& padding) {
    std::vector<int> explicit_paddings;
    if (padding == "EXPLICIT") {
        explicit_paddings.resize(8);
        for (int i = 0; i < 8; ++i) {
            if (offset < total_size) {
                uint8_t val = data[offset++];
                explicit_paddings[i] = val % 3;
            } else {
                explicit_paddings[i] = 0;
            }
        }
    }
    return explicit_paddings;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        auto input = tensorflow::ops::Const(root, input_tensor);
        
        std::vector<int> ksize = parseKsize(data, offset, size);
        std::vector<int> strides = parseStrides(data, offset, size);
        
        uint8_t padding_selector = (offset < size) ? data[offset++] : 0;
        std::string padding = parsePadding(padding_selector);
        
        std::vector<int> explicit_paddings = parseExplicitPaddings(data, offset, size, padding);
        
        uint8_t data_format_selector = (offset < size) ? data[offset++] : 0;
        std::string data_format = parseDataFormat(data_format_selector);
        
        auto maxpool_attrs = tensorflow::ops::MaxPool::Attrs()
            .DataFormat(data_format)
            .ExplicitPaddings(explicit_paddings);
        
        auto maxpool = tensorflow::ops::MaxPool(root, input, ksize, strides, padding, maxpool_attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({maxpool}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}