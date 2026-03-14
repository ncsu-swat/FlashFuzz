#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/tstring.h"

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

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

void fillTensorWithString(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + 1 <= total_size) {
            uint8_t len = data[offset] % 8;
            offset++;
            if (offset + len <= total_size) {
                flat(i) = std::string(reinterpret_cast<const char*>(data + offset), len);
                offset += len;
            } else {
                flat(i) = "";
            }
        } else {
            flat(i) = "";
        }
    }
}

std::vector<int32_t> parseIndexVector(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    std::vector<int32_t> vec;
    vec.reserve(rank);
    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof(int32_t) <= total_size) {
            int32_t val;
            std::memcpy(&val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            vec.push_back(val);
        } else {
            vec.push_back(0);
        }
    }
    return vec;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 4) return 0;
        size_t offset = 0;

        uint8_t dtype_selector = Data[offset++];
        uint8_t rank_byte = Data[offset++];
        uint8_t index_dtype_selector = Data[offset++];
        uint8_t size_flag = Data[offset++];

        tensorflow::DataType input_dtype = parseDataType(dtype_selector);
        uint8_t rank = parseRank(rank_byte);

        tensorflow::DataType index_dtype = (index_dtype_selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;

        std::vector<int64_t> input_shape = parseShape(Data, offset, Size, rank);

        std::cout << "Input dtype: " << tensorflow::DataTypeString(input_dtype)
                  << ", Rank: " << static_cast<int>(rank)
                  << ", Shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));

        if (input_dtype == tensorflow::DT_STRING) {
            fillTensorWithString(input_tensor, Data, offset, Size);
        } else {
            fillTensorWithDataByType(input_tensor, input_dtype, Data, offset, Size);
        }

        std::cout << "Input tensor: " << input_tensor.DebugString() << std::endl;

        tensorflow::Tensor begin_tensor(index_dtype, tensorflow::TensorShape({static_cast<int64_t>(rank)}));
        tensorflow::Tensor size_tensor(index_dtype, tensorflow::TensorShape({static_cast<int64_t>(rank)}));

        std::vector<int32_t> begin_vec = parseIndexVector(Data, offset, Size, rank);
        std::vector<int32_t> size_vec = parseIndexVector(Data, offset, Size, rank);

        if (size_flag % 4 == 0) {
            for (auto& s : size_vec) s = -1;
        }

        if (index_dtype == tensorflow::DT_INT32) {
            std::copy(begin_vec.begin(), begin_vec.end(), begin_tensor.flat<int32_t>().data());
            std::copy(size_vec.begin(), size_vec.end(), size_tensor.flat<int32_t>().data());
        } else {
            auto begin_flat = begin_tensor.flat<int64_t>();
            auto size_flat = size_tensor.flat<int64_t>();
            for (size_t i = 0; i < rank; ++i) {
                begin_flat(i) = static_cast<int64_t>(begin_vec[i]);
                size_flat(i) = static_cast<int64_t>(size_vec[i]);
            }
        }

        std::cout << "Begin tensor: " << begin_tensor.DebugString() << std::endl;
        std::cout << "Size tensor: " << size_tensor.DebugString() << std::endl;

        // Build Graph using Scope + ClientSession pattern
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();

        auto input_node = tensorflow::ops::Const(root, input_tensor);
        auto begin_node = tensorflow::ops::Const(root, begin_tensor);
        auto size_node = tensorflow::ops::Const(root, size_tensor);

        auto slice_op = tensorflow::ops::Slice(root, input_node, begin_node, size_node);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({slice_op}, &outputs);

        if (!status.ok()) {
            std::cout << "Status: " << status.ToString() << std::endl;
        } else {
            if (!outputs.empty()) {
                std::cout << "Output tensor: " << outputs[0].DebugString() << std::endl;
            }
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
