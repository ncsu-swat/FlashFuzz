#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>

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

void fillTensorWithStringData(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + 1 <= total_size) {
            uint8_t len = data[offset++] % 8;
            if (offset + len <= total_size) {
                flat(i).assign(reinterpret_cast<const char*>(data + offset), len);
                offset += len;
            } else {
                flat(i) = "";
            }
        } else {
            flat(i) = "";
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor, tensorflow::DataType dtype, const uint8_t* data, size_t& offset, size_t total_size) {
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
        case tensorflow::DT_STRING:
            fillTensorWithStringData(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::vector<int64_t> generateReshapeShape(const uint8_t* data, size_t& offset, size_t total_size, int64_t num_elements) {
    std::vector<int64_t> shape;
    if (offset >= total_size) {
        shape.push_back(num_elements > 0 ? num_elements : 1);
        return shape;
    }

    uint8_t rank_byte = data[offset++];
    uint8_t rank = rank_byte % (MAX_RANK + 1);

    if (rank == 0) {
        if (num_elements == 1 || num_elements == 0) {
            return {};
        }
        shape.push_back(num_elements);
        return shape;
    }

    shape.reserve(rank);
    int neg_one_idx = -1;

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof(int64_t) <= total_size) {
            int64_t dim;
            std::memcpy(&dim, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);

            int64_t dim_val = dim % 10 - 3;
            if (dim_val == 0) dim_val = 1;

            if (dim_val == -1) {
                if (neg_one_idx == -1) {
                    neg_one_idx = i;
                    shape.push_back(-1);
                } else {
                    shape.push_back(1);
                }
            } else {
                shape.push_back(dim_val);
            }
        } else {
            shape.push_back(1);
        }
    }

    return shape;
}

void printTensor(const tensorflow::Tensor& tensor, const char* name) {
    std::cout << name << ": dtype=" << tensorflow::DataTypeString(tensor.dtype())
              << ", shape=" << tensor.shape().DebugString()
              << ", num_elements=" << tensor.NumElements() << std::endl;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 4) return 0;

        size_t offset = 0;

        uint8_t dtype_selector = Data[offset++];
        uint8_t rank_byte = (offset < Size) ? Data[offset++] : 0;
        uint8_t shape_rank_byte = (offset < Size) ? Data[offset++] : 0;

        tensorflow::DataType dtype = parseDataType(dtype_selector);
        uint8_t tensor_rank = parseRank(rank_byte);

        std::vector<int64_t> tensor_shape = parseShape(Data, offset, Size, tensor_rank);

        tensorflow::TensorShape shape(tensor_shape);
        tensorflow::Tensor tensor(dtype, shape);

        if (tensor.NumElements() > 1000000) return 0;

        fillTensorWithDataByType(tensor, dtype, Data, offset, Size);

        printTensor(tensor, "Input tensor");

        tensorflow::DataType shape_dtype = (offset < Size && Data[offset++] % 2 == 0)
            ? tensorflow::DT_INT32 : tensorflow::DT_INT64;

        std::vector<int64_t> reshape_dims = generateReshapeShape(Data, offset, Size, tensor.NumElements());

        tensorflow::Tensor shape_tensor(shape_dtype, tensorflow::TensorShape({static_cast<int64_t>(reshape_dims.size())}));

        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat = shape_tensor.flat<int32_t>();
            for (size_t i = 0; i < reshape_dims.size(); ++i) {
                flat(i) = static_cast<int32_t>(reshape_dims[i]);
            }
        } else {
            auto flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < reshape_dims.size(); ++i) {
                flat(i) = reshape_dims[i];
            }
        }

        printTensor(shape_tensor, "Shape tensor");

        std::cout << "Reshape dims: [";
        for (size_t i = 0; i < reshape_dims.size(); ++i) {
            std::cout << reshape_dims[i];
            if (i < reshape_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Build Graph using Scope + ClientSession pattern
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();

        auto input_op = tensorflow::ops::Const(root, tensor);
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto reshape_op = tensorflow::ops::Reshape(root, input_op, shape_op);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({reshape_op}, &outputs);

        if (status.ok()) {
            if (!outputs.empty()) {
                printTensor(outputs[0], "Output tensor");
            }
        } else {
            std::cout << "Status: " << status.ToString() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
