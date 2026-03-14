#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/public/session.h"

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

namespace tensorflow {

DataType parseDataType(uint8_t selector) {
    DataType dtype;
    switch (selector % 23) {
        case 0: dtype = DT_FLOAT; break;
        case 1: dtype = DT_DOUBLE; break;
        case 2: dtype = DT_INT32; break;
        case 3: dtype = DT_UINT8; break;
        case 4: dtype = DT_INT16; break;
        case 5: dtype = DT_INT8; break;
        case 6: dtype = DT_STRING; break;
        case 7: dtype = DT_COMPLEX64; break;
        case 8: dtype = DT_INT64; break;
        case 9: dtype = DT_BOOL; break;
        case 10: dtype = DT_QINT8; break;
        case 11: dtype = DT_QUINT8; break;
        case 12: dtype = DT_QINT32; break;
        case 13: dtype = DT_BFLOAT16; break;
        case 14: dtype = DT_QINT16; break;
        case 15: dtype = DT_QUINT16; break;
        case 16: dtype = DT_UINT16; break;
        case 17: dtype = DT_COMPLEX128; break;
        case 18: dtype = DT_HALF; break;
        case 19: dtype = DT_UINT32; break;
        case 20: dtype = DT_UINT64; break;
        default: dtype = DT_FLOAT; break;
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
void fillTensorWithData(Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
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

void fillTensorWithDataByType(Tensor& tensor, DataType dtype, const uint8_t* data, size_t& offset, size_t total_size) {
    switch (dtype) {
        case DT_FLOAT: fillTensorWithData<float>(tensor, data, offset, total_size); break;
        case DT_DOUBLE: fillTensorWithData<double>(tensor, data, offset, total_size); break;
        case DT_INT32: fillTensorWithData<int32_t>(tensor, data, offset, total_size); break;
        case DT_UINT8: fillTensorWithData<uint8_t>(tensor, data, offset, total_size); break;
        case DT_INT16: fillTensorWithData<int16_t>(tensor, data, offset, total_size); break;
        case DT_INT8: fillTensorWithData<int8_t>(tensor, data, offset, total_size); break;
        case DT_INT64: fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
        case DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
        case DT_UINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
        case DT_UINT32: fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
        case DT_UINT64: fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
        case DT_BFLOAT16: fillTensorWithData<bfloat16>(tensor, data, offset, total_size); break;
        case DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
        case DT_COMPLEX64: fillTensorWithData<complex64>(tensor, data, offset, total_size); break;
        case DT_COMPLEX128: fillTensorWithData<complex128>(tensor, data, offset, total_size); break;
        default: break;
    }
}

bool isMatMulSupportedDtype(DataType dtype) {
    switch (dtype) {
        case DT_BFLOAT16:
        case DT_HALF:
        case DT_FLOAT:
        case DT_DOUBLE:
        case DT_INT32:
        case DT_INT64:
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
        case DT_COMPLEX64:
        case DT_COMPLEX128:
            return true;
        default:
            return false;
    }
}

} // namespace tensorflow

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    using namespace tensorflow;

    if (Size < 5) {
        return 0;
    }

    try {
        size_t offset = 0;

        // Parse dtype selector
        uint8_t dtype_selector = Data[offset++];
        DataType dtype = parseDataType(dtype_selector);

        // Skip unsupported dtypes for MatMul
        if (!isMatMulSupportedDtype(dtype)) {
            return 0;
        }

        // Parse ranks for tensors a and b
        uint8_t rank_a_byte = (offset < Size) ? Data[offset++] : 2;
        uint8_t rank_b_byte = (offset < Size) ? Data[offset++] : 2;

        uint8_t rank_a = parseRank(rank_a_byte);
        uint8_t rank_b = parseRank(rank_b_byte);

        // Parse transpose flags
        bool transpose_a = (offset < Size) ? (Data[offset++] & 0x01) : false;
        bool transpose_b = (offset < Size) ? (Data[offset++] & 0x01) : false;

        // Parse shapes
        std::vector<int64_t> shape_a = parseShape(Data, offset, Size, rank_a);
        std::vector<int64_t> shape_b = parseShape(Data, offset, Size, rank_b);

        // Print debug info
        std::cout << "MatMul Fuzz Input:" << std::endl;
        std::cout << "  dtype: " << DataType_Name(dtype) << std::endl;
        std::cout << "  rank_a: " << static_cast<int>(rank_a) << std::endl;
        std::cout << "  rank_b: " << static_cast<int>(rank_b) << std::endl;
        std::cout << "  shape_a: [";
        for (size_t i = 0; i < shape_a.size(); ++i) {
            std::cout << shape_a[i];
            if (i < shape_a.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  shape_b: [";
        for (size_t i = 0; i < shape_b.size(); ++i) {
            std::cout << shape_b[i];
            if (i < shape_b.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  transpose_a: " << transpose_a << std::endl;
        std::cout << "  transpose_b: " << transpose_b << std::endl;

        // Create tensors
        TensorShape tensor_shape_a(shape_a);
        TensorShape tensor_shape_b(shape_b);

        Tensor tensor_a(dtype, tensor_shape_a);
        Tensor tensor_b(dtype, tensor_shape_b);

        // Fill tensors with data
        fillTensorWithDataByType(tensor_a, dtype, Data, offset, Size);
        fillTensorWithDataByType(tensor_b, dtype, Data, offset, Size);

        std::cout << "  tensor_a shape: " << tensor_a.shape().DebugString() << std::endl;
        std::cout << "  tensor_b shape: " << tensor_b.shape().DebugString() << std::endl;

        // Build the graph using Scope + ClientSession
        Scope root = Scope::NewRootScope();

        auto op = ops::MatMul(root, tensor_a, tensor_b,
                              ops::MatMul::TransposeA(transpose_a)
                              .TransposeB(transpose_b));

        // Run the session
        ClientSession session(root);
        std::vector<Tensor> outputs;

        Status status = session.Run({op}, &outputs);

        if (!status.ok()) {
            std::cout << "  MatMul error: " << status.ToString() << std::endl;
        } else {
            std::cout << "  MatMul completed successfully" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }

    return 0;
}
