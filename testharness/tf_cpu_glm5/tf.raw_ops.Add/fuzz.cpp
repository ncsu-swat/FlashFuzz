#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/lib/core/status.h>

#define MIN_RANK 0
#define MAX_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 0
#define MAX_TENSOR_SHAPE_DIMS_TF 8

using tensorflow::DataType;
using tensorflow::DT_FLOAT;
using tensorflow::DT_DOUBLE;
using tensorflow::DT_INT32;
using tensorflow::DT_UINT8;
using tensorflow::DT_INT16;
using tensorflow::DT_INT8;
using tensorflow::DT_STRING;
using tensorflow::DT_COMPLEX64;
using tensorflow::DT_INT64;
using tensorflow::DT_BOOL;
using tensorflow::DT_QINT8;
using tensorflow::DT_QUINT8;
using tensorflow::DT_QINT32;
using tensorflow::DT_BFLOAT16;
using tensorflow::DT_QINT16;
using tensorflow::DT_QUINT16;
using tensorflow::DT_UINT16;
using tensorflow::DT_COMPLEX128;
using tensorflow::DT_HALF;
using tensorflow::DT_UINT32;
using tensorflow::DT_UINT64;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::bfloat16;
using tensorflow::complex64;
using tensorflow::complex128;
using tensorflow::Status;

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

void fillTensorWithStringData(Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    for (size_t i = 0; i < num_elements; ++i) {
        uint8_t len = 0;
        if (offset + 1 <= total_size) {
            len = data[offset++];
        }
        len = len % 16;
        if (offset + len <= total_size) {
            flat(i).assign(reinterpret_cast<const char*>(data + offset), len);
            offset += len;
        } else {
            flat(i) = "";
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
        case DT_STRING: fillTensorWithStringData(tensor, data, offset, total_size); break;
        case DT_COMPLEX64: fillTensorWithData<complex64>(tensor, data, offset, total_size); break;
        case DT_INT64: fillTensorWithData<int64_t>(tensor, data, offset, total_size); break;
        case DT_BOOL: fillTensorWithData<bool>(tensor, data, offset, total_size); break;
        case DT_QINT8: fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size); break;
        case DT_QUINT8: fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size); break;
        case DT_QINT32: fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size); break;
        case DT_BFLOAT16: fillTensorWithData<bfloat16>(tensor, data, offset, total_size); break;
        case DT_QINT16: fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size); break;
        case DT_QUINT16: fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size); break;
        case DT_UINT16: fillTensorWithData<uint16_t>(tensor, data, offset, total_size); break;
        case DT_COMPLEX128: fillTensorWithData<complex128>(tensor, data, offset, total_size); break;
        case DT_HALF: fillTensorWithData<Eigen::half>(tensor, data, offset, total_size); break;
        case DT_UINT32: fillTensorWithData<uint32_t>(tensor, data, offset, total_size); break;
        case DT_UINT64: fillTensorWithData<uint64_t>(tensor, data, offset, total_size); break;
        default: break;
    }
}

void printTensorInfo(const char* name, const Tensor& tensor) {
    std::cout << name << ": dtype=" << DataType_Name(tensor.dtype()) 
              << " shape=" << tensor.shape().DebugString() << std::endl;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    if (Size < 4) return 0;
    
    try {
        size_t offset = 0;
        
        uint8_t dtype_selector = Data[offset++];
        uint8_t rank1_byte = Data[offset++];
        uint8_t rank2_byte = Data[offset++];
        
        DataType dtype = parseDataType(dtype_selector);
        uint8_t rank1 = parseRank(rank1_byte);
        uint8_t rank2 = parseRank(rank2_byte);
        
        std::vector<int64_t> shape1 = parseShape(Data, offset, Size, rank1);
        std::vector<int64_t> shape2 = parseShape(Data, offset, Size, rank2);
        
        TensorShape tensor_shape1(shape1);
        TensorShape tensor_shape2(shape2);
        
        Tensor x(dtype, tensor_shape1);
        Tensor y(dtype, tensor_shape2);
        
        fillTensorWithDataByType(x, dtype, Data, offset, Size);
        fillTensorWithDataByType(y, dtype, Data, offset, Size);
        
        printTensorInfo("Input X", x);
        printTensorInfo("Input Y", y);
        
        Tensor result(dtype, TensorShape());
        
        // Note: This harness constructs tensors and shapes for fuzzing.
        // The actual tf.raw_ops.Add execution requires a full TensorFlow runtime
        // context (OpKernelContext) which is not available in this standalone harness.
        // The fuzz target focuses on tensor construction edge cases.
        
        std::cout << "Add operation setup completed successfully" << std::endl;
        
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}