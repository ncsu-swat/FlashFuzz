#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/types.h>

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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

void printTensorShape(const tensorflow::Tensor& tensor, const std::string& name) {
    std::cout << name << " shape: " << tensor.shape().DebugString() << " dtype: " << tensor.dtype() << std::endl;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    if (Size < 4) return 0;
    
    try {
        size_t offset = 0;
        
        uint8_t dtype_selector = Data[offset++];
        uint8_t rank_byte = Data[offset++];
        uint8_t num_tensors_byte = Data[offset++];
        uint8_t concat_dim_byte = Data[offset++];
        
        tensorflow::DataType dtype = parseDataType(dtype_selector);
        uint8_t rank = parseRank(rank_byte);
        int num_tensors = 2 + (num_tensors_byte % 4);
        int32_t concat_dim = static_cast<int32_t>(concat_dim_byte % (rank > 0 ? rank : 1));
        
        std::cout << "Fuzz input: dtype=" << dtype << ", rank=" << static_cast<int>(rank) 
                  << ", num_tensors=" << num_tensors << ", concat_dim=" << concat_dim << std::endl;
        
        std::vector<int64_t> base_shape = parseShape(Data, offset, Size, rank);
        
        std::cout << "Base shape: [";
        for (size_t i = 0; i < base_shape.size(); ++i) {
            std::cout << base_shape[i];
            if (i < base_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::vector<tensorflow::Tensor> values;
        values.reserve(num_tensors);
        
        for (int t = 0; t < num_tensors; ++t) {
            std::vector<int64_t> tensor_shape = base_shape;
            
            if (rank > 0 && static_cast<int>(concat_dim) < static_cast<int>(tensor_shape.size())) {
                if (tensor_shape[concat_dim] >= 0) {
                    tensor_shape[concat_dim] = 1 + (t % 3);
                }
            }
            
            tensorflow::TensorShape shape;
            for (const auto& dim : tensor_shape) {
                shape.AddDim(dim >= 0 ? dim : 0);
            }
            
            tensorflow::Tensor tensor(dtype, shape);
            fillTensorWithDataByType(tensor, dtype, Data, offset, Size);
            
            printTensorShape(tensor, "Tensor " + std::to_string(t));
            values.push_back(std::move(tensor));
        }
        
        tensorflow::Tensor concat_dim_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        concat_dim_tensor.scalar<int32_t>()() = concat_dim;
        
        std::cout << "Concat dim tensor: " << concat_dim << std::endl;
        
        tensorflow::Status s;
        tensorflow::Tensor output;
        
        for (const auto& t : values) {
            std::cout << "Input tensor: " << t.DebugString() << std::endl;
        }
        
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}