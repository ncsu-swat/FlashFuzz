#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 21) {  
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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
            dtype = tensorflow::DT_UINT64;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType element_dtype = parseDataType(data[offset++]);
        
        uint8_t element_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> element_shape_dims = parseShape(data, offset, size, element_shape_rank);
        
        if (offset >= size) return 0;
        
        int32_t max_num_elements_val;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_num_elements_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_num_elements_val = std::abs(max_num_elements_val) % 100 + 1;
        } else {
            max_num_elements_val = 10;
        }

        tensorflow::TensorShape element_shape_tensor_shape({static_cast<int64_t>(element_shape_dims.size())});
        tensorflow::Tensor element_shape_tensor(tensorflow::DT_INT64, element_shape_tensor_shape);
        auto element_shape_flat = element_shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < element_shape_dims.size(); ++i) {
            element_shape_flat(i) = element_shape_dims[i];
        }

        tensorflow::TensorShape max_num_elements_shape({});
        tensorflow::Tensor max_num_elements_tensor(tensorflow::DT_INT32, max_num_elements_shape);
        max_num_elements_tensor.scalar<int32_t>()() = max_num_elements_val;

        std::cout << "element_dtype: " << tensorflow::DataTypeString(element_dtype) << std::endl;
        std::cout << "element_shape: ";
        for (auto dim : element_shape_dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "max_num_elements: " << max_num_elements_val << std::endl;

        auto element_shape_op = tensorflow::ops::Const(root, element_shape_tensor);
        auto max_num_elements_op = tensorflow::ops::Const(root, max_num_elements_tensor);

        auto empty_tensor_list = tensorflow::ops::EmptyTensorList(
            root, 
            element_shape_op, 
            max_num_elements_op, 
            element_dtype
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({empty_tensor_list}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "Output tensor dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
