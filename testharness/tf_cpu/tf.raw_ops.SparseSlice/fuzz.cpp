#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 15) {
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
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 11:
            dtype = tensorflow::DT_HALF;
            break;
        case 12:
            dtype = tensorflow::DT_UINT32;
            break;
        case 13:
            dtype = tensorflow::DT_UINT64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        uint8_t sparse_rank = parseRank(data[offset++]);
        std::vector<int64_t> sparse_shape = parseShape(data, offset, size, sparse_rank);
        
        if (offset >= size) return 0;
        
        uint8_t num_entries_byte = data[offset++];
        int64_t num_entries = 1 + (num_entries_byte % 5);
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({num_entries, sparse_rank}));
        auto indices_flat = indices_tensor.flat<int64_t>();
        for (int64_t i = 0; i < num_entries * sparse_rank; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t idx_val;
                std::memcpy(&idx_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                int64_t dim_idx = i % sparse_rank;
                indices_flat(i) = std::abs(idx_val) % sparse_shape[dim_idx];
            } else {
                indices_flat(i) = 0;
            }
        }
        
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape({num_entries}));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({sparse_rank}));
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (int i = 0; i < sparse_rank; ++i) {
            shape_flat(i) = sparse_shape[i];
        }
        
        tensorflow::Tensor start_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({sparse_rank}));
        auto start_flat = start_tensor.flat<int64_t>();
        for (int i = 0; i < sparse_rank; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t start_val;
                std::memcpy(&start_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                start_flat(i) = std::abs(start_val) % sparse_shape[i];
            } else {
                start_flat(i) = 0;
            }
        }
        
        tensorflow::Tensor size_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({sparse_rank}));
        auto size_flat = size_tensor.flat<int64_t>();
        for (int i = 0; i < sparse_rank; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t size_val;
                std::memcpy(&size_val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                size_flat(i) = 1 + (std::abs(size_val) % (sparse_shape[i] - start_flat(i)));
            } else {
                size_flat(i) = 1;
            }
        }
        
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto start_input = tensorflow::ops::Const(root, start_tensor);
        auto size_input = tensorflow::ops::Const(root, size_tensor);
        
        auto sparse_slice_op = tensorflow::ops::SparseSlice(root, indices_input, values_input, shape_input, start_input, size_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_slice_op.output_indices, sparse_slice_op.output_values, sparse_slice_op.output_shape}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
