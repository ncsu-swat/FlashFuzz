#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 12) {
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
            dtype = tensorflow::DT_HALF;
            break;
        case 11:
            dtype = tensorflow::DT_UINT32;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        if (input_rank < 2) input_rank = 2;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        if (input_shape.size() < 2) return 0;
        
        int64_t M = input_shape[input_shape.size() - 2];
        int64_t N = input_shape[input_shape.size() - 1];
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        if (offset >= size) return 0;
        
        int32_t k_val = 0;
        bool is_k_scalar = true;
        int32_t k_low = 0, k_high = 0;
        
        if (offset + sizeof(uint8_t) <= size) {
            uint8_t k_type = data[offset++];
            if (k_type % 2 == 0) {
                if (offset + sizeof(int32_t) <= size) {
                    std::memcpy(&k_val, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    k_val = k_val % std::min(M, N);
                    k_low = k_high = k_val;
                }
            } else {
                is_k_scalar = false;
                if (offset + 2 * sizeof(int32_t) <= size) {
                    std::memcpy(&k_low, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    std::memcpy(&k_high, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    
                    k_low = k_low % std::min(M, N);
                    k_high = k_high % std::min(M, N);
                    if (k_low > k_high) std::swap(k_low, k_high);
                }
            }
        }
        
        tensorflow::TensorShape k_shape;
        tensorflow::Tensor k_tensor;
        if (is_k_scalar) {
            k_shape = tensorflow::TensorShape({});
            k_tensor = tensorflow::Tensor(tensorflow::DT_INT32, k_shape);
            k_tensor.scalar<int32_t>()() = k_val;
        } else {
            k_shape = tensorflow::TensorShape({2});
            k_tensor = tensorflow::Tensor(tensorflow::DT_INT32, k_shape);
            auto k_flat = k_tensor.flat<int32_t>();
            k_flat(0) = k_low;
            k_flat(1) = k_high;
        }
        
        int64_t max_diag_len = std::min(M + std::min(k_high, 0), N + std::min(-k_low, 0));
        if (max_diag_len <= 0) return 0;
        
        std::vector<int64_t> diagonal_shape = input_shape;
        diagonal_shape.pop_back();
        diagonal_shape.pop_back();
        
        if (is_k_scalar || k_low == k_high) {
            diagonal_shape.push_back(max_diag_len);
        } else {
            int64_t num_diags = k_high - k_low + 1;
            diagonal_shape.push_back(num_diags);
            diagonal_shape.push_back(max_diag_len);
        }
        
        tensorflow::TensorShape diagonal_tensor_shape(diagonal_shape);
        tensorflow::Tensor diagonal_tensor(input_dtype, diagonal_tensor_shape);
        fillTensorWithDataByType(diagonal_tensor, input_dtype, data, offset, size);
        
        std::string align = "RIGHT_LEFT";
        if (offset < size) {
            uint8_t align_selector = data[offset++];
            switch (align_selector % 4) {
                case 0: align = "RIGHT_LEFT"; break;
                case 1: align = "LEFT_RIGHT"; break;
                case 2: align = "LEFT_LEFT"; break;
                case 3: align = "RIGHT_RIGHT"; break;
            }
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto diagonal_op = tensorflow::ops::Const(root, diagonal_tensor);
        auto k_op = tensorflow::ops::Const(root, k_tensor);
        
        auto matrix_set_diag_op = tensorflow::ops::MatrixSetDiagV3(
            root, input_op, diagonal_op, k_op,
            tensorflow::ops::MatrixSetDiagV3::Align(align)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({matrix_set_diag_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}