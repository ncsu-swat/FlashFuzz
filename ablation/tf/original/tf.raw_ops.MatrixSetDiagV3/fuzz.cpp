#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/lib/core/status.h>

constexpr uint8_t MIN_RANK = 2;
constexpr uint8_t MAX_RANK = 6;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT16;
            break;
        case 11:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 12:
            dtype = tensorflow::DT_HALF;
            break;
        case 13:
            dtype = tensorflow::DT_UINT32;
            break;
        case 14:
            dtype = tensorflow::DT_UINT64;
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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        if (input_shape.size() < 2) {
            return 0;
        }
        
        int64_t M = input_shape[input_shape.size() - 2];
        int64_t N = input_shape[input_shape.size() - 1];
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(dtype, input_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        std::cout << "Input tensor shape: ";
        for (auto dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        if (offset >= size) {
            return 0;
        }
        
        uint8_t k_type = data[offset++] % 2;
        std::vector<int32_t> k_values;
        
        if (k_type == 0) {
            if (offset + sizeof(int32_t) > size) {
                return 0;
            }
            int32_t k_val;
            std::memcpy(&k_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            k_val = k_val % (M + N - 1) - (M - 1);
            k_values.push_back(k_val);
        } else {
            if (offset + 2 * sizeof(int32_t) > size) {
                return 0;
            }
            int32_t k0, k1;
            std::memcpy(&k0, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            std::memcpy(&k1, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            k0 = k0 % (M + N - 1) - (M - 1);
            k1 = k1 % (M + N - 1) - (M - 1);
            
            if (k0 > k1) {
                std::swap(k0, k1);
            }
            
            k_values.push_back(k0);
            k_values.push_back(k1);
        }
        
        tensorflow::TensorShape k_shape({static_cast<int64_t>(k_values.size())});
        tensorflow::Tensor k_tensor(tensorflow::DT_INT32, k_shape);
        auto k_flat = k_tensor.flat<int32_t>();
        for (size_t i = 0; i < k_values.size(); ++i) {
            k_flat(i) = k_values[i];
        }
        
        std::cout << "K values: ";
        for (auto k : k_values) {
            std::cout << k << " ";
        }
        std::cout << std::endl;
        
        int32_t k0 = k_values[0];
        int32_t k1 = (k_values.size() == 1) ? k0 : k_values[1];
        
        int64_t max_diag_len = std::min(M + std::min(k1, 0), N + std::min(-k0, 0));
        
        std::vector<int64_t> diagonal_shape = input_shape;
        diagonal_shape.pop_back();
        diagonal_shape.pop_back();
        
        if (k_values.size() == 1 || k0 == k1) {
            diagonal_shape.push_back(max_diag_len);
        } else {
            int64_t num_diags = k1 - k0 + 1;
            diagonal_shape.push_back(num_diags);
            diagonal_shape.push_back(max_diag_len);
        }
        
        tensorflow::TensorShape diagonal_tensor_shape(diagonal_shape);
        tensorflow::Tensor diagonal_tensor(dtype, diagonal_tensor_shape);
        
        fillTensorWithDataByType(diagonal_tensor, dtype, data, offset, size);
        
        std::cout << "Diagonal tensor shape: ";
        for (auto dim : diagonal_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::string align = "RIGHT_LEFT";
        if (offset < size) {
            uint8_t align_selector = data[offset++] % 4;
            switch (align_selector) {
                case 0:
                    align = "RIGHT_LEFT";
                    break;
                case 1:
                    align = "LEFT_RIGHT";
                    break;
                case 2:
                    align = "LEFT_LEFT";
                    break;
                case 3:
                    align = "RIGHT_RIGHT";
                    break;
            }
        }
        
        std::cout << "Align: " << align << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto diagonal_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto k_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        auto matrix_set_diag = tensorflow::ops::MatrixSetDiagV3(
            root, input_placeholder, diagonal_placeholder, k_placeholder,
            tensorflow::ops::MatrixSetDiagV3::Align(align));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {diagonal_placeholder, diagonal_tensor},
             {k_placeholder, k_tensor}},
            {matrix_set_diag}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "MatrixSetDiagV3 operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MatrixSetDiagV3 operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}