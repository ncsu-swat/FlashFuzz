#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/random_ops.h"
#include <iostream>
#include <cstring>
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

tensorflow::DataType parseShapeDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseAlphaDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType shape_dtype = parseShapeDataType(data[offset++]);
        tensorflow::DataType alpha_dtype = parseAlphaDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        uint8_t alpha_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        std::vector<int64_t> alpha_dims = parseShape(data, offset, size, alpha_rank);
        
        tensorflow::TensorShape shape_tensor_shape;
        for (int64_t dim : shape_dims) {
            shape_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape alpha_tensor_shape;
        for (int64_t dim : alpha_dims) {
            alpha_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        tensorflow::Tensor alpha_tensor(alpha_dtype, alpha_tensor_shape);
        
        fillTensorWithDataByType(shape_tensor, shape_dtype, data, offset, size);
        fillTensorWithDataByType(alpha_tensor, alpha_dtype, data, offset, size);
        
        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat = shape_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % 100 + 1;
            }
        } else if (shape_dtype == tensorflow::DT_INT64) {
            auto flat = shape_tensor.flat<int64_t>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % 100 + 1;
            }
        }
        
        if (alpha_dtype == tensorflow::DT_FLOAT) {
            auto flat = alpha_tensor.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) + 0.1f;
            }
        } else if (alpha_dtype == tensorflow::DT_DOUBLE) {
            auto flat = alpha_tensor.flat<double>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) + 0.1;
            }
        } else if (alpha_dtype == tensorflow::DT_HALF) {
            auto flat = alpha_tensor.flat<Eigen::half>();
            for (int i = 0; i < flat.size(); ++i) {
                float val = static_cast<float>(flat(i));
                flat(i) = Eigen::half(std::abs(val) + 0.1f);
            }
        }
        
        int seed = 0;
        int seed2 = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto alpha_input = tensorflow::ops::Const(root, alpha_tensor);
        
        auto random_gamma_op = tensorflow::ops::RandomGamma(root, shape_input, alpha_input,
                                                           tensorflow::ops::RandomGamma::Seed(seed).Seed2(seed2));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({random_gamma_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}