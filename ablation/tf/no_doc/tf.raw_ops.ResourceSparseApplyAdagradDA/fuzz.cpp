#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_var.h>
#include <tensorflow/core/kernels/training_ops.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/platform/cpu_info.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType var_dtype = parseDataType(data[offset++]);
        uint8_t var_rank = parseRank(data[offset++]);
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        
        tensorflow::DataType accum_dtype = var_dtype;
        uint8_t accum_rank = var_rank;
        std::vector<int64_t> accum_shape = var_shape;
        
        tensorflow::DataType squared_accum_dtype = var_dtype;
        uint8_t squared_accum_rank = var_rank;
        std::vector<int64_t> squared_accum_shape = var_shape;
        
        tensorflow::DataType grad_dtype = var_dtype;
        uint8_t grad_rank = var_rank;
        std::vector<int64_t> grad_shape = var_shape;
        
        uint8_t indices_rank = 1;
        std::vector<int64_t> indices_shape = {var_shape.empty() ? 1 : var_shape[0]};
        
        if (offset >= size) return 0;
        
        float lr_value = 0.01f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&lr_value, data + offset, sizeof(float));
            offset += sizeof(float);
            lr_value = std::abs(lr_value);
            if (lr_value > 1.0f) lr_value = 0.01f;
        }
        
        float l1_value = 0.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l1_value, data + offset, sizeof(float));
            offset += sizeof(float);
            l1_value = std::abs(l1_value);
            if (l1_value > 1.0f) l1_value = 0.0f;
        }
        
        float l2_value = 0.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l2_value, data + offset, sizeof(float));
            offset += sizeof(float);
            l2_value = std::abs(l2_value);
            if (l2_value > 1.0f) l2_value = 0.0f;
        }
        
        int64_t global_step_value = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&global_step_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            global_step_value = std::abs(global_step_value);
            if (global_step_value == 0) global_step_value = 1;
        }

        tensorflow::TensorShape var_tensor_shape(var_shape);
        tensorflow::TensorShape accum_tensor_shape(accum_shape);
        tensorflow::TensorShape squared_accum_tensor_shape(squared_accum_shape);
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape lr_tensor_shape({});
        tensorflow::TensorShape l1_tensor_shape({});
        tensorflow::TensorShape l2_tensor_shape({});
        tensorflow::TensorShape global_step_tensor_shape({});

        tensorflow::Tensor var_tensor(var_dtype, var_tensor_shape);
        tensorflow::Tensor accum_tensor(accum_dtype, accum_tensor_shape);
        tensorflow::Tensor squared_accum_tensor(squared_accum_dtype, squared_accum_tensor_shape);
        tensorflow::Tensor grad_tensor(grad_dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, lr_tensor_shape);
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, l1_tensor_shape);
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, l2_tensor_shape);
        tensorflow::Tensor global_step_tensor(tensorflow::DT_INT64, global_step_tensor_shape);

        fillTensorWithDataByType(var_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, accum_dtype, data, offset, size);
        fillTensorWithDataByType(squared_accum_tensor, squared_accum_dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, grad_dtype, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t idx;
                std::memcpy(&idx, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                indices_flat(i) = std::abs(idx) % (var_shape.empty() ? 1 : static_cast<int32_t>(var_shape[0]));
            } else {
                indices_flat(i) = 0;
            }
        }
        
        lr_tensor.scalar<float>()() = lr_value;
        l1_tensor.scalar<float>()() = l1_value;
        l2_tensor.scalar<float>()() = l2_value;
        global_step_tensor.scalar<int64_t>()() = global_step_value;

        std::cout << "var_tensor shape: ";
        for (int i = 0; i < var_tensor.dims(); ++i) {
            std::cout << var_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "grad_tensor shape: ";
        for (int i = 0; i < grad_tensor.dims(); ++i) {
            std::cout << grad_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.dims(); ++i) {
            std::cout << indices_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "lr: " << lr_value << ", l1: " << l1_value << ", l2: " << l2_value << ", global_step: " << global_step_value << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}