#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
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
    switch (selector % 7) {
        case 0:
            dtype = tensorflow::DT_UINT8;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
            break;
        case 2:
            dtype = tensorflow::DT_INT64;
            break;
        case 3:
            dtype = tensorflow::DT_HALF;
            break;
        case 4:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 5:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 6:
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
        tensorflow::DataType images_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> images_shape = {1, 2, 2, 1};
        tensorflow::TensorShape images_tensor_shape(images_shape);
        tensorflow::Tensor images_tensor(images_dtype, images_tensor_shape);
        fillTensorWithDataByType(images_tensor, images_dtype, data, offset, size);
        
        std::vector<int64_t> transforms_shape = {1, 8};
        tensorflow::TensorShape transforms_tensor_shape(transforms_shape);
        tensorflow::Tensor transforms_tensor(tensorflow::DT_FLOAT, transforms_tensor_shape);
        fillTensorWithData<float>(transforms_tensor, data, offset, size);
        
        std::vector<int64_t> output_shape_shape = {2};
        tensorflow::TensorShape output_shape_tensor_shape(output_shape_shape);
        tensorflow::Tensor output_shape_tensor(tensorflow::DT_INT32, output_shape_tensor_shape);
        auto output_shape_flat = output_shape_tensor.flat<int32_t>();
        output_shape_flat(0) = 3;
        output_shape_flat(1) = 3;
        
        tensorflow::Tensor fill_value_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto fill_value_scalar = fill_value_tensor.scalar<float>();
        fill_value_scalar() = 0.0f;
        
        std::string interpolation = "NEAREST";
        if (offset < size && data[offset] % 2 == 1) {
            interpolation = "BILINEAR";
        }
        offset++;
        
        std::string fill_mode = "CONSTANT";
        if (offset < size) {
            switch (data[offset] % 4) {
                case 0: fill_mode = "CONSTANT"; break;
                case 1: fill_mode = "REFLECT"; break;
                case 2: fill_mode = "WRAP"; break;
                case 3: fill_mode = "NEAREST"; break;
            }
        }
        
        auto images_placeholder = tensorflow::ops::Placeholder(root, images_dtype);
        auto transforms_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto output_shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto fill_value_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Use raw_ops namespace for ImageProjectiveTransformV3
        auto result = tensorflow::ops::internal::ImageProjectiveTransformV3(
            root,
            images_placeholder,
            transforms_placeholder,
            output_shape_placeholder,
            fill_value_placeholder,
            interpolation,
            fill_mode
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{images_placeholder, images_tensor},
             {transforms_placeholder, transforms_tensor},
             {output_shape_placeholder, output_shape_tensor},
             {fill_value_placeholder, fill_value_tensor}},
            {result},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
