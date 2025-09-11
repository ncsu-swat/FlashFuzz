#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
    switch (selector % 10) {
        case 0:
            dtype = tensorflow::DT_INT8;
            break;
        case 1:
            dtype = tensorflow::DT_UINT8;
            break;
        case 2:
            dtype = tensorflow::DT_INT16;
            break;
        case 3:
            dtype = tensorflow::DT_UINT16;
            break;
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
        case 5:
            dtype = tensorflow::DT_INT64;
            break;
        case 6:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 7:
            dtype = tensorflow::DT_HALF;
            break;
        case 8:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 9:
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
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType images_dtype = parseDataType(data[offset++]);
        uint8_t images_rank = parseRank(data[offset++]);
        
        if (images_rank < 3) images_rank = 4;
        
        std::vector<int64_t> images_shape = parseShape(data, offset, size, images_rank);
        if (images_shape.size() < 3) {
            images_shape = {2, 10, 10, 3};
        }
        
        tensorflow::Tensor images_tensor(images_dtype, tensorflow::TensorShape(images_shape));
        fillTensorWithDataByType(images_tensor, images_dtype, data, offset, size);
        
        std::vector<int64_t> size_shape = {2};
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(size_shape));
        auto size_flat = size_tensor.flat<int32_t>();
        if (offset + sizeof(int32_t) <= size) {
            int32_t h_val;
            std::memcpy(&h_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            size_flat(0) = std::abs(h_val) % 100 + 1;
        } else {
            size_flat(0) = 10;
        }
        if (offset + sizeof(int32_t) <= size) {
            int32_t w_val;
            std::memcpy(&w_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            size_flat(1) = std::abs(w_val) % 100 + 1;
        } else {
            size_flat(1) = 10;
        }
        
        std::vector<int64_t> scale_shape = {2};
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        auto scale_flat = scale_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float scale_x;
            std::memcpy(&scale_x, data + offset, sizeof(float));
            offset += sizeof(float);
            scale_flat(0) = std::abs(scale_x) + 0.1f;
        } else {
            scale_flat(0) = 1.0f;
        }
        if (offset + sizeof(float) <= size) {
            float scale_y;
            std::memcpy(&scale_y, data + offset, sizeof(float));
            offset += sizeof(float);
            scale_flat(1) = std::abs(scale_y) + 0.1f;
        } else {
            scale_flat(1) = 1.0f;
        }
        
        std::vector<int64_t> translation_shape = {2};
        tensorflow::Tensor translation_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(translation_shape));
        auto translation_flat = translation_tensor.flat<float>();
        if (offset + sizeof(float) <= size) {
            float trans_x;
            std::memcpy(&trans_x, data + offset, sizeof(float));
            offset += sizeof(float);
            translation_flat(0) = trans_x;
        } else {
            translation_flat(0) = 0.0f;
        }
        if (offset + sizeof(float) <= size) {
            float trans_y;
            std::memcpy(&trans_y, data + offset, sizeof(float));
            offset += sizeof(float);
            translation_flat(1) = trans_y;
        } else {
            translation_flat(1) = 0.0f;
        }
        
        std::string kernel_type = "lanczos3";
        bool antialias = true;
        if (offset < size) {
            uint8_t kernel_selector = data[offset++];
            switch (kernel_selector % 4) {
                case 0: kernel_type = "lanczos3"; break;
                case 1: kernel_type = "lanczos5"; break;
                case 2: kernel_type = "gaussian"; break;
                case 3: kernel_type = "box"; break;
            }
        }
        if (offset < size) {
            antialias = (data[offset++] % 2) == 0;
        }
        
        auto images_input = tensorflow::ops::Placeholder(root, images_dtype);
        auto size_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto scale_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto translation_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto scale_and_translate_op = tensorflow::ops::ScaleAndTranslate(
            root, images_input, size_input, scale_input, translation_input,
            tensorflow::ops::ScaleAndTranslate::KernelType(kernel_type)
                .Antialias(antialias));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {images_input, images_tensor},
            {size_input, size_tensor},
            {scale_input, scale_tensor},
            {translation_input, translation_tensor}
        }, {scale_and_translate_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
