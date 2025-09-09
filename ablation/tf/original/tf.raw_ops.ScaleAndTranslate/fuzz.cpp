#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType images_dtype = parseDataType(data[offset++]);
        uint8_t images_rank = parseRank(data[offset++]);
        
        if (images_rank < 3 || images_rank > 4) {
            images_rank = 4;
        }
        
        std::vector<int64_t> images_shape = parseShape(data, offset, size, images_rank);
        
        tensorflow::TensorShape images_tensor_shape;
        for (int64_t dim : images_shape) {
            images_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor images_tensor(images_dtype, images_tensor_shape);
        fillTensorWithDataByType(images_tensor, images_dtype, data, offset, size);
        
        std::cout << "Images tensor shape: ";
        for (int i = 0; i < images_tensor_shape.dims(); ++i) {
            std::cout << images_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::TensorShape size_shape({2});
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, size_shape);
        fillTensorWithData<int32_t>(size_tensor, data, offset, size);
        
        std::cout << "Size tensor: ";
        auto size_flat = size_tensor.flat<int32_t>();
        for (int i = 0; i < size_flat.size(); ++i) {
            std::cout << size_flat(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::TensorShape scale_shape({2});
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, scale_shape);
        fillTensorWithData<float>(scale_tensor, data, offset, size);
        
        std::cout << "Scale tensor: ";
        auto scale_flat = scale_tensor.flat<float>();
        for (int i = 0; i < scale_flat.size(); ++i) {
            std::cout << scale_flat(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::TensorShape translation_shape({2});
        tensorflow::Tensor translation_tensor(tensorflow::DT_FLOAT, translation_shape);
        fillTensorWithData<float>(translation_tensor, data, offset, size);
        
        std::cout << "Translation tensor: ";
        auto translation_flat = translation_tensor.flat<float>();
        for (int i = 0; i < translation_flat.size(); ++i) {
            std::cout << translation_flat(i) << " ";
        }
        std::cout << std::endl;
        
        std::string kernel_type = "lanczos3";
        bool antialias = true;
        
        if (offset < size) {
            uint8_t kernel_selector = data[offset++];
            switch (kernel_selector % 4) {
                case 0:
                    kernel_type = "lanczos3";
                    break;
                case 1:
                    kernel_type = "lanczos5";
                    break;
                case 2:
                    kernel_type = "gaussian";
                    break;
                case 3:
                    kernel_type = "box";
                    break;
            }
        }
        
        if (offset < size) {
            antialias = (data[offset++] % 2) == 1;
        }
        
        std::cout << "Kernel type: " << kernel_type << ", Antialias: " << antialias << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto images_placeholder = tensorflow::ops::Placeholder(root, images_dtype);
        auto size_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto scale_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto translation_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto scale_and_translate_op = tensorflow::ops::ScaleAndTranslate(
            root, images_placeholder, size_placeholder, scale_placeholder, translation_placeholder,
            tensorflow::ops::ScaleAndTranslate::KernelType(kernel_type).Antialias(antialias)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{images_placeholder, images_tensor},
             {size_placeholder, size_tensor},
             {scale_placeholder, scale_tensor},
             {translation_placeholder, translation_tensor}},
            {scale_and_translate_op}, &outputs
        );
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "ScaleAndTranslate operation succeeded. Output shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "ScaleAndTranslate operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}