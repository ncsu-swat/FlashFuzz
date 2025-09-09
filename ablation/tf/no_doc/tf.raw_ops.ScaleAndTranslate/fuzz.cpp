#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_UINT8;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
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

        tensorflow::DataType image_dtype = parseDataType(data[offset++]);
        uint8_t image_rank = parseRank(data[offset++]);
        if (image_rank < 3 || image_rank > 4) {
            image_rank = 4;
        }
        
        std::vector<int64_t> image_shape = parseShape(data, offset, size, image_rank);
        if (image_shape.size() < 3) {
            return 0;
        }
        
        tensorflow::TensorShape image_tensor_shape;
        for (int64_t dim : image_shape) {
            image_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor image_tensor(image_dtype, image_tensor_shape);
        fillTensorWithDataByType(image_tensor, image_dtype, data, offset, size);
        
        std::cout << "Image tensor shape: ";
        for (int i = 0; i < image_tensor_shape.dims(); ++i) {
            std::cout << image_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        if (offset + 4 * sizeof(float) > size) {
            return 0;
        }
        
        float scale_x, scale_y, translate_x, translate_y;
        std::memcpy(&scale_x, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&scale_y, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&translate_x, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&translate_y, data + offset, sizeof(float));
        offset += sizeof(float);
        
        scale_x = std::max(0.1f, std::min(10.0f, std::abs(scale_x)));
        scale_y = std::max(0.1f, std::min(10.0f, std::abs(scale_y)));
        translate_x = std::max(-100.0f, std::min(100.0f, translate_x));
        translate_y = std::max(-100.0f, std::min(100.0f, translate_y));
        
        tensorflow::TensorShape scale_shape({2});
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, scale_shape);
        auto scale_flat = scale_tensor.flat<float>();
        scale_flat(0) = scale_x;
        scale_flat(1) = scale_y;
        
        tensorflow::TensorShape translation_shape({2});
        tensorflow::Tensor translation_tensor(tensorflow::DT_FLOAT, translation_shape);
        auto translation_flat = translation_tensor.flat<float>();
        translation_flat(0) = translate_x;
        translation_flat(1) = translate_y;
        
        std::string interpolation = "BILINEAR";
        bool antialias = false;
        
        std::cout << "Scale: [" << scale_x << ", " << scale_y << "]" << std::endl;
        std::cout << "Translation: [" << translate_x << ", " << translate_y << "]" << std::endl;
        std::cout << "Interpolation: " << interpolation << std::endl;
        std::cout << "Antialias: " << antialias << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto image_placeholder = tensorflow::ops::Placeholder(root, image_dtype);
        auto scale_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto translation_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto scale_and_translate_op = tensorflow::ops::ScaleAndTranslate(
            root, 
            image_placeholder, 
            image_tensor_shape.dim_size(image_tensor_shape.dims() - 3),
            image_tensor_shape.dim_size(image_tensor_shape.dims() - 2),
            scale_placeholder,
            translation_placeholder,
            tensorflow::ops::ScaleAndTranslate::Interpolation(interpolation).Antialias(antialias)
        );
        
        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to create graph: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {image_placeholder.node()->name(), image_tensor},
            {scale_placeholder.node()->name(), scale_tensor},
            {translation_placeholder.node()->name(), translation_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {scale_and_translate_op.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "ScaleAndTranslate operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
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