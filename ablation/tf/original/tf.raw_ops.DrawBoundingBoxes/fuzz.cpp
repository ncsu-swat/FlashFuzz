#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType images_dtype = parseDataType(data[offset++]);
        
        uint8_t images_rank = 4;
        std::vector<int64_t> images_shape = {1, 2, 2, 1};
        
        if (offset + 4 * sizeof(int64_t) <= size) {
            int64_t batch, height, width, depth;
            std::memcpy(&batch, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&height, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&width, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&depth, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            batch = std::abs(batch) % 3 + 1;
            height = std::abs(height) % 10 + 1;
            width = std::abs(width) % 10 + 1;
            depth = std::abs(depth) % 4 + 1;
            
            images_shape = {batch, height, width, depth};
        }

        tensorflow::TensorShape images_tensor_shape(images_shape);
        tensorflow::Tensor images_tensor(images_dtype, images_tensor_shape);
        fillTensorWithDataByType(images_tensor, images_dtype, data, offset, size);

        std::vector<int64_t> boxes_shape = {images_shape[0], 2, 4};
        tensorflow::TensorShape boxes_tensor_shape(boxes_shape);
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, boxes_tensor_shape);
        fillTensorWithData<float>(boxes_tensor, data, offset, size);

        auto boxes_flat = boxes_tensor.flat<float>();
        for (int i = 0; i < boxes_flat.size(); ++i) {
            float val = boxes_flat(i);
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            boxes_flat(i) = val;
        }

        std::cout << "Images tensor shape: ";
        for (int i = 0; i < images_tensor.shape().dims(); ++i) {
            std::cout << images_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Boxes tensor shape: ";
        for (int i = 0; i < boxes_tensor.shape().dims(); ++i) {
            std::cout << boxes_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto images_placeholder = tensorflow::ops::Placeholder(root, images_dtype);
        auto boxes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto draw_bounding_boxes = tensorflow::ops::DrawBoundingBoxes(root, images_placeholder, boxes_placeholder);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{images_placeholder, images_tensor}, {boxes_placeholder, boxes_tensor}}, 
                                               {draw_bounding_boxes}, &outputs);

        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}