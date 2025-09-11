#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseImageSizeDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_UINT8;
            break;
        case 1:
            dtype = tensorflow::DT_INT8;
            break;
        case 2:
            dtype = tensorflow::DT_INT16;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_INT64;
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
        tensorflow::DataType image_size_dtype = parseImageSizeDataType(data[offset++]);
        
        tensorflow::TensorShape image_size_shape({3});
        tensorflow::Tensor image_size_tensor(image_size_dtype, image_size_shape);
        fillTensorWithDataByType(image_size_tensor, image_size_dtype, data, offset, size);
        
        auto image_size_flat = image_size_tensor.flat<int32_t>();
        if (image_size_dtype == tensorflow::DT_INT32) {
            for (int i = 0; i < 3; ++i) {
                int32_t val = image_size_flat(i);
                if (val <= 0) val = 1;
                if (val > 1000) val = 1000;
                image_size_flat(i) = val;
            }
        }
        
        tensorflow::TensorShape bounding_boxes_shape({1, 1, 4});
        tensorflow::Tensor bounding_boxes_tensor(tensorflow::DT_FLOAT, bounding_boxes_shape);
        fillTensorWithData<float>(bounding_boxes_tensor, data, offset, size);
        
        auto bbox_flat = bounding_boxes_tensor.flat<float>();
        for (int i = 0; i < 4; ++i) {
            float val = bbox_flat(i);
            if (std::isnan(val) || std::isinf(val)) val = 0.0f;
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            bbox_flat(i) = val;
        }
        
        int seed = 0;
        int seed2 = 0;
        float min_object_covered = 0.1f;
        std::vector<float> aspect_ratio_range = {0.75f, 1.33f};
        std::vector<float> area_range = {0.05f, 1.0f};
        int max_attempts = 100;
        bool use_image_if_no_bounding_boxes = false;
        
        if (offset < size) {
            seed = static_cast<int>(data[offset++]);
        }
        if (offset < size) {
            seed2 = static_cast<int>(data[offset++]);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_object_covered, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(min_object_covered) || std::isinf(min_object_covered) || min_object_covered < 0.0f) {
                min_object_covered = 0.1f;
            }
        }
        if (offset < size) {
            use_image_if_no_bounding_boxes = (data[offset++] % 2) == 1;
        }

        auto image_size_input = tensorflow::ops::Const(root, image_size_tensor);
        auto bounding_boxes_input = tensorflow::ops::Const(root, bounding_boxes_tensor);

        auto sample_op = tensorflow::ops::SampleDistortedBoundingBox(
            root, 
            image_size_input, 
            bounding_boxes_input,
            tensorflow::ops::SampleDistortedBoundingBox::Attrs()
                .Seed(seed)
                .Seed2(seed2)
                .MinObjectCovered(min_object_covered)
                .AspectRatioRange(aspect_ratio_range)
                .AreaRange(area_range)
                .MaxAttempts(max_attempts)
                .UseImageIfNoBoundingBoxes(use_image_if_no_bounding_boxes)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sample_op.begin, sample_op.size, sample_op.bboxes}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
