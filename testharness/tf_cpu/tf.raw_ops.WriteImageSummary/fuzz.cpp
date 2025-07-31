#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& msg, const uint8_t* data, size_t size) {
        std::cerr << msg << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_UINT8;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
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
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto writer_tensor = tensorflow::Tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        auto writer = tensorflow::ops::Const(root, writer_tensor);

        int64_t step_value = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&step_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        auto step = tensorflow::ops::Const(root, step_value);

        std::string tag_value = "test_tag";
        if (offset + 8 <= size) {
            tag_value = std::string(reinterpret_cast<const char*>(data + offset), 8);
            offset += 8;
        }
        auto tag = tensorflow::ops::Const(root, tag_value);

        if (offset >= size) return 0;
        
        tensorflow::DataType tensor_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t tensor_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> tensor_shape = parseShape(data, offset, size, tensor_rank);
        
        tensorflow::TensorShape tf_tensor_shape;
        for (auto dim : tensor_shape) {
            tf_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensor_dtype, tf_tensor_shape);
        fillTensorWithDataByType(input_tensor, tensor_dtype, data, offset, size);
        auto tensor = tensorflow::ops::Const(root, input_tensor);

        tensorflow::Tensor bad_color_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({3}));
        auto bad_color_flat = bad_color_tensor.flat<uint8_t>();
        for (int i = 0; i < 3; ++i) {
            if (offset < size) {
                bad_color_flat(i) = data[offset++];
            } else {
                bad_color_flat(i) = 255;
            }
        }
        auto bad_color = tensorflow::ops::Const(root, bad_color_tensor);

        int max_images = 3;
        if (offset < size) {
            max_images = (data[offset++] % 10) + 1;
        }

        // Use raw_ops for WriteImageSummary
        std::vector<tensorflow::Output> inputs = {writer, step, tag, tensor, bad_color};
        tensorflow::NodeBuilder node_builder = tensorflow::NodeBuilder("write_image_summary", "WriteImageSummary")
            .Input(tensorflow::NodeBuilder::NodeOut(writer.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(step.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(tag.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(tensor.node()))
            .Input(tensorflow::NodeBuilder::NodeOut(bad_color.node()))
            .Attr("max_images", max_images);

        tensorflow::Node* node;
        root.UpdateStatus(root.graph()->AddNode(node_builder, &node));
        
        tensorflow::ClientSession session(root);
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}