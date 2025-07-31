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

#define MAX_RANK 3
#define MIN_RANK 3
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 100

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
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType image_dtype = parseDataType(data[offset++]);
        
        uint8_t image_rank = 3;
        std::vector<int64_t> image_shape = parseShape(data, offset, size, image_rank);
        
        if (image_shape.size() != 3) {
            return 0;
        }
        
        tensorflow::TensorShape image_tensor_shape;
        for (int64_t dim : image_shape) {
            image_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor image_tensor(image_dtype, image_tensor_shape);
        fillTensorWithDataByType(image_tensor, image_dtype, data, offset, size);
        
        std::cout << "Image tensor shape: [" << image_shape[0] << ", " << image_shape[1] << ", " << image_shape[2] << "]" << std::endl;
        std::cout << "Image tensor dtype: " << tensorflow::DataTypeString(image_dtype) << std::endl;
        
        int64_t crop_height = 1;
        int64_t crop_width = 1;
        
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&crop_height, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            crop_height = std::abs(crop_height) % image_shape[0] + 1;
            if (crop_height > image_shape[0]) crop_height = image_shape[0];
        }
        
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&crop_width, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            crop_width = std::abs(crop_width) % image_shape[1] + 1;
            if (crop_width > image_shape[1]) crop_width = image_shape[1];
        }
        
        tensorflow::TensorShape size_tensor_shape({2});
        tensorflow::Tensor size_tensor(tensorflow::DT_INT64, size_tensor_shape);
        auto size_flat = size_tensor.flat<int64_t>();
        size_flat(0) = crop_height;
        size_flat(1) = crop_width;
        
        std::cout << "Size tensor: [" << crop_height << ", " << crop_width << "]" << std::endl;
        
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
        
        std::cout << "Seeds: " << seed << ", " << seed2 << std::endl;
        
        auto image_input = tensorflow::ops::Const(root, image_tensor);
        auto size_input = tensorflow::ops::Const(root, size_tensor);
        
        // Use raw_ops namespace for RandomCrop
        auto random_crop_op = tensorflow::ops::internal::RandomCrop(root.WithOpName("RandomCrop"), 
                                                                   image_input, 
                                                                   size_input,
                                                                   tensorflow::ops::internal::RandomCrop::Seed(seed).Seed2(seed2));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({random_crop_op}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}