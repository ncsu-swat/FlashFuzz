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
    switch (selector % 3) {  
        case 0:
            dtype = tensorflow::DT_STRING;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
            break;
        case 2:
            dtype = tensorflow::DT_UINT8;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t string_length = std::min(static_cast<size_t>(100), total_size - offset);
            if (string_length > 0) {
                std::string str(reinterpret_cast<const char*>(data + offset), string_length);
                flat(i) = tensorflow::tstring(str);
                offset += string_length;
            } else {
                flat(i) = tensorflow::tstring("");
            }
        } else {
            flat(i) = tensorflow::tstring("");
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType contents_dtype = tensorflow::DT_STRING;
        uint8_t contents_rank = 0;
        std::vector<int64_t> contents_shape = {};
        
        tensorflow::TensorShape contents_tensor_shape;
        for (auto dim : contents_shape) {
            contents_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor contents_tensor(contents_dtype, contents_tensor_shape);
        fillTensorWithDataByType(contents_tensor, contents_dtype, data, offset, size);
        
        tensorflow::DataType crop_window_dtype = tensorflow::DT_INT32;
        uint8_t crop_window_rank = 1;
        std::vector<int64_t> crop_window_shape = {4};
        
        tensorflow::TensorShape crop_window_tensor_shape;
        for (auto dim : crop_window_shape) {
            crop_window_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor crop_window_tensor(crop_window_dtype, crop_window_tensor_shape);
        fillTensorWithDataByType(crop_window_tensor, crop_window_dtype, data, offset, size);
        
        auto contents_input = tensorflow::ops::Const(root, contents_tensor);
        auto crop_window_input = tensorflow::ops::Const(root, crop_window_tensor);
        
        int channels = 0;
        int ratio = 1;
        bool fancy_upscaling = true;
        bool try_recover_truncated = false;
        float acceptable_fraction = 1.0f;
        std::string dct_method = "";
        
        if (offset < size) {
            channels = data[offset] % 4;
            offset++;
        }
        if (offset < size) {
            uint8_t ratio_selector = data[offset] % 4;
            switch (ratio_selector) {
                case 0: ratio = 1; break;
                case 1: ratio = 2; break;
                case 2: ratio = 4; break;
                case 3: ratio = 8; break;
            }
            offset++;
        }
        if (offset < size) {
            fancy_upscaling = (data[offset] % 2) == 1;
            offset++;
        }
        if (offset < size) {
            try_recover_truncated = (data[offset] % 2) == 1;
            offset++;
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&acceptable_fraction, data + offset, sizeof(float));
            offset += sizeof(float);
            acceptable_fraction = std::max(0.0f, std::min(1.0f, acceptable_fraction));
        }
        if (offset < size) {
            uint8_t dct_selector = data[offset] % 3;
            switch (dct_selector) {
                case 0: dct_method = ""; break;
                case 1: dct_method = "INTEGER_FAST"; break;
                case 2: dct_method = "INTEGER_ACCURATE"; break;
            }
            offset++;
        }
        
        auto decode_and_crop_jpeg = tensorflow::ops::DecodeAndCropJpeg(
            root, contents_input, crop_window_input,
            tensorflow::ops::DecodeAndCropJpeg::Channels(channels)
                .Ratio(ratio)
                .FancyUpscaling(fancy_upscaling)
                .TryRecoverTruncated(try_recover_truncated)
                .AcceptableFraction(acceptable_fraction)
                .DctMethod(dct_method)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({decode_and_crop_jpeg}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
