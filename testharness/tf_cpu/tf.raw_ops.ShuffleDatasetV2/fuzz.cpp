#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 21) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
        case 17:
            dtype = tensorflow::DT_UINT32;
            break;
        case 18:
            dtype = tensorflow::DT_UINT64;
            break;
        case 19:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 20:
            dtype = tensorflow::DT_STRING;
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
        case tensorflow::DT_STRING:
            {
                auto flat = tensor.flat<tensorflow::tstring>();
                for (int i = 0; i < flat.size(); ++i) {
                    if (offset < total_size) {
                        uint8_t str_len = data[offset] % 10 + 1;
                        offset++;
                        std::string str;
                        for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                            str += static_cast<char>(data[offset]);
                            offset++;
                        }
                        flat(i) = str;
                    } else {
                        flat(i) = "";
                    }
                }
            }
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
        tensorflow::DataType output_dtype = parseDataType(data[offset++]);
        uint8_t output_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape = parseShape(data, offset, size, output_rank);
        
        tensorflow::TensorShape dataset_shape(output_shape);
        
        // Create a range dataset
        auto start = tensorflow::ops::Const(root, 0);
        auto stop = tensorflow::ops::Const(root, 10);
        auto step = tensorflow::ops::Const(root, 1);
        
        // Create a tensor to hold output types
        tensorflow::Tensor output_types_tensor(tensorflow::DT_INT32, {1});
        output_types_tensor.flat<int32_t>()(0) = tensorflow::DT_INT64;
        auto output_types = tensorflow::ops::Const(root, output_types_tensor);
        
        // Create a tensor to hold output shapes
        tensorflow::Tensor output_shapes_tensor(tensorflow::DT_INT64, {1, 0});
        auto output_shapes = tensorflow::ops::Const(root, output_shapes_tensor);
        
        // Create RangeDataset using raw ops
        auto range_dataset = tensorflow::ops::_RawOps::RangeDataset(
            root, start, stop, step, output_types, output_shapes);

        int64_t buffer_size_val = 5;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&buffer_size_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            buffer_size_val = std::abs(buffer_size_val) % 100 + 1;
        }
        auto buffer_size = tensorflow::ops::Const(root, buffer_size_val);

        // Create seed tensor
        tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, {2});
        seed_tensor.flat<int64_t>()(0) = 1;
        seed_tensor.flat<int64_t>()(1) = 2;
        auto seed = tensorflow::ops::Const(root, seed_tensor);
        
        // Create seed2 tensor
        tensorflow::Tensor seed2_tensor(tensorflow::DT_INT64, {2});
        seed2_tensor.flat<int64_t>()(0) = 3;
        seed2_tensor.flat<int64_t>()(1) = 4;
        auto seed2 = tensorflow::ops::Const(root, seed2_tensor);
        
        // Create seed generator
        auto seed_generator = tensorflow::ops::_RawOps::AnonymousSeedGenerator(
            root, seed, seed2);

        // Create output types tensor for ShuffleDatasetV2
        tensorflow::Tensor shuffle_output_types_tensor(tensorflow::DT_INT32, {1});
        shuffle_output_types_tensor.flat<int32_t>()(0) = output_dtype;
        auto shuffle_output_types = tensorflow::ops::Const(root, shuffle_output_types_tensor);
        
        // Create output shapes tensor for ShuffleDatasetV2
        tensorflow::Tensor shuffle_output_shapes_tensor(tensorflow::DT_INT64, {1, static_cast<int64_t>(output_shape.size())});
        auto shuffle_output_shapes_flat = shuffle_output_shapes_tensor.flat<int64_t>();
        for (size_t i = 0; i < output_shape.size(); ++i) {
            shuffle_output_shapes_flat(i) = output_shape[i];
        }
        auto shuffle_output_shapes = tensorflow::ops::Const(root, shuffle_output_shapes_tensor);

        // Create ShuffleDatasetV2 using raw ops
        auto shuffle_dataset = tensorflow::ops::_RawOps::ShuffleDatasetV2(
            root, range_dataset, buffer_size, seed_generator, 
            shuffle_output_types, shuffle_output_shapes);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::cout << "Buffer size: " << buffer_size_val << std::endl;
        std::cout << "Output dtype: " << output_dtype << std::endl;
        std::cout << "Output rank: " << static_cast<int>(output_rank) << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
