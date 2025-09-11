#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10 + 1;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset] % 128);
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "default";
                }
            }
            break;
        }
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
        tensorflow::Tensor variant_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        if (offset >= size) return 0;
        uint8_t path_len = data[offset] % 20 + 1;
        offset++;
        
        std::string path_str = "/tmp/test_dataset_";
        for (uint8_t i = 0; i < path_len && offset < size; ++i) {
            path_str += static_cast<char>('a' + (data[offset] % 26));
            offset++;
        }
        
        tensorflow::Tensor path_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        path_tensor.scalar<tensorflow::tstring>()() = path_str;
        
        if (offset >= size) return 0;
        uint8_t num_other_args = data[offset] % 3;
        offset++;
        
        std::vector<tensorflow::Input> shard_func_other_args;
        for (uint8_t i = 0; i < num_other_args && offset < size; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto placeholder = tensorflow::ops::Placeholder(root, dtype, 
                tensorflow::ops::Placeholder::Shape(tensor_shape));
            shard_func_other_args.push_back(placeholder);
        }
        
        if (offset >= size) return 0;
        uint8_t num_output_types = (data[offset] % 3) + 1;
        offset++;
        
        std::vector<tensorflow::DataType> output_types;
        for (uint8_t i = 0; i < num_output_types && offset < size; ++i) {
            output_types.push_back(parseDataType(data[offset]));
            offset++;
        }
        
        std::vector<tensorflow::TensorShape> output_shapes;
        for (uint8_t i = 0; i < num_output_types && offset < size; ++i) {
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            output_shapes.push_back(tensor_shape);
        }
        
        if (output_shapes.size() < output_types.size()) {
            while (output_shapes.size() < output_types.size()) {
                output_shapes.push_back(tensorflow::TensorShape({1}));
            }
        }
        
        std::string compression = "";
        if (offset < size) {
            if (data[offset] % 2 == 1) {
                compression = "GZIP";
            }
            offset++;
        }
        
        bool use_shard_func = true;
        if (offset < size) {
            use_shard_func = (data[offset] % 2 == 1);
            offset++;
        }
        
        auto input_dataset = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        auto path_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        tensorflow::NameAttrList shard_func_attr;
        shard_func_attr.set_name("identity_func");
        
        // Using raw_ops directly
        auto save_dataset = tensorflow::ops::internal::SaveDatasetV2(
            root,
            input_dataset,
            path_placeholder,
            shard_func_other_args,
            compression,
            use_shard_func,
            shard_func_attr
        );
        
        std::cout << "SaveDatasetV2 operation created successfully" << std::endl;
        std::cout << "Path: " << path_str << std::endl;
        std::cout << "Compression: " << compression << std::endl;
        std::cout << "UseShardFunc: " << use_shard_func << std::endl;
        std::cout << "Output types count: " << output_types.size() << std::endl;
        std::cout << "Output shapes count: " << output_shapes.size() << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
