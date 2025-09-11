#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#include <cstring>
#include <vector>
#include <iostream>

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
        default:
            dtype = tensorflow::DT_FLOAT;
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
                    flat(i) = "";
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_dataset_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        auto input_dataset = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        
        int32_t num_parallel_calls_val = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_parallel_calls_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_parallel_calls_val = std::abs(num_parallel_calls_val) % 8 + 1;
        }
        
        tensorflow::Tensor num_parallel_calls_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_parallel_calls_tensor.scalar<int32_t>()() = num_parallel_calls_val;
        auto num_parallel_calls = tensorflow::ops::Const(root, num_parallel_calls_tensor);
        
        std::vector<tensorflow::DataType> output_types = {input_dtype};
        std::vector<tensorflow::PartialTensorShape> output_shapes = {tensorflow::PartialTensorShape(input_shape)};
        
        bool use_inter_op_parallelism = (data[offset % size] % 2) == 0;
        bool sloppy = (data[(offset + 1) % size] % 2) == 0;
        bool preserve_cardinality = (data[(offset + 2) % size] % 2) == 0;
        std::string metadata = "";
        
        tensorflow::NameAttrList f_attr;
        f_attr.set_name("identity_func");
        
        auto parallel_map_dataset_op = tensorflow::ops::experimental::ParallelMapDataset(
            root,
            input_dataset,
            {},
            num_parallel_calls,
            f_attr,
            output_types,
            output_shapes,
            use_inter_op_parallelism,
            sloppy,
            preserve_cardinality
        );

        tensorflow::ClientSession session(root);
        
        std::cout << "Input dataset dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Num parallel calls: " << num_parallel_calls_val << std::endl;
        std::cout << "Use inter op parallelism: " << use_inter_op_parallelism << std::endl;
        std::cout << "Sloppy: " << sloppy << std::endl;
        std::cout << "Preserve cardinality: " << preserve_cardinality << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
