#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_FIELD_NAMES 5
#define MAX_VALUES_COUNT 5

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 11) {  
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
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
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
            uint8_t str_len = data[offset] % 20 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("default");
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_field_names = (data[offset] % MAX_FIELD_NAMES) + 1;
        offset++;

        std::vector<std::string> field_names;
        for (uint8_t i = 0; i < num_field_names; ++i) {
            if (offset >= size) break;
            uint8_t name_len = (data[offset] % 10) + 1;
            offset++;
            
            std::string field_name = "field_";
            for (uint8_t j = 0; j < name_len && offset < size; ++j) {
                field_name += static_cast<char>('a' + (data[offset] % 26));
                offset++;
            }
            field_names.push_back(field_name);
        }

        if (field_names.empty()) {
            field_names.push_back("default_field");
        }

        if (offset >= size) return 0;
        uint8_t sizes_rank = parseRank(data[offset]);
        offset++;

        std::vector<int64_t> sizes_shape = parseShape(data, offset, size, sizes_rank);
        if (sizes_shape.empty()) {
            sizes_shape = {static_cast<int64_t>(field_names.size())};
        } else {
            sizes_shape.back() = static_cast<int64_t>(field_names.size());
        }

        tensorflow::Tensor sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(sizes_shape));
        fillTensorWithDataByType(sizes_tensor, tensorflow::DT_INT32, data, offset, size);

        std::cout << "Sizes tensor shape: ";
        for (int i = 0; i < sizes_tensor.shape().dims(); ++i) {
            std::cout << sizes_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        if (offset >= size) return 0;
        uint8_t num_values = (data[offset] % std::min(static_cast<size_t>(MAX_VALUES_COUNT), field_names.size())) + 1;
        offset++;

        std::vector<tensorflow::Output> values;
        for (uint8_t i = 0; i < num_values && i < field_names.size(); ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType value_dtype = parseDataType(data[offset]);
            offset++;
            
            if (offset >= size) break;
            uint8_t value_rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> value_shape = parseShape(data, offset, size, value_rank);
            if (value_shape.empty()) {
                value_shape = {1};
            }

            tensorflow::Tensor value_tensor(value_dtype, tensorflow::TensorShape(value_shape));
            fillTensorWithDataByType(value_tensor, value_dtype, data, offset, size);

            std::cout << "Value tensor " << i << " shape: ";
            for (int j = 0; j < value_tensor.shape().dims(); ++j) {
                std::cout << value_tensor.shape().dim_size(j) << " ";
            }
            std::cout << " dtype: " << tensorflow::DataTypeString(value_dtype) << std::endl;

            auto placeholder = tensorflow::ops::Placeholder(root, value_dtype);
            values.push_back(placeholder);
        }

        if (values.empty()) {
            tensorflow::Tensor default_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({1}));
            auto flat = default_tensor.flat<tensorflow::tstring>();
            flat(0) = tensorflow::tstring("default_value");
            auto placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
            values.push_back(placeholder);
        }

        std::string message_type = "TestMessage";
        std::string descriptor_source = "local://";

        std::cout << "Field names: ";
        for (const auto& name : field_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        std::cout << "Message type: " << message_type << std::endl;
        std::cout << "Descriptor source: " << descriptor_source << std::endl;

        auto sizes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        // Use raw_ops.EncodeProto instead of ops::EncodeProto
        auto encode_proto_op = tensorflow::ops::internal::EncodeProto(
            root.WithOpName("EncodeProto"),
            sizes_placeholder,
            values,
            field_names,
            message_type,
            descriptor_source
        );

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        feed_dict.push_back({sizes_placeholder.node()->name(), sizes_tensor});
        
        for (size_t i = 0; i < values.size() && i < num_values; ++i) {
            tensorflow::DataType value_dtype = parseDataType(data[1 + field_names.size() * 11 + 1 + i * 10]);
            std::vector<int64_t> value_shape = {1};
            tensorflow::Tensor value_tensor(value_dtype, tensorflow::TensorShape(value_shape));
            fillTensorWithDataByType(value_tensor, value_dtype, data, offset, size);
            feed_dict.push_back({values[i].node()->name(), value_tensor});
        }

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, {encode_proto_op}, {}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
