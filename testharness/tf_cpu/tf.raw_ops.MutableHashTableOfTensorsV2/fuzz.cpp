#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/lookup_ops.h"
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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "";
    }
    
    uint8_t str_len = data[offset] % 32;
    offset++;
    
    std::string result;
    for (uint8_t i = 0; i < str_len && offset < total_size; ++i) {
        result += static_cast<char>(data[offset]);
        offset++;
    }
    
    return result;
}

tensorflow::TensorShape parseValueShape(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return tensorflow::TensorShape({});
    }
    
    uint8_t rank = parseRank(data[offset]);
    offset++;
    
    std::vector<int64_t> shape = parseShape(data, offset, total_size, rank);
    return tensorflow::TensorShape(shape);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {

        tensorflow::DataType key_dtype = parseDataType(data[offset]);
        offset++;
        
        tensorflow::DataType value_dtype = parseDataType(data[offset]);
        offset++;
        
        std::string container = parseString(data, offset, size);
        std::string shared_name = parseString(data, offset, size);
        
        bool use_node_name_sharing = false;
        if (offset < size) {
            use_node_name_sharing = (data[offset] % 2) == 1;
            offset++;
        }
        
        tensorflow::TensorShape value_shape = parseValueShape(data, offset, size);
        
        std::cout << "key_dtype: " << tensorflow::DataTypeString(key_dtype) << std::endl;
        std::cout << "value_dtype: " << tensorflow::DataTypeString(value_dtype) << std::endl;
        std::cout << "container: " << container << std::endl;
        std::cout << "shared_name: " << shared_name << std::endl;
        std::cout << "use_node_name_sharing: " << use_node_name_sharing << std::endl;
        std::cout << "value_shape: " << value_shape.DebugString() << std::endl;

        auto hash_table = tensorflow::ops::MutableHashTableOfTensors(
            root,
            key_dtype,
            value_dtype,
            tensorflow::ops::MutableHashTableOfTensors::Container(container)
                .SharedName(shared_name)
                .UseNodeNameSharing(use_node_name_sharing)
                .ValueShape(value_shape)
        );

        std::cout << "Created hash table operation" << std::endl;

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({hash_table}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::cout << "Session ran successfully, outputs size: " << outputs.size() << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
