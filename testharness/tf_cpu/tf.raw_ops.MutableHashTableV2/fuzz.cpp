#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/lookup_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "";
    }
    
    uint8_t length = data[offset] % 32;
    offset++;
    
    std::string result;
    for (uint8_t i = 0; i < length && offset < total_size; ++i) {
        char c = static_cast<char>(data[offset] % 128);
        if (c == 0) c = 'a';
        result += c;
        offset++;
    }
    
    return result;
}

bool parseBool(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return false;
    }
    bool result = (data[offset] % 2) == 1;
    offset++;
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;
    
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");
    
    try {

        tensorflow::DataType key_dtype = parseDataType(data[offset++]);
        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        
        std::string container = parseString(data, offset, size);
        std::string shared_name = parseString(data, offset, size);
        bool use_node_name_sharing = parseBool(data, offset, size);
        
        std::cout << "key_dtype: " << tensorflow::DataTypeString(key_dtype) << std::endl;
        std::cout << "value_dtype: " << tensorflow::DataTypeString(value_dtype) << std::endl;
        std::cout << "container: " << container << std::endl;
        std::cout << "shared_name: " << shared_name << std::endl;
        std::cout << "use_node_name_sharing: " << use_node_name_sharing << std::endl;
        
        auto mutable_hash_table = tensorflow::ops::MutableHashTable(
            root,
            key_dtype,
            value_dtype,
            tensorflow::ops::MutableHashTable::Attrs()
                .Container(container)
                .SharedName(shared_name)
                .UseNodeNameSharing(use_node_name_sharing)
        );
        
        std::cout << "MutableHashTable operation created successfully" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({mutable_hash_table.table_handle}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        std::cout << "Session run successfully, output tensor shape: ";
        if (!outputs.empty()) {
            std::cout << outputs[0].shape().DebugString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }
    
    return 0;
}
