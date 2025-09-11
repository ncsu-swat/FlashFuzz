#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t num_dtypes_byte = data[offset++];
        uint8_t num_dtypes = (num_dtypes_byte % 5) + 1;
        
        std::vector<tensorflow::DataType> dtypes;
        for (uint8_t i = 0; i < num_dtypes; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            dtypes.push_back(dtype);
        }
        
        if (dtypes.empty()) {
            dtypes.push_back(tensorflow::DT_FLOAT);
        }

        if (offset >= size) return 0;
        int64_t capacity = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&capacity, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            capacity = std::abs(capacity) % 1000;
        }

        if (offset >= size) return 0;
        int64_t memory_limit = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&memory_limit, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            memory_limit = std::abs(memory_limit) % 1000000;
        }

        std::string container = "";
        if (offset < size) {
            uint8_t container_len = data[offset++] % 10;
            for (uint8_t i = 0; i < container_len && offset < size; ++i) {
                container += static_cast<char>(data[offset++] % 128);
            }
        }

        std::string shared_name = "";
        if (offset < size) {
            uint8_t shared_name_len = data[offset++] % 10;
            for (uint8_t i = 0; i < shared_name_len && offset < size; ++i) {
                shared_name += static_cast<char>(data[offset++] % 128);
            }
        }

        auto ordered_map_size_op = tensorflow::ops::OrderedMapSize(
            root,
            dtypes,
            tensorflow::ops::OrderedMapSize::Attrs()
                .Capacity(capacity)
                .MemoryLimit(memory_limit)
                .Container(container)
                .SharedName(shared_name)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({ordered_map_size_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
