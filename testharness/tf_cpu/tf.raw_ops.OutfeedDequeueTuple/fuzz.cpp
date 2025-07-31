#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_OUTPUTS 5

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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 16:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 17:
            dtype = tensorflow::DT_HALF;
            break;
        case 18:
            dtype = tensorflow::DT_UINT32;
            break;
        case 19:
            dtype = tensorflow::DT_UINT64;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t num_outputs_byte = data[offset++];
        int num_outputs = (num_outputs_byte % MAX_NUM_OUTPUTS) + 1;

        std::vector<tensorflow::DataType> dtypes;
        std::vector<tensorflow::TensorShape> shapes;

        for (int i = 0; i < num_outputs; ++i) {
            if (offset >= size) return 0;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            dtypes.push_back(dtype);

            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape shape;
            for (int64_t dim : shape_dims) {
                shape.AddDim(dim);
            }
            shapes.push_back(shape);
        }

        if (offset >= size) return 0;
        int32_t device_ordinal_raw;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&device_ordinal_raw, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        } else {
            device_ordinal_raw = -1;
        }
        int32_t device_ordinal = (device_ordinal_raw % 3) - 1;

        std::cout << "Creating OutfeedDequeueTuple with " << num_outputs << " outputs" << std::endl;
        std::cout << "Device ordinal: " << device_ordinal << std::endl;
        
        for (int i = 0; i < num_outputs; ++i) {
            std::cout << "Output " << i << ": dtype=" << dtypes[i] << ", shape=[";
            for (int j = 0; j < shapes[i].dims(); ++j) {
                std::cout << shapes[i].dim_size(j);
                if (j < shapes[i].dims() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }

        // Use raw_ops API instead of ops namespace
        auto outfeed_op = tensorflow::ops::OutfeedDequeueTuple(
            root,
            dtypes,
            shapes,
            tensorflow::ops::OutfeedDequeueTuple::DeviceOrdinal(device_ordinal)
        );

        tensorflow::ClientSession session(root);

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}