#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/framework/scope.h>
#include <vector>
#include <memory>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        uint8_t num_dtypes = (data[offset++] % 5) + 1;
        std::vector<tensorflow::DataType> dtypes;
        
        for (uint8_t i = 0; i < num_dtypes && offset < size; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            dtypes.push_back(dtype);
            std::cout << "DataType " << i << ": " << tensorflow::DataTypeString(dtype) << std::endl;
        }

        if (offset >= size) return 0;

        int64_t capacity = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&capacity, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            capacity = std::abs(capacity) % 1000;
        }
        std::cout << "Capacity: " << capacity << std::endl;

        int64_t memory_limit = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&memory_limit, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            memory_limit = std::abs(memory_limit) % 10000;
        }
        std::cout << "Memory limit: " << memory_limit << std::endl;

        std::string container = "";
        if (offset < size) {
            uint8_t container_len = data[offset++] % 10;
            for (uint8_t i = 0; i < container_len && offset < size; ++i) {
                container += static_cast<char>(data[offset++] % 128);
            }
        }
        std::cout << "Container: " << container << std::endl;

        std::string shared_name = "";
        if (offset < size) {
            uint8_t shared_name_len = data[offset++] % 10;
            for (uint8_t i = 0; i < shared_name_len && offset < size; ++i) {
                shared_name += static_cast<char>(data[offset++] % 128);
            }
        }
        std::cout << "Shared name: " << shared_name << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        tensorflow::ops::MapClear::Attrs attrs;
        attrs.capacity_ = capacity;
        attrs.memory_limit_ = memory_limit;
        attrs.container_ = container;
        attrs.shared_name_ = shared_name;

        auto map_clear_op = tensorflow::ops::MapClear(root.WithOpName("map_clear"), dtypes, attrs);

        std::cout << "MapClear operation created successfully" << std::endl;

        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to convert scope to GraphDef: " << status.ToString() << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {}, {"map_clear"}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
        } else {
            std::cout << "MapClear operation executed successfully" << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}