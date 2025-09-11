#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_component_types = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::DataType> component_types;
        for (uint8_t i = 0; i < num_component_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            component_types.push_back(dtype);
        }
        
        if (component_types.empty()) {
            component_types.push_back(tensorflow::DT_FLOAT);
        }

        std::vector<tensorflow::PartialTensorShape> shapes;
        if (offset < size) {
            uint8_t use_shapes = data[offset++] % 2;
            if (use_shapes) {
                for (size_t i = 0; i < component_types.size() && offset < size; ++i) {
                    uint8_t rank = parseRank(data[offset++]);
                    std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
                    tensorflow::PartialTensorShape shape(shape_dims);
                    shapes.push_back(shape);
                }
            }
        }

        int32_t capacity = -1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&capacity, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            capacity = capacity % 1000;
            if (capacity < -1) capacity = -1;
        }

        std::string container = "";
        if (offset < size) {
            uint8_t container_len = data[offset++] % 10;
            if (offset + container_len <= size) {
                container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
                offset += container_len;
            }
        }

        std::string shared_name = "";
        if (offset < size) {
            uint8_t shared_name_len = data[offset++] % 10;
            if (offset + shared_name_len <= size) {
                shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
                offset += shared_name_len;
            }
        }

        std::cout << "Creating FIFOQueue with:" << std::endl;
        std::cout << "  component_types size: " << component_types.size() << std::endl;
        std::cout << "  shapes size: " << shapes.size() << std::endl;
        std::cout << "  capacity: " << capacity << std::endl;
        std::cout << "  container: " << container << std::endl;
        std::cout << "  shared_name: " << shared_name << std::endl;

        auto fifo_queue_attrs = tensorflow::ops::FIFOQueue::Attrs()
            .Capacity(capacity)
            .Container(container)
            .SharedName(shared_name);

        if (!shapes.empty()) {
            fifo_queue_attrs = fifo_queue_attrs.Shapes(shapes);
        }

        auto fifo_queue = tensorflow::ops::FIFOQueue(
            root,
            component_types,
            fifo_queue_attrs
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({fifo_queue}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (!outputs.empty()) {
            std::cout << "FIFOQueue output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
