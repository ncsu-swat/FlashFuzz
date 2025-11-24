#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/node_builder.h"
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t num_types_byte = data[offset++];
        size_t num_types = (num_types_byte % 5) + 1;

        std::vector<tensorflow::DataType> output_types;
        for (size_t i = 0; i < num_types; ++i) {
            if (offset >= size) return 0;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            output_types.push_back(dtype);
        }

        std::vector<tensorflow::PartialTensorShape> output_shapes;
        for (size_t i = 0; i < num_types; ++i) {
            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::PartialTensorShape partial_shape;
            if (rank == 0) {
                partial_shape = tensorflow::PartialTensorShape({});
            } else {
                partial_shape = tensorflow::PartialTensorShape(shape);
            }
            output_shapes.push_back(partial_shape);
        }

        std::cout << "Creating AnonymousIteratorV2 with " << num_types << " output types" << std::endl;
        for (size_t i = 0; i < output_types.size(); ++i) {
            std::cout << "Type " << i << ": " << tensorflow::DataTypeString(output_types[i]) << std::endl;
        }

        // Convert output_types to a tensor
        tensorflow::Tensor output_types_tensor(tensorflow::DT_INT32, {static_cast<int64_t>(output_types.size())});
        auto output_types_flat = output_types_tensor.flat<int32_t>();
        for (size_t i = 0; i < output_types.size(); ++i) {
            output_types_flat(i) = static_cast<int32_t>(output_types[i]);
        }

        // Convert output_shapes to a tensor
        std::vector<tensorflow::Tensor> output_shapes_tensors;
        for (const auto& shape : output_shapes) {
            std::vector<int64_t> dims;
            for (int i = 0; i < shape.dims(); ++i) {
                dims.push_back(shape.dim_size(i));
            }
            tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, {static_cast<int64_t>(dims.size())});
            auto shape_flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < dims.size(); ++i) {
                shape_flat(i) = dims[i];
            }
            output_shapes_tensors.push_back(shape_tensor);
        }

        // Create a tensor for output_shapes
        tensorflow::Tensor output_shapes_tensor(tensorflow::DT_INT64, 
            {static_cast<int64_t>(output_shapes.size()), 
             static_cast<int64_t>(output_shapes.empty() ? 0 : output_shapes[0].dims())});
        auto output_shapes_flat = output_shapes_tensor.matrix<int64_t>();
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            for (int j = 0; j < output_shapes[i].dims(); ++j) {
                output_shapes_flat(i, j) = output_shapes[i].dim_size(j);
            }
        }

        tensorflow::Node* iterator_node = nullptr;
        auto iterator_builder = tensorflow::NodeBuilder(
                                    root.GetUniqueNameForOp("AnonymousIteratorV2"),
                                    "AnonymousIteratorV2")
                                    .Attr("output_types", output_types)
                                    .Attr("output_shapes", output_shapes);
        root.UpdateStatus(iterator_builder.Finalize(root.graph(), &iterator_node));
        if (!root.ok() || iterator_node == nullptr) {
            return -1;
        }

        tensorflow::Output iterator_handle(iterator_node, 0);
        tensorflow::Output iterator_deleter(iterator_node, 1);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({}, {iterator_handle, iterator_deleter}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::cout << "AnonymousIteratorV2 executed successfully" << std::endl;
        std::cout << "Handle tensor shape: " << outputs[0].shape().DebugString() << std::endl;
        std::cout << "Deleter tensor shape: " << outputs[1].shape().DebugString() << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
