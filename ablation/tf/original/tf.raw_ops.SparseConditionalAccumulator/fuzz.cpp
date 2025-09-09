#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 20) {
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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
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

std::string parseReductionType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return "MEAN";
        case 1:
            return "SUM";
        default:
            return "MEAN";
    }
}

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size, size_t max_len = 32) {
    if (offset >= total_size) {
        return "";
    }
    
    size_t len = std::min(max_len, total_size - offset);
    if (len > 0) {
        uint8_t str_len = data[offset] % static_cast<uint8_t>(len);
        offset++;
        
        if (offset + str_len <= total_size) {
            std::string result(reinterpret_cast<const char*>(data + offset), str_len);
            offset += str_len;
            return result;
        }
    }
    return "";
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        std::string container = parseString(data, offset, size, 16);
        std::string shared_name = parseString(data, offset, size, 16);
        std::string reduction_type = parseReductionType(data[offset % size]);
        offset++;

        std::cout << "dtype: " << dtype << std::endl;
        std::cout << "rank: " << static_cast<int>(rank) << std::endl;
        std::cout << "shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "container: " << container << std::endl;
        std::cout << "shared_name: " << shared_name << std::endl;
        std::cout << "reduction_type: " << reduction_type << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }

        auto sparse_conditional_accumulator = tensorflow::ops::SparseConditionalAccumulator(
            root,
            dtype,
            tensor_shape,
            tensorflow::ops::SparseConditionalAccumulator::Container(container)
                .SharedName(shared_name)
                .ReductionType(reduction_type)
        );

        std::cout << "SparseConditionalAccumulator operation created successfully" << std::endl;

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
        status = session->Run({}, {sparse_conditional_accumulator.handle.name()}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
        } else {
            std::cout << "Session run successful, output tensor count: " << outputs.size() << std::endl;
            if (!outputs.empty()) {
                std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
                std::cout << "Output tensor dtype: " << outputs[0].dtype() << std::endl;
            }
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}