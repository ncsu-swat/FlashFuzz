#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/graph/node_builder.h>

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
    try {
        size_t offset = 0;
        
        if (size < 3) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);

        std::cout << "DataType: " << tensorflow::DataTypeString(dtype) << std::endl;
        std::cout << "Rank: " << static_cast<int>(rank) << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }

        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        
        tensorflow::Node* placeholder_node;
        tensorflow::NodeBuilder placeholder_builder("placeholder", "Placeholder");
        placeholder_builder.Attr("dtype", dtype);
        placeholder_builder.Attr("shape", tensor_shape);
        
        tensorflow::Status status = placeholder_builder.Finalize(&graph, &placeholder_node);
        if (!status.ok()) {
            std::cout << "Failed to create Placeholder node: " << status.ToString() << std::endl;
            return 0;
        }

        std::cout << "Placeholder node created successfully" << std::endl;
        std::cout << "Node name: " << placeholder_node->name() << std::endl;
        std::cout << "Node type: " << placeholder_node->type_string() << std::endl;

        tensorflow::GraphDef graph_def;
        graph.ToGraphDef(&graph_def);

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        
        if (dtype == tensorflow::DT_STRING) {
            auto flat = input_tensor.flat<tensorflow::tstring>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = "test_string";
            }
        } else {
            size_t temp_offset = offset;
            switch (dtype) {
                case tensorflow::DT_FLOAT: {
                    auto flat = input_tensor.flat<float>();
                    for (int i = 0; i < flat.size() && temp_offset + sizeof(float) <= size; ++i) {
                        float value;
                        std::memcpy(&value, data + temp_offset, sizeof(float));
                        temp_offset += sizeof(float);
                        flat(i) = value;
                    }
                    break;
                }
                case tensorflow::DT_DOUBLE: {
                    auto flat = input_tensor.flat<double>();
                    for (int i = 0; i < flat.size() && temp_offset + sizeof(double) <= size; ++i) {
                        double value;
                        std::memcpy(&value, data + temp_offset, sizeof(double));
                        temp_offset += sizeof(double);
                        flat(i) = value;
                    }
                    break;
                }
                case tensorflow::DT_INT32: {
                    auto flat = input_tensor.flat<int32_t>();
                    for (int i = 0; i < flat.size() && temp_offset + sizeof(int32_t) <= size; ++i) {
                        int32_t value;
                        std::memcpy(&value, data + temp_offset, sizeof(int32_t));
                        temp_offset += sizeof(int32_t);
                        flat(i) = value;
                    }
                    break;
                }
                case tensorflow::DT_INT64: {
                    auto flat = input_tensor.flat<int64_t>();
                    for (int i = 0; i < flat.size() && temp_offset + sizeof(int64_t) <= size; ++i) {
                        int64_t value;
                        std::memcpy(&value, data + temp_offset, sizeof(int64_t));
                        temp_offset += sizeof(int64_t);
                        flat(i) = value;
                    }
                    break;
                }
                case tensorflow::DT_BOOL: {
                    auto flat = input_tensor.flat<bool>();
                    for (int i = 0; i < flat.size() && temp_offset < size; ++i) {
                        flat(i) = (data[temp_offset++] % 2) == 1;
                    }
                    break;
                }
                default: {
                    auto flat = input_tensor.flat<float>();
                    for (int i = 0; i < flat.size(); ++i) {
                        flat(i) = 0.0f;
                    }
                    break;
                }
            }
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"placeholder:0", input_tensor}
        };
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"placeholder:0"};

        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Session run failed: " << status.ToString() << std::endl;
        } else {
            std::cout << "Session run successful" << std::endl;
            if (!outputs.empty()) {
                std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
                std::cout << "Output tensor dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
            }
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}