#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += 4;
        int32_t dim2 = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += 4;
        
        // Extract quantization ranges
        float min_x = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float max_x = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float min_y = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float max_y = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        
        // Ensure valid ranges
        if (min_x >= max_x) {
            min_x = -1.0f;
            max_x = 1.0f;
        }
        if (min_y >= max_y) {
            min_y = -1.0f;
            max_y = 1.0f;
        }
        
        // Select data types
        tensorflow::DataType x_type = tensorflow::DT_QINT8;
        tensorflow::DataType y_type = tensorflow::DT_QINT8;
        tensorflow::DataType output_type = tensorflow::DT_QINT32;
        
        if (offset < size) {
            uint8_t type_selector = data[offset++];
            switch (type_selector % 5) {
                case 0: x_type = tensorflow::DT_QINT8; break;
                case 1: x_type = tensorflow::DT_QUINT8; break;
                case 2: x_type = tensorflow::DT_QINT32; break;
                case 3: x_type = tensorflow::DT_QINT16; break;
                case 4: x_type = tensorflow::DT_QUINT16; break;
            }
        }
        
        if (offset < size) {
            uint8_t type_selector = data[offset++];
            switch (type_selector % 5) {
                case 0: y_type = tensorflow::DT_QINT8; break;
                case 1: y_type = tensorflow::DT_QUINT8; break;
                case 2: y_type = tensorflow::DT_QINT32; break;
                case 3: y_type = tensorflow::DT_QINT16; break;
                case 4: y_type = tensorflow::DT_QUINT16; break;
            }
        }
        
        // Create tensors
        tensorflow::TensorShape shape({dim1, dim2});
        tensorflow::Tensor x_tensor(x_type, shape);
        tensorflow::Tensor y_tensor(y_type, shape);
        
        // Fill tensors with fuzz data
        size_t tensor_size = dim1 * dim2;
        size_t remaining_data = size - offset;
        
        if (x_type == tensorflow::DT_QINT8) {
            auto x_flat = x_tensor.flat<tensorflow::qint8>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                x_flat(i) = tensorflow::qint8(static_cast<int8_t>(data[offset++]));
            }
        } else if (x_type == tensorflow::DT_QUINT8) {
            auto x_flat = x_tensor.flat<tensorflow::quint8>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                x_flat(i) = tensorflow::quint8(data[offset++]);
            }
        }
        
        if (y_type == tensorflow::DT_QINT8) {
            auto y_flat = y_tensor.flat<tensorflow::qint8>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                y_flat(i) = tensorflow::qint8(static_cast<int8_t>(data[offset++]));
            }
        } else if (y_type == tensorflow::DT_QUINT8) {
            auto y_flat = y_tensor.flat<tensorflow::quint8>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                y_flat(i) = tensorflow::quint8(data[offset++]);
            }
        }
        
        // Create scalar tensors for min/max values
        tensorflow::Tensor min_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        min_x_tensor.scalar<float>()() = min_x;
        max_x_tensor.scalar<float>()() = max_x;
        min_y_tensor.scalar<float>()() = min_y;
        max_y_tensor.scalar<float>()() = max_y;
        
        // Create a simple session to test the operation
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root, x_type);
        auto y_placeholder = tensorflow::ops::Placeholder(root, y_type);
        auto min_x_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_x_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto min_y_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_y_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create the QuantizedMul operation
        tensorflow::Node* quantized_mul_node;
        tensorflow::NodeBuilder builder("quantized_mul", "QuantizedMul");
        builder.Input(x_placeholder.node())
               .Input(y_placeholder.node())
               .Input(min_x_placeholder.node())
               .Input(max_x_placeholder.node())
               .Input(min_y_placeholder.node())
               .Input(max_y_placeholder.node())
               .Attr("T1", x_type)
               .Attr("T2", y_type)
               .Attr("Toutput", output_type);
        
        tensorflow::Status status = builder.Finalize(root.graph(), &quantized_mul_node);
        if (!status.ok()) {
            return 0;
        }
        
        tensorflow::GraphDef graph_def;
        status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {x_placeholder.node()->name(), x_tensor},
            {y_placeholder.node()->name(), y_tensor},
            {min_x_placeholder.node()->name(), min_x_tensor},
            {max_x_placeholder.node()->name(), max_x_tensor},
            {min_y_placeholder.node()->name(), min_y_tensor},
            {max_y_placeholder.node()->name(), max_y_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            quantized_mul_node->name() + ":0",
            quantized_mul_node->name() + ":1", 
            quantized_mul_node->name() + ":2"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (status.ok() && outputs.size() == 3) {
            // Successfully executed QuantizedMul
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}