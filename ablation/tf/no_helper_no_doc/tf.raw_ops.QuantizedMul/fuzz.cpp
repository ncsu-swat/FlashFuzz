#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for input tensors
        int32_t x_dim = (data[offset] % 4) + 1;
        offset++;
        int32_t y_dim = (data[offset] % 4) + 1;
        offset++;
        
        // Extract quantization parameters
        float min_x = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float max_x = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float min_y = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float max_y = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        if (offset >= size) return 0;
        
        // Ensure valid ranges
        if (min_x >= max_x) {
            max_x = min_x + 1.0f;
        }
        if (min_y >= max_y) {
            max_y = min_y + 1.0f;
        }
        
        // Create input tensors
        tensorflow::TensorShape x_shape({x_dim});
        tensorflow::TensorShape y_shape({y_dim});
        
        tensorflow::Tensor x_tensor(tensorflow::DT_QUINT8, x_shape);
        tensorflow::Tensor y_tensor(tensorflow::DT_QUINT8, y_shape);
        tensorflow::Tensor min_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Fill input tensors with fuzz data
        auto x_flat = x_tensor.flat<tensorflow::quint8>();
        auto y_flat = y_tensor.flat<tensorflow::quint8>();
        
        for (int i = 0; i < x_dim && offset < size; i++) {
            x_flat(i) = tensorflow::quint8(data[offset++]);
        }
        
        for (int i = 0; i < y_dim && offset < size; i++) {
            y_flat(i) = tensorflow::quint8(data[offset++]);
        }
        
        // Set quantization parameters
        min_x_tensor.scalar<float>()() = min_x;
        max_x_tensor.scalar<float>()() = max_x;
        min_y_tensor.scalar<float>()() = min_y;
        max_y_tensor.scalar<float>()() = max_y;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("quantized_mul");
        node_def->set_op("QuantizedMul");
        node_def->add_input("x:0");
        node_def->add_input("y:0");
        node_def->add_input("min_x:0");
        node_def->add_input("max_x:0");
        node_def->add_input("min_y:0");
        node_def->add_input("max_y:0");
        
        // Set attributes
        tensorflow::AttrValue toutput_attr;
        toutput_attr.set_type(tensorflow::DT_QINT32);
        (*node_def->mutable_attr())["Toutput"] = toutput_attr;
        
        tensorflow::AttrValue t1_attr;
        t1_attr.set_type(tensorflow::DT_QUINT8);
        (*node_def->mutable_attr())["T1"] = t1_attr;
        
        tensorflow::AttrValue t2_attr;
        t2_attr.set_type(tensorflow::DT_QUINT8);
        (*node_def->mutable_attr())["T2"] = t2_attr;
        
        // Add input nodes
        auto add_input_node = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* input_node = graph_def.add_node();
            input_node->set_name(name);
            input_node->set_op("Placeholder");
            tensorflow::AttrValue dtype_attr;
            dtype_attr.set_type(dtype);
            (*input_node->mutable_attr())["dtype"] = dtype_attr;
            tensorflow::AttrValue shape_attr;
            shape.AsProto(shape_attr.mutable_shape());
            (*input_node->mutable_attr())["shape"] = shape_attr;
        };
        
        add_input_node("x", tensorflow::DT_QUINT8, x_shape);
        add_input_node("y", tensorflow::DT_QUINT8, y_shape);
        add_input_node("min_x", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_input_node("max_x", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_input_node("min_y", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        add_input_node("max_y", tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Create session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x:0", x_tensor},
            {"y:0", y_tensor},
            {"min_x:0", min_x_tensor},
            {"max_x:0", max_x_tensor},
            {"min_y:0", min_y_tensor},
            {"max_y:0", max_y_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"quantized_mul:0", "quantized_mul:1", "quantized_mul:2"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}