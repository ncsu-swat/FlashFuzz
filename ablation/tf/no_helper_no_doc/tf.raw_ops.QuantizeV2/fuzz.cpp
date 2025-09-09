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
        
        // Extract dimensions for input tensor
        int32_t batch_size = (data[offset] % 4) + 1;
        offset++;
        int32_t height = (data[offset] % 8) + 1;
        offset++;
        int32_t width = (data[offset] % 8) + 1;
        offset++;
        int32_t channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract quantization parameters
        float min_range = -10.0f + (data[offset] % 100) * 0.2f;
        offset++;
        float max_range = min_range + 0.1f + (data[offset] % 100) * 0.2f;
        offset++;
        
        // Extract mode
        std::string mode;
        uint8_t mode_selector = data[offset] % 4;
        offset++;
        switch (mode_selector) {
            case 0: mode = "MIN_COMBINED"; break;
            case 1: mode = "MIN_FIRST"; break;
            case 2: mode = "SCALED"; break;
            default: mode = "MIN_COMBINED"; break;
        }
        
        // Extract round_mode
        std::string round_mode;
        uint8_t round_mode_selector = data[offset] % 3;
        offset++;
        switch (round_mode_selector) {
            case 0: round_mode = "HALF_AWAY_FROM_ZERO"; break;
            case 1: round_mode = "HALF_TO_EVEN"; break;
            default: round_mode = "HALF_AWAY_FROM_ZERO"; break;
        }
        
        // Extract narrow_range
        bool narrow_range = (data[offset] % 2) == 1;
        offset++;
        
        // Extract axis (optional)
        int32_t axis = -1;
        if (offset < size) {
            axis = (int32_t)(data[offset] % 4) - 1; // -1, 0, 1, 2
            offset++;
        }
        
        // Create input tensor with remaining data
        tensorflow::TensorShape input_shape({batch_size, height, width, channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with data
        int64_t num_elements = input_tensor.NumElements();
        for (int64_t i = 0; i < num_elements && offset < size; ++i) {
            float val = (float)(data[offset % size]) / 255.0f * (max_range - min_range) + min_range;
            input_flat(i) = val;
            offset++;
        }
        
        // Create min_range and max_range tensors
        tensorflow::Tensor min_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_range_tensor.scalar<float>()() = min_range;
        
        tensorflow::Tensor max_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_range_tensor.scalar<float>()() = max_range;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("quantize_v2");
        node_def->set_op("QuantizeV2");
        node_def->add_input("input:0");
        node_def->add_input("min_range:0");
        node_def->add_input("max_range:0");
        
        // Set attributes
        tensorflow::AttrValue attr_T;
        attr_T.set_type(tensorflow::DT_QUINT8);
        (*node_def->mutable_attr())["T"] = attr_T;
        
        tensorflow::AttrValue attr_mode;
        attr_mode.set_s(mode);
        (*node_def->mutable_attr())["mode"] = attr_mode;
        
        tensorflow::AttrValue attr_round_mode;
        attr_round_mode.set_s(round_mode);
        (*node_def->mutable_attr())["round_mode"] = attr_round_mode;
        
        tensorflow::AttrValue attr_narrow_range;
        attr_narrow_range.set_b(narrow_range);
        (*node_def->mutable_attr())["narrow_range"] = attr_narrow_range;
        
        if (axis >= 0) {
            tensorflow::AttrValue attr_axis;
            attr_axis.set_i(axis);
            (*node_def->mutable_attr())["axis"] = attr_axis;
        }
        
        // Add input nodes
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue input_dtype;
        input_dtype.set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["dtype"] = input_dtype;
        
        tensorflow::NodeDef* min_node = graph_def.add_node();
        min_node->set_name("min_range");
        min_node->set_op("Placeholder");
        tensorflow::AttrValue min_dtype;
        min_dtype.set_type(tensorflow::DT_FLOAT);
        (*min_node->mutable_attr())["dtype"] = min_dtype;
        
        tensorflow::NodeDef* max_node = graph_def.add_node();
        max_node->set_name("max_range");
        max_node->set_op("Placeholder");
        tensorflow::AttrValue max_dtype;
        max_dtype.set_type(tensorflow::DT_FLOAT);
        (*max_node->mutable_attr())["dtype"] = max_dtype;
        
        // Create session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input:0", input_tensor},
            {"min_range:0", min_range_tensor},
            {"max_range:0", max_range_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"quantize_v2:0", "quantize_v2:1", "quantize_v2:2"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}