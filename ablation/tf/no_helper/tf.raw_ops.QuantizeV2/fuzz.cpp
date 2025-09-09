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
#include <tensorflow/core/graph/graph_def_builder.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        uint32_t input_size = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        input_size = input_size % 1000 + 1; // Limit size to reasonable range
        
        float min_range_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float max_range_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        // Ensure valid range
        if (min_range_val >= max_range_val) {
            max_range_val = min_range_val + 1.0f;
        }
        
        uint8_t mode_idx = data[offset++] % 3;
        uint8_t round_mode_idx = data[offset++] % 2;
        uint8_t dtype_idx = data[offset++] % 5;
        bool narrow_range = (data[offset++] % 2) == 1;
        int32_t axis = static_cast<int32_t>(data[offset++]) - 128; // -128 to 127
        float ensure_min_range = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        if (offset >= size) return 0;
        
        // Map indices to actual values
        std::string mode;
        switch (mode_idx) {
            case 0: mode = "MIN_COMBINED"; break;
            case 1: mode = "MIN_FIRST"; break;
            case 2: mode = "SCALED"; break;
        }
        
        std::string round_mode;
        switch (round_mode_idx) {
            case 0: round_mode = "HALF_AWAY_FROM_ZERO"; break;
            case 1: round_mode = "HALF_TO_EVEN"; break;
        }
        
        tensorflow::DataType output_dtype;
        switch (dtype_idx) {
            case 0: output_dtype = tensorflow::DT_QINT8; break;
            case 1: output_dtype = tensorflow::DT_QUINT8; break;
            case 2: output_dtype = tensorflow::DT_QINT32; break;
            case 3: output_dtype = tensorflow::DT_QINT16; break;
            case 4: output_dtype = tensorflow::DT_QUINT16; break;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(input_size)}));
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzzer data
        size_t remaining_data = size - offset;
        size_t float_count = std::min(remaining_data / sizeof(float), static_cast<size_t>(input_size));
        
        for (size_t i = 0; i < float_count && offset + sizeof(float) <= size; ++i) {
            input_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        // Fill remaining with zeros if needed
        for (size_t i = float_count; i < input_size; ++i) {
            input_flat(i) = 0.0f;
        }
        
        // Create min_range and max_range tensors
        tensorflow::Tensor min_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_range_tensor.scalar<float>()() = min_range_val;
        
        tensorflow::Tensor max_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_range_tensor.scalar<float>()() = max_range_val;
        
        // Create session and graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto input_node = tensorflow::ops::Const(input_tensor, builder.opts().WithName("input"));
        auto min_range_node = tensorflow::ops::Const(min_range_tensor, builder.opts().WithName("min_range"));
        auto max_range_node = tensorflow::ops::Const(max_range_tensor, builder.opts().WithName("max_range"));
        
        tensorflow::NodeDefBuilder quantize_builder("quantize", "QuantizeV2");
        quantize_builder.Input(input_node.name(), 0, tensorflow::DT_FLOAT)
                       .Input(min_range_node.name(), 0, tensorflow::DT_FLOAT)
                       .Input(max_range_node.name(), 0, tensorflow::DT_FLOAT)
                       .Attr("T", output_dtype)
                       .Attr("mode", mode)
                       .Attr("round_mode", round_mode)
                       .Attr("narrow_range", narrow_range)
                       .Attr("axis", axis)
                       .Attr("ensure_minimum_range", ensure_min_range);
        
        tensorflow::NodeDef quantize_def;
        auto status = quantize_builder.Finalize(&quantize_def);
        if (!status.ok()) {
            return 0;
        }
        
        builder.opts().FinalizeGraph();
        tensorflow::GraphDef graph_def;
        status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add the QuantizeV2 node to the graph
        *graph_def.add_node() = quantize_def;
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"quantize:0", "quantize:1", "quantize:2"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
        if (status.ok() && outputs.size() == 3) {
            // Successfully executed QuantizeV2
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}