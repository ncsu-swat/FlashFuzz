#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for input tensor
        int32_t batch_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        int32_t height_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        int32_t width_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        int32_t channel_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        
        // Clamp dimensions to reasonable values
        batch_dim = std::max(1, std::min(batch_dim, 64));
        height_dim = std::max(1, std::min(height_dim, 32));
        width_dim = std::max(1, std::min(width_dim, 32));
        channel_dim = std::max(1, std::min(channel_dim, 16));
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch_dim, height_dim, width_dim, channel_dim}));
        
        // Fill input tensor with fuzz data
        auto input_flat = input_tensor.flat<float>();
        size_t tensor_size = input_flat.size();
        size_t remaining_data = size - offset;
        
        for (int i = 0; i < tensor_size && offset < size; ++i) {
            if (offset + 4 <= size) {
                input_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            } else {
                input_flat(i) = 0.0f;
            }
        }
        
        // Create block_shape tensor (2D for height and width)
        tensorflow::Tensor block_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto block_shape_flat = block_shape_tensor.flat<int32_t>();
        
        // Extract block shape values or use defaults
        int32_t block_height = 2, block_width = 2;
        if (offset + 8 <= size) {
            block_height = std::max(1, std::min(*reinterpret_cast<const int32_t*>(data + offset), 8));
            offset += 4;
            block_width = std::max(1, std::min(*reinterpret_cast<const int32_t*>(data + offset), 8));
            offset += 4;
        }
        
        block_shape_flat(0) = block_height;
        block_shape_flat(1) = block_width;
        
        // Create crops tensor (2x2 for height and width crops)
        tensorflow::Tensor crops_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
        auto crops_flat = crops_tensor.flat<int32_t>();
        
        // Extract crop values or use defaults (no cropping)
        int32_t crop_top = 0, crop_bottom = 0, crop_left = 0, crop_right = 0;
        if (offset + 16 <= size) {
            crop_top = std::max(0, std::min(*reinterpret_cast<const int32_t*>(data + offset), height_dim));
            offset += 4;
            crop_bottom = std::max(0, std::min(*reinterpret_cast<const int32_t*>(data + offset), height_dim));
            offset += 4;
            crop_left = std::max(0, std::min(*reinterpret_cast<const int32_t*>(data + offset), width_dim));
            offset += 4;
            crop_right = std::max(0, std::min(*reinterpret_cast<const int32_t*>(data + offset), width_dim));
            offset += 4;
        }
        
        crops_flat(0) = crop_top;    // crops[0][0]
        crops_flat(1) = crop_bottom; // crops[0][1]
        crops_flat(2) = crop_left;   // crops[1][0]
        crops_flat(3) = crop_right;  // crops[1][1]
        
        // Create a simple session for testing
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        
        // Add input placeholders
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* block_shape_node = graph_def.add_node();
        block_shape_node->set_name("block_shape");
        block_shape_node->set_op("Placeholder");
        (*block_shape_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* crops_node = graph_def.add_node();
        crops_node->set_name("crops");
        crops_node->set_op("Placeholder");
        (*crops_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        // Add BatchToSpaceND operation
        tensorflow::NodeDef* batch_to_space_node = graph_def.add_node();
        batch_to_space_node->set_name("batch_to_space");
        batch_to_space_node->set_op("BatchToSpaceND");
        batch_to_space_node->add_input("input");
        batch_to_space_node->add_input("block_shape");
        batch_to_space_node->add_input("crops");
        (*batch_to_space_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*batch_to_space_node->mutable_attr())["Tblock_shape"].set_type(tensorflow::DT_INT32);
        (*batch_to_space_node->mutable_attr())["Tcrops"].set_type(tensorflow::DT_INT32);
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor},
            {"block_shape", block_shape_tensor},
            {"crops", crops_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"batch_to_space"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}