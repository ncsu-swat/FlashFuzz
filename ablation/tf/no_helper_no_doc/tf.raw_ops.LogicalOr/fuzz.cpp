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
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_testutil.h>
#include <tensorflow/core/lib/core/status_test_util.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract dimensions for tensors
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable values
        dim1 = (dim1 % 100) + 1;
        dim2 = (dim2 % 100) + 1;
        
        size_t total_elements = dim1 * dim2;
        size_t required_bytes = total_elements * 2; // 2 tensors
        
        if (offset + required_bytes > size) return 0;
        
        // Create tensor shapes
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create input tensors
        tensorflow::Tensor x_tensor(tensorflow::DT_BOOL, shape);
        tensorflow::Tensor y_tensor(tensorflow::DT_BOOL, shape);
        
        auto x_flat = x_tensor.flat<bool>();
        auto y_flat = y_tensor.flat<bool>();
        
        // Fill tensors with fuzz data
        for (size_t i = 0; i < total_elements && offset < size; ++i) {
            x_flat(i) = (data[offset] & 1) != 0;
            offset++;
        }
        
        for (size_t i = 0; i < total_elements && offset < size; ++i) {
            y_flat(i) = (data[offset] & 1) != 0;
            offset++;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Add placeholder nodes
        tensorflow::NodeDef* x_node = graph_def.add_node();
        x_node->set_name("x");
        x_node->set_op("Placeholder");
        (*x_node->mutable_attr())["dtype"].set_type(tensorflow::DT_BOOL);
        
        tensorflow::NodeDef* y_node = graph_def.add_node();
        y_node->set_name("y");
        y_node->set_op("Placeholder");
        (*y_node->mutable_attr())["dtype"].set_type(tensorflow::DT_BOOL);
        
        // Add LogicalOr node
        tensorflow::NodeDef* logical_or_node = graph_def.add_node();
        logical_or_node->set_name("logical_or");
        logical_or_node->set_op("LogicalOr");
        logical_or_node->add_input("x");
        logical_or_node->add_input("y");
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"x", x_tensor}, {"y", y_tensor}}, {"logical_or"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape matches input shapes
            if (outputs[0].shape() == shape && outputs[0].dtype() == tensorflow::DT_BOOL) {
                auto result_flat = outputs[0].flat<bool>();
                // Basic sanity check - verify logical OR operation
                for (int64_t i = 0; i < total_elements; ++i) {
                    bool expected = x_flat(i) || y_flat(i);
                    if (result_flat(i) != expected) {
                        std::cout << "LogicalOr result mismatch at index " << i << std::endl;
                        break;
                    }
                }
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}