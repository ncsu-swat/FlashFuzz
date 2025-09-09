#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract shape dimensions
        int32_t shape_dims = (data[offset] % 4) + 1;
        offset++;
        
        if (offset + shape_dims * 4 + 8 > size) return 0;
        
        std::vector<int32_t> shape_data;
        for (int i = 0; i < shape_dims; i++) {
            int32_t dim = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
            shape_data.push_back(std::abs(dim));
            offset += 4;
        }
        
        // Extract rate value
        float rate_val = *reinterpret_cast<const float*>(data + offset);
        if (std::isnan(rate_val) || std::isinf(rate_val) || rate_val < 0) {
            rate_val = 1.0f;
        }
        offset += 4;
        
        // Extract seeds
        int seed = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        
        if (offset + 4 > size) return 0;
        int seed2 = *reinterpret_cast<const int32_t*>(data + offset);
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create shape tensor
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({shape_dims}));
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (int i = 0; i < shape_dims; i++) {
            shape_flat(i) = shape_data[i];
        }
        
        // Create rate tensor
        tensorflow::Tensor rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        rate_tensor.scalar<float>()() = rate_val;
        
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto rate_op = tensorflow::ops::Const(root, rate_tensor);
        
        // Create RandomPoissonV2 operation
        tensorflow::NodeDef node_def;
        tensorflow::NodeDefBuilder builder("random_poisson", "RandomPoissonV2");
        builder.Input(shape_op.node()->name(), 0, tensorflow::DT_INT32);
        builder.Input(rate_op.node()->name(), 0, tensorflow::DT_FLOAT);
        builder.Attr("seed", seed);
        builder.Attr("seed2", seed2);
        builder.Attr("dtype", tensorflow::DT_INT64);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::GraphDef graph_def;
        status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add the RandomPoissonV2 node to graph
        auto* new_node = graph_def.add_node();
        *new_node = node_def;
        
        std::unique_ptr<tensorflow::Session> tf_session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = tf_session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
        inputs.push_back({shape_op.node()->name() + ":0", shape_tensor});
        inputs.push_back({rate_op.node()->name() + ":0", rate_tensor});
        
        status = tf_session->Run(inputs, {"random_poisson:0"}, {}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            // Verify output tensor properties
            if (result.dtype() == tensorflow::DT_INT64) {
                auto result_flat = result.flat<int64_t>();
                for (int i = 0; i < std::min(10, static_cast<int>(result_flat.size())); i++) {
                    if (result_flat(i) < 0) {
                        return 0;
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}