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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract shape dimensions
        int32_t shape_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        shape_dim = std::abs(shape_dim) % 4 + 1; // Limit to reasonable size
        
        // Extract rate value
        double rate = *reinterpret_cast<const double*>(data + offset);
        offset += sizeof(double);
        
        // Extract seed values
        int32_t seed = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        int32_t seed2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Clamp rate to reasonable range to avoid extreme values
        rate = std::max(0.1, std::min(rate, 100.0));
        if (!std::isfinite(rate)) rate = 1.0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create shape tensor
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
        shape_tensor.flat<int32_t>()(0) = shape_dim;
        
        // Create rate tensor
        tensorflow::Tensor rate_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({}));
        rate_tensor.scalar<double>()() = rate;
        
        // Create seed tensor
        tensorflow::Tensor seed_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        seed_tensor.flat<int32_t>()(0) = seed;
        seed_tensor.flat<int32_t>()(1) = seed2;
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto shape_node = tensorflow::ops::Const(shape_tensor, builder.opts().WithName("shape"));
        auto rate_node = tensorflow::ops::Const(rate_tensor, builder.opts().WithName("rate"));
        auto seed_node = tensorflow::ops::Const(seed_tensor, builder.opts().WithName("seed"));
        
        auto random_poisson = tensorflow::ops::UnaryOp("RandomPoissonV2", 
            tensorflow::ops::NodeOut(rate_node, 0), 
            builder.opts()
                .WithName("random_poisson")
                .WithAttr("shape", shape_tensor)
                .WithAttr("seed", seed)
                .WithAttr("seed2", seed2)
                .WithAttr("dtype", tensorflow::DT_DOUBLE));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"shape:0", shape_tensor}, 
                              {"rate:0", rate_tensor}, 
                              {"seed:0", seed_tensor}}, 
                             {"random_poisson:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_DOUBLE && 
                output.NumElements() > 0 && 
                output.NumElements() <= 10000) {
                // Basic validation of output values
                auto output_flat = output.flat<double>();
                for (int i = 0; i < std::min(static_cast<int>(output.NumElements()), 100); ++i) {
                    double val = output_flat(i);
                    if (!std::isfinite(val) || val < 0) {
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