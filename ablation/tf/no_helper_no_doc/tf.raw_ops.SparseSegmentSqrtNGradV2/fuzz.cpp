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
        
        // Extract dimensions and parameters from fuzz input
        int32_t grad_rows = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t grad_cols = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 50 + 1;
        offset += 4;
        int32_t segment_ids_size = indices_size;
        int32_t num_segments = *reinterpret_cast<const int32_t*>(data + offset) % 20 + 1;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create input tensors
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({grad_rows, grad_cols}));
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < grad_rows * grad_cols && offset + 4 <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
            if (offset >= size) break;
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; ++i) {
            indices_flat(i) = i % grad_rows; // Ensure valid indices
        }
        
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({segment_ids_size}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < segment_ids_size; ++i) {
            segment_ids_flat(i) = i % num_segments; // Ensure valid segment IDs
        }
        
        tensorflow::Tensor output_dim0_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        output_dim0_tensor.scalar<int32_t>()() = grad_rows;
        
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_segments_tensor.scalar<int32_t>()() = num_segments;
        
        // Build the node
        auto node_builder = tensorflow::NodeDefBuilder("sparse_segment_sqrt_n_grad_v2", "SparseSegmentSqrtNGradV2")
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32));
        
        tensorflow::NodeDef node_def;
        auto status = node_builder.Finalize(&node_def);
        if (!status.ok()) return 0;
        
        *graph_def.mutable_node()->Add() = node_def;
        
        // Create the session and run
        status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"sparse_segment_sqrt_n_grad_v2:0", grad_tensor},
            {"sparse_segment_sqrt_n_grad_v2:1", indices_tensor},
            {"sparse_segment_sqrt_n_grad_v2:2", segment_ids_tensor},
            {"sparse_segment_sqrt_n_grad_v2:3", output_dim0_tensor},
            {"sparse_segment_sqrt_n_grad_v2:4", num_segments_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"sparse_segment_sqrt_n_grad_v2:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}