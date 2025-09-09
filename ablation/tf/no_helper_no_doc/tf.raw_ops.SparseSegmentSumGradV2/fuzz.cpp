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
        
        if (offset + grad_rows * grad_cols * sizeof(float) + 
            indices_size * sizeof(int32_t) + 
            segment_ids_size * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create input tensors
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({grad_rows, grad_cols}));
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < grad_rows * grad_cols && offset + sizeof(float) <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset + sizeof(int32_t) <= size; ++i) {
            indices_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % grad_rows;
            offset += sizeof(int32_t);
        }
        
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                            tensorflow::TensorShape({segment_ids_size}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < segment_ids_size && offset + sizeof(int32_t) <= size; ++i) {
            segment_ids_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % num_segments;
            offset += sizeof(int32_t);
        }
        
        tensorflow::Tensor output_dim0_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        output_dim0_tensor.scalar<int32_t>()() = grad_rows;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("sparse_segment_sum_grad_v2");
        node_def->set_op("SparseSegmentSumGradV2");
        
        // Add input nodes
        tensorflow::NodeDef* grad_node = graph_def.add_node();
        grad_node->set_name("grad");
        grad_node->set_op("Placeholder");
        (*grad_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        (*indices_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* segment_ids_node = graph_def.add_node();
        segment_ids_node->set_name("segment_ids");
        segment_ids_node->set_op("Placeholder");
        (*segment_ids_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* output_dim0_node = graph_def.add_node();
        output_dim0_node->set_name("output_dim0");
        output_dim0_node->set_op("Placeholder");
        (*output_dim0_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        node_def->add_input("grad");
        node_def->add_input("indices");
        node_def->add_input("segment_ids");
        node_def->add_input("output_dim0");
        (*node_def->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["Tidx"].set_type(tensorflow::DT_INT32);
        (*node_def->mutable_attr())["Tsegmentids"].set_type(tensorflow::DT_INT32);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"grad", grad_tensor},
            {"indices", indices_tensor},
            {"segment_ids", segment_ids_tensor},
            {"output_dim0", output_dim0_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_segment_sum_grad_v2"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed the operation
            auto output_shape = outputs[0].shape();
            if (output_shape.dims() >= 1 && output_shape.dim_size(0) > 0) {
                // Access some output values to ensure computation happened
                auto output_flat = outputs[0].flat<float>();
                volatile float sum = 0.0f;
                for (int i = 0; i < std::min(10, static_cast<int>(output_flat.size())); ++i) {
                    sum += output_flat(i);
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