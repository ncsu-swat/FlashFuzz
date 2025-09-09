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
        int32_t ref_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t ref_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 50 + 1;
        offset += 4;
        int32_t updates_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        
        if (ref_dim0 <= 0 || ref_dim1 <= 0 || indices_size <= 0 || updates_dim1 <= 0) return 0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create ref tensor (variable to be scattered)
        tensorflow::NodeDef* ref_node = graph_def.add_node();
        ref_node->set_name("ref");
        ref_node->set_op("Placeholder");
        (*ref_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        tensorflow::TensorShapeProto* ref_shape = (*ref_node->mutable_attr())["shape"].mutable_shape();
        ref_shape->add_dim()->set_size(ref_dim0);
        ref_shape->add_dim()->set_size(ref_dim1);
        
        // Create indices tensor
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        (*indices_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        tensorflow::TensorShapeProto* indices_shape = (*indices_node->mutable_attr())["shape"].mutable_shape();
        indices_shape->add_dim()->set_size(indices_size);
        
        // Create updates tensor
        tensorflow::NodeDef* updates_node = graph_def.add_node();
        updates_node->set_name("updates");
        updates_node->set_op("Placeholder");
        (*updates_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        tensorflow::TensorShapeProto* updates_shape = (*updates_node->mutable_attr())["shape"].mutable_shape();
        updates_shape->add_dim()->set_size(indices_size);
        updates_shape->add_dim()->set_size(updates_dim1);
        
        // Create ScatterMul node
        tensorflow::NodeDef* scatter_node = graph_def.add_node();
        scatter_node->set_name("scatter_mul");
        scatter_node->set_op("ScatterMul");
        scatter_node->add_input("ref");
        scatter_node->add_input("indices");
        scatter_node->add_input("updates");
        (*scatter_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*scatter_node->mutable_attr())["Tindices"].set_type(tensorflow::DT_INT32);
        (*scatter_node->mutable_attr())["use_locking"].set_b(false);
        
        // Create the session with the graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Create input tensors
        tensorflow::Tensor ref_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ref_dim0, ref_dim1}));
        auto ref_flat = ref_tensor.flat<float>();
        for (int i = 0; i < ref_flat.size() && offset < size; ++i) {
            ref_flat(i) = (offset < size) ? static_cast<float>(data[offset++] % 100) / 10.0f + 1.0f : 1.0f;
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset < size; ++i) {
            indices_flat(i) = (offset < size) ? static_cast<int32_t>(data[offset++]) % ref_dim0 : 0;
        }
        
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({indices_size, updates_dim1}));
        auto updates_flat = updates_tensor.flat<float>();
        for (int i = 0; i < updates_flat.size() && offset < size; ++i) {
            updates_flat(i) = (offset < size) ? static_cast<float>(data[offset++] % 100) / 10.0f + 0.1f : 0.1f;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"ref", ref_tensor},
            {"indices", indices_tensor},
            {"updates", updates_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"scatter_mul"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}