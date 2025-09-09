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
        uint32_t ref_dim0 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t ref_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t indices_size = *reinterpret_cast<const uint32_t*>(data + offset) % std::min(ref_dim0, 5u) + 1;
        offset += 4;
        uint32_t updates_dim1 = ref_dim1;
        offset += 4;
        
        if (offset + indices_size * 4 + indices_size * updates_dim1 * 4 + ref_dim0 * ref_dim1 * 4 > size) {
            return 0;
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create ref tensor (variable)
        auto ref_node = graph_def.add_node();
        ref_node->set_name("ref");
        ref_node->set_op("Placeholder");
        (*ref_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*ref_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(ref_dim0);
        (*ref_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(ref_dim1);
        
        // Create indices tensor
        auto indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        (*indices_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        (*indices_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(indices_size);
        
        // Create updates tensor
        auto updates_node = graph_def.add_node();
        updates_node->set_name("updates");
        updates_node->set_op("Placeholder");
        (*updates_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*updates_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(indices_size);
        (*updates_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(updates_dim1);
        
        // Create ScatterDiv node
        auto scatter_node = graph_def.add_node();
        scatter_node->set_name("scatter_div");
        scatter_node->set_op("ScatterDiv");
        scatter_node->add_input("ref");
        scatter_node->add_input("indices");
        scatter_node->add_input("updates");
        (*scatter_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*scatter_node->mutable_attr())["Tindices"].set_type(tensorflow::DT_INT32);
        (*scatter_node->mutable_attr())["use_locking"].set_b(false);
        
        // Create session and add graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare input tensors
        tensorflow::Tensor ref_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(ref_dim0), static_cast<int64_t>(ref_dim1)}));
        auto ref_flat = ref_tensor.flat<float>();
        for (int i = 0; i < ref_dim0 * ref_dim1; i++) {
            if (offset + 4 <= size) {
                float val = *reinterpret_cast<const float*>(data + offset);
                // Avoid division by zero by ensuring non-zero values
                ref_flat(i) = (val == 0.0f) ? 1.0f : val;
                offset += 4;
            } else {
                ref_flat(i) = 1.0f;
            }
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + 4 <= size) {
                int32_t idx = *reinterpret_cast<const int32_t*>(data + offset) % ref_dim0;
                indices_flat(i) = std::abs(idx);
                offset += 4;
            } else {
                indices_flat(i) = 0;
            }
        }
        
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(indices_size), static_cast<int64_t>(updates_dim1)}));
        auto updates_flat = updates_tensor.flat<float>();
        for (int i = 0; i < indices_size * updates_dim1; i++) {
            if (offset + 4 <= size) {
                float val = *reinterpret_cast<const float*>(data + offset);
                // Avoid division by zero
                updates_flat(i) = (val == 0.0f) ? 1.0f : val;
                offset += 4;
            } else {
                updates_flat(i) = 1.0f;
            }
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"ref", ref_tensor}, {"indices", indices_tensor}, {"updates", updates_tensor}}, 
                             {"scatter_div"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed ScatterDiv operation
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}