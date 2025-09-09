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
        
        // Extract seed values
        int32_t seed = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        int32_t seed2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Extract shape dimensions
        if (offset + sizeof(int32_t) > size) return 0;
        int32_t shape_dims = (*reinterpret_cast<const int32_t*>(data + offset)) % 4 + 1;
        offset += sizeof(int32_t);
        
        // Extract alpha dimensions
        if (offset + sizeof(int32_t) > size) return 0;
        int32_t alpha_dims = (*reinterpret_cast<const int32_t*>(data + offset)) % 4 + 1;
        offset += sizeof(int32_t);
        
        // Create shape tensor
        tensorflow::TensorShape shape_tensor_shape({shape_dims});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int32_t>();
        
        for (int i = 0; i < shape_dims && offset + sizeof(int32_t) <= size; ++i) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            shape_flat(i) = std::abs(dim_size % 100) + 1; // Keep dimensions reasonable
            offset += sizeof(int32_t);
        }
        
        // Fill remaining dimensions if not enough data
        for (int i = offset / sizeof(int32_t) - 4; i < shape_dims; ++i) {
            shape_flat(i) = 1;
        }
        
        // Create alpha tensor
        tensorflow::TensorShape alpha_shape;
        std::vector<int64_t> alpha_dims_vec;
        for (int i = 0; i < alpha_dims && offset + sizeof(int32_t) <= size; ++i) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            alpha_dims_vec.push_back(std::abs(dim_size % 10) + 1);
            offset += sizeof(int32_t);
        }
        
        // Fill remaining alpha dimensions if not enough data
        for (int i = alpha_dims_vec.size(); i < alpha_dims; ++i) {
            alpha_dims_vec.push_back(1);
        }
        
        alpha_shape = tensorflow::TensorShape(alpha_dims_vec);
        tensorflow::Tensor alpha_tensor(tensorflow::DT_FLOAT, alpha_shape);
        auto alpha_flat = alpha_tensor.flat<float>();
        
        // Fill alpha values
        for (int i = 0; i < alpha_flat.size() && offset + sizeof(float) <= size; ++i) {
            float alpha_val = *reinterpret_cast<const float*>(data + offset);
            // Ensure positive alpha values for gamma distribution
            alpha_flat(i) = std::abs(alpha_val) + 0.1f;
            offset += sizeof(float);
        }
        
        // Fill remaining alpha values if not enough data
        for (int i = offset / sizeof(float) - (offset - alpha_dims_vec.size() * sizeof(int32_t)) / sizeof(float); 
             i < alpha_flat.size(); ++i) {
            alpha_flat(i) = 1.0f;
        }
        
        // Create a simple session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* shape_node = graph_def.add_node();
        shape_node->set_name("shape");
        shape_node->set_op("Const");
        tensorflow::AttrValue shape_attr;
        shape_tensor.AsProtoTensorContent(shape_attr.mutable_tensor());
        (*shape_node->mutable_attr())["value"] = shape_attr;
        tensorflow::AttrValue shape_dtype_attr;
        shape_dtype_attr.set_type(tensorflow::DT_INT32);
        (*shape_node->mutable_attr())["dtype"] = shape_dtype_attr;
        
        tensorflow::NodeDef* alpha_node = graph_def.add_node();
        alpha_node->set_name("alpha");
        alpha_node->set_op("Const");
        tensorflow::AttrValue alpha_attr;
        alpha_tensor.AsProtoTensorContent(alpha_attr.mutable_tensor());
        (*alpha_node->mutable_attr())["value"] = alpha_attr;
        tensorflow::AttrValue alpha_dtype_attr;
        alpha_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*alpha_node->mutable_attr())["dtype"] = alpha_dtype_attr;
        
        tensorflow::NodeDef* random_gamma_node = graph_def.add_node();
        random_gamma_node->set_name("random_gamma");
        random_gamma_node->set_op("RandomGamma");
        random_gamma_node->add_input("shape");
        random_gamma_node->add_input("alpha");
        
        tensorflow::AttrValue seed_attr;
        seed_attr.set_i(seed);
        (*random_gamma_node->mutable_attr())["seed"] = seed_attr;
        
        tensorflow::AttrValue seed2_attr;
        seed2_attr.set_i(seed2);
        (*random_gamma_node->mutable_attr())["seed2"] = seed2_attr;
        
        tensorflow::AttrValue output_dtype_attr;
        output_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*random_gamma_node->mutable_attr())["T"] = output_dtype_attr;
        
        tensorflow::AttrValue shape_dtype_attr2;
        shape_dtype_attr2.set_type(tensorflow::DT_INT32);
        (*random_gamma_node->mutable_attr())["S"] = shape_dtype_attr2;
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (session->Create(graph_def).ok()) {
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session->Run({}, {"random_gamma"}, {}, &outputs);
            session->Close();
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}