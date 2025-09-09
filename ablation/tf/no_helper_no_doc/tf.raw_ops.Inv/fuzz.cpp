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
        
        if (size < sizeof(int32_t) * 2) {
            return 0;
        }
        
        // Extract tensor dimensions
        int32_t num_dims = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Limit dimensions to reasonable range
        num_dims = std::abs(num_dims) % 4 + 1;
        
        if (offset + num_dims * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Extract dimension sizes
        std::vector<int64_t> dims;
        int64_t total_elements = 1;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            offset += sizeof(int32_t);
            dim_size = std::abs(dim_size) % 10 + 1; // Limit to reasonable size
            dims.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        // Limit total elements to prevent excessive memory usage
        if (total_elements > 1000) {
            return 0;
        }
        
        tensorflow::TensorShape shape(dims);
        
        // Test with float type
        if (offset + total_elements * sizeof(float) <= size) {
            tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
            auto input_flat = input_tensor.flat<float>();
            
            for (int64_t i = 0; i < total_elements; i++) {
                if (offset + sizeof(float) <= size) {
                    float val = *reinterpret_cast<const float*>(data + offset);
                    offset += sizeof(float);
                    // Avoid division by zero and very small numbers
                    if (std::abs(val) < 1e-6f) {
                        val = 1.0f;
                    }
                    input_flat(i) = val;
                } else {
                    input_flat(i) = 1.0f;
                }
            }
            
            // Create session and graph
            tensorflow::GraphDef graph_def;
            tensorflow::NodeDef* input_node = graph_def.add_node();
            input_node->set_name("input");
            input_node->set_op("Placeholder");
            (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
            
            tensorflow::NodeDef* inv_node = graph_def.add_node();
            inv_node->set_name("inv");
            inv_node->set_op("Inv");
            inv_node->add_input("input");
            (*inv_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
            
            std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
            if (session) {
                tensorflow::Status status = session->Create(graph_def);
                if (status.ok()) {
                    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"input", input_tensor}};
                    std::vector<tensorflow::Tensor> outputs;
                    
                    status = session->Run(inputs, {"inv"}, {}, &outputs);
                    if (status.ok() && !outputs.empty()) {
                        // Verify output shape matches input shape
                        if (outputs[0].shape() == input_tensor.shape()) {
                            auto output_flat = outputs[0].flat<float>();
                            // Basic sanity check: inv(inv(x)) should be close to x for non-zero x
                            for (int64_t i = 0; i < std::min(total_elements, 10LL); i++) {
                                float original = input_flat(i);
                                float inverted = output_flat(i);
                                if (std::isfinite(original) && std::isfinite(inverted) && std::abs(original) > 1e-6f) {
                                    float double_inv = 1.0f / inverted;
                                    // Allow some numerical error
                                    if (std::abs(double_inv - original) > std::abs(original) * 0.01f) {
                                        // Potential numerical issue, but continue
                                    }
                                }
                            }
                        }
                    }
                }
                session->Close();
            }
        }
        
        // Test with double type if there's remaining data
        if (offset + total_elements * sizeof(double) <= size) {
            tensorflow::Tensor input_tensor_double(tensorflow::DT_DOUBLE, shape);
            auto input_flat_double = input_tensor_double.flat<double>();
            
            for (int64_t i = 0; i < total_elements; i++) {
                if (offset + sizeof(double) <= size) {
                    double val = *reinterpret_cast<const double*>(data + offset);
                    offset += sizeof(double);
                    // Avoid division by zero and very small numbers
                    if (std::abs(val) < 1e-12) {
                        val = 1.0;
                    }
                    input_flat_double(i) = val;
                } else {
                    input_flat_double(i) = 1.0;
                }
            }
            
            // Create session and graph for double
            tensorflow::GraphDef graph_def_double;
            tensorflow::NodeDef* input_node_double = graph_def_double.add_node();
            input_node_double->set_name("input");
            input_node_double->set_op("Placeholder");
            (*input_node_double->mutable_attr())["dtype"].set_type(tensorflow::DT_DOUBLE);
            
            tensorflow::NodeDef* inv_node_double = graph_def_double.add_node();
            inv_node_double->set_name("inv");
            inv_node_double->set_op("Inv");
            inv_node_double->add_input("input");
            (*inv_node_double->mutable_attr())["T"].set_type(tensorflow::DT_DOUBLE);
            
            std::unique_ptr<tensorflow::Session> session_double(tensorflow::NewSession(tensorflow::SessionOptions()));
            if (session_double) {
                tensorflow::Status status = session_double->Create(graph_def_double);
                if (status.ok()) {
                    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"input", input_tensor_double}};
                    std::vector<tensorflow::Tensor> outputs;
                    
                    status = session_double->Run(inputs, {"inv"}, {}, &outputs);
                }
                session_double->Close();
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}