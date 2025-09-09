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
        
        // Extract dimensions
        uint32_t num_segments = (data[offset] % 10) + 1;
        offset += 1;
        uint32_t data_rows = (data[offset] % 10) + 1;
        offset += 1;
        uint32_t data_cols = (data[offset] % 10) + 1;
        offset += 1;
        
        if (offset + data_rows * data_cols * sizeof(float) + data_rows * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create data tensor
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({static_cast<int64_t>(data_rows), 
                                                             static_cast<int64_t>(data_cols)}));
        auto data_flat = data_tensor.flat<float>();
        
        for (int i = 0; i < data_rows * data_cols; ++i) {
            if (offset + sizeof(float) > size) return 0;
            float val;
            memcpy(&val, data + offset, sizeof(float));
            data_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create segment_ids tensor
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                             tensorflow::TensorShape({static_cast<int64_t>(data_rows)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        
        for (int i = 0; i < data_rows; ++i) {
            if (offset + sizeof(int32_t) > size) return 0;
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            // Ensure segment_ids are non-negative and sorted
            val = abs(val) % num_segments;
            if (i > 0 && val < segment_ids_flat(i-1)) {
                val = segment_ids_flat(i-1);
            }
            segment_ids_flat(i) = val;
            offset += sizeof(int32_t);
        }
        
        // Create graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* data_node = graph_def.add_node();
        data_node->set_name("data");
        data_node->set_op("Placeholder");
        (*data_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*data_node->mutable_attr())["shape"].mutable_shape();
        
        tensorflow::NodeDef* segment_ids_node = graph_def.add_node();
        segment_ids_node->set_name("segment_ids");
        segment_ids_node->set_op("Placeholder");
        (*segment_ids_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        (*segment_ids_node->mutable_attr())["shape"].mutable_shape();
        
        tensorflow::NodeDef* segment_prod_node = graph_def.add_node();
        segment_prod_node->set_name("segment_prod");
        segment_prod_node->set_op("SegmentProd");
        segment_prod_node->add_input("data");
        segment_prod_node->add_input("segment_ids");
        (*segment_prod_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*segment_prod_node->mutable_attr())["Tindices"].set_type(tensorflow::DT_INT32);
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"data", data_tensor},
            {"segment_ids", segment_ids_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"segment_prod"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape and basic properties
            const tensorflow::Tensor& result = outputs[0];
            if (result.dims() == 2 && result.dim_size(1) == data_cols) {
                auto result_flat = result.flat<float>();
                for (int i = 0; i < result.NumElements(); ++i) {
                    if (!std::isfinite(result_flat(i))) {
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