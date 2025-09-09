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
        int32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        int32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        int32_t indices_size = (data[offset] % data_rows) + 1;
        offset++;
        int32_t segment_ids_size = indices_size;
        int32_t num_segments = (data[offset] % 5) + 1;
        offset++;
        
        if (offset + data_rows * data_cols * sizeof(float) + 
            indices_size * sizeof(int32_t) + 
            segment_ids_size * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create input tensors
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({data_rows, data_cols}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols; i++) {
            if (offset + sizeof(float) > size) return 0;
            float val;
            memcpy(&val, data + offset, sizeof(float));
            data_flat(i) = val;
            offset += sizeof(float);
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + sizeof(int32_t) > size) return 0;
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            indices_flat(i) = abs(val) % data_rows;
            offset += sizeof(int32_t);
        }
        
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                            tensorflow::TensorShape({segment_ids_size}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < segment_ids_size; i++) {
            if (offset + sizeof(int32_t) > size) return 0;
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            segment_ids_flat(i) = abs(val) % num_segments;
            offset += sizeof(int32_t);
        }
        
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, 
                                             tensorflow::TensorShape({}));
        num_segments_tensor.scalar<int32_t>()() = num_segments;
        
        // Create session and graph
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("sparse_segment_sqrt_n");
        node->set_op("SparseSegmentSqrtNWithNumSegments");
        node->add_input("data:0");
        node->add_input("indices:0");
        node->add_input("segment_ids:0");
        node->add_input("num_segments:0");
        
        tensorflow::NodeDef* data_node = graph_def.add_node();
        data_node->set_name("data");
        data_node->set_op("Placeholder");
        (*data_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        (*indices_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* segment_ids_node = graph_def.add_node();
        segment_ids_node->set_name("segment_ids");
        segment_ids_node->set_op("Placeholder");
        (*segment_ids_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::NodeDef* num_segments_node = graph_def.add_node();
        num_segments_node->set_name("num_segments");
        num_segments_node->set_op("Placeholder");
        (*num_segments_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"data:0", data_tensor},
            {"indices:0", indices_tensor},
            {"segment_ids:0", segment_ids_tensor},
            {"num_segments:0", num_segments_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_segment_sqrt_n:0"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}