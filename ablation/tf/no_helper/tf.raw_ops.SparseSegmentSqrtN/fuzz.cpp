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
        uint32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        uint32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        uint32_t indices_size = (data[offset] % data_rows) + 1;
        offset++;
        uint32_t num_segments = (data[offset] % indices_size) + 1;
        offset++;
        bool sparse_gradient = data[offset] % 2;
        offset++;
        
        if (offset + data_rows * data_cols * sizeof(float) + indices_size * sizeof(int32_t) + indices_size * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create data tensor
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(data_rows), static_cast<int64_t>(data_cols)}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols; i++) {
            if (offset + sizeof(float) <= size) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                data_flat(i) = val;
                offset += sizeof(float);
            } else {
                data_flat(i) = 0.0f;
            }
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                indices_flat(i) = abs(val) % data_rows;
                offset += sizeof(int32_t);
            } else {
                indices_flat(i) = i % data_rows;
            }
        }
        
        // Create segment_ids tensor (sorted)
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                segment_ids_flat(i) = abs(val) % num_segments;
                offset += sizeof(int32_t);
            } else {
                segment_ids_flat(i) = i % num_segments;
            }
        }
        
        // Sort segment_ids to meet the requirement
        std::vector<int32_t> sorted_segments(indices_size);
        for (int i = 0; i < indices_size; i++) {
            sorted_segments[i] = segment_ids_flat(i);
        }
        std::sort(sorted_segments.begin(), sorted_segments.end());
        for (int i = 0; i < indices_size; i++) {
            segment_ids_flat(i) = sorted_segments[i];
        }
        
        // Build the graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("sparse_segment_sqrt_n");
        node->set_op("SparseSegmentSqrtN");
        
        // Add inputs
        node->add_input("data:0");
        node->add_input("indices:0");
        node->add_input("segment_ids:0");
        
        // Set attributes
        tensorflow::AttrValue sparse_gradient_attr;
        sparse_gradient_attr.set_b(sparse_gradient);
        (*node->mutable_attr())["sparse_gradient"] = sparse_gradient_attr;
        
        // Add placeholder nodes
        tensorflow::NodeDef* data_node = graph_def.add_node();
        data_node->set_name("data");
        data_node->set_op("Placeholder");
        tensorflow::AttrValue data_dtype;
        data_dtype.set_type(tensorflow::DT_FLOAT);
        (*data_node->mutable_attr())["dtype"] = data_dtype;
        
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        tensorflow::AttrValue indices_dtype;
        indices_dtype.set_type(tensorflow::DT_INT32);
        (*indices_node->mutable_attr())["dtype"] = indices_dtype;
        
        tensorflow::NodeDef* segment_ids_node = graph_def.add_node();
        segment_ids_node->set_name("segment_ids");
        segment_ids_node->set_op("Placeholder");
        tensorflow::AttrValue segment_ids_dtype;
        segment_ids_dtype.set_type(tensorflow::DT_INT32);
        (*segment_ids_node->mutable_attr())["dtype"] = segment_ids_dtype;
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"data:0", data_tensor},
            {"indices:0", indices_tensor},
            {"segment_ids:0", segment_ids_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_segment_sqrt_n:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed the operation
            auto output_flat = outputs[0].flat<float>();
            volatile float sum = 0.0f;
            for (int i = 0; i < outputs[0].NumElements(); i++) {
                sum += output_flat(i);
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