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
        
        if (size < 20) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        uint32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        uint32_t indices_size = (data[offset] % data_rows) + 1;
        offset++;
        uint32_t num_segments_val = (data[offset] % 5) + 1;
        offset++;
        bool sparse_gradient = data[offset] % 2;
        offset++;
        
        if (offset + data_rows * data_cols * 4 + indices_size * 4 + indices_size * 4 + 4 > size) {
            return 0;
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create data tensor (float32)
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({static_cast<int64_t>(data_rows), 
                                                             static_cast<int64_t>(data_cols)}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            data_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create indices tensor (int32)
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            indices_flat(i) = std::abs(val) % data_rows;
            offset += sizeof(int32_t);
        }
        
        // Create segment_ids tensor (int32)
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                             tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            segment_ids_flat(i) = std::abs(val) % num_segments_val;
            offset += sizeof(int32_t);
        }
        
        // Create num_segments tensor (int32)
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_segments_tensor.scalar<int32_t>()() = num_segments_val;
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("sparse_segment_sqrt_n_with_num_segments");
        node_def->set_op("SparseSegmentSqrtNWithNumSegments");
        
        // Add input names
        node_def->add_input("data:0");
        node_def->add_input("indices:0");
        node_def->add_input("segment_ids:0");
        node_def->add_input("num_segments:0");
        
        // Set sparse_gradient attribute
        tensorflow::AttrValue sparse_gradient_attr;
        sparse_gradient_attr.set_b(sparse_gradient);
        (*node_def->mutable_attr())["sparse_gradient"] = sparse_gradient_attr;
        
        // Add placeholder nodes for inputs
        tensorflow::NodeDef* data_node = graph_def.add_node();
        data_node->set_name("data");
        data_node->set_op("Placeholder");
        tensorflow::AttrValue data_dtype_attr;
        data_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*data_node->mutable_attr())["dtype"] = data_dtype_attr;
        
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Placeholder");
        tensorflow::AttrValue indices_dtype_attr;
        indices_dtype_attr.set_type(tensorflow::DT_INT32);
        (*indices_node->mutable_attr())["dtype"] = indices_dtype_attr;
        
        tensorflow::NodeDef* segment_ids_node = graph_def.add_node();
        segment_ids_node->set_name("segment_ids");
        segment_ids_node->set_op("Placeholder");
        tensorflow::AttrValue segment_ids_dtype_attr;
        segment_ids_dtype_attr.set_type(tensorflow::DT_INT32);
        (*segment_ids_node->mutable_attr())["dtype"] = segment_ids_dtype_attr;
        
        tensorflow::NodeDef* num_segments_node = graph_def.add_node();
        num_segments_node->set_name("num_segments");
        num_segments_node->set_op("Placeholder");
        tensorflow::AttrValue num_segments_dtype_attr;
        num_segments_dtype_attr.set_type(tensorflow::DT_INT32);
        (*num_segments_node->mutable_attr())["dtype"] = num_segments_dtype_attr;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"data:0", data_tensor},
            {"indices:0", indices_tensor},
            {"segment_ids:0", segment_ids_tensor},
            {"num_segments:0", num_segments_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"sparse_segment_sqrt_n_with_num_segments:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
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