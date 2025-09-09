#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/segment_reduction_ops.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/platform/cpu_info.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t data_dims = (data[offset] % 4) + 1;
        offset++;
        
        int32_t segment_dims = 1;
        
        // Extract data tensor dimensions
        std::vector<int64_t> data_shape;
        for (int i = 0; i < data_dims && offset < size; i++) {
            int64_t dim = (data[offset] % 10) + 1;
            data_shape.push_back(dim);
            offset++;
        }
        
        if (data_shape.empty()) {
            data_shape.push_back(1);
        }
        
        // Calculate total elements needed
        int64_t total_data_elements = 1;
        for (auto dim : data_shape) {
            total_data_elements *= dim;
        }
        
        // Limit size to prevent excessive memory usage
        if (total_data_elements > 1000) {
            total_data_elements = 1000;
            data_shape[0] = total_data_elements;
            for (int i = 1; i < data_shape.size(); i++) {
                data_shape[i] = 1;
            }
        }
        
        // Create segment_ids tensor (must be sorted)
        int64_t num_segments = (data_shape[0] / 2) + 1;
        if (num_segments < 1) num_segments = 1;
        
        tensorflow::TensorShape data_tensor_shape;
        for (auto dim : data_shape) {
            data_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape segment_shape;
        segment_shape.AddDim(data_shape[0]);
        
        // Create data tensor with float values
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, data_tensor_shape);
        auto data_flat = data_tensor.flat<float>();
        
        // Fill data tensor with values from fuzz input
        for (int64_t i = 0; i < total_data_elements && offset < size; i++) {
            float val = static_cast<float>(data[offset % size]) / 255.0f;
            data_flat(i) = val;
            offset++;
        }
        
        // Create segment_ids tensor (must be sorted and non-negative)
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, segment_shape);
        auto segment_flat = segment_ids_tensor.flat<int32_t>();
        
        // Generate sorted segment IDs
        int32_t current_segment = 0;
        for (int64_t i = 0; i < data_shape[0]; i++) {
            segment_flat(i) = current_segment;
            if (offset < size && (data[offset] % 3) == 0 && current_segment < num_segments - 1) {
                current_segment++;
            }
            offset++;
        }
        
        // Create session and graph
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::GraphDef graph_def;
        
        // Create SegmentProd node
        tensorflow::NodeDef* segment_prod_node = graph_def.add_node();
        segment_prod_node->set_name("segment_prod");
        segment_prod_node->set_op("SegmentProd");
        segment_prod_node->add_input("data:0");
        segment_prod_node->add_input("segment_ids:0");
        
        tensorflow::AttrValue data_type_attr;
        data_type_attr.set_type(tensorflow::DT_FLOAT);
        (*segment_prod_node->mutable_attr())["T"] = data_type_attr;
        
        tensorflow::AttrValue segment_type_attr;
        segment_type_attr.set_type(tensorflow::DT_INT32);
        (*segment_prod_node->mutable_attr())["Tindices"] = segment_type_attr;
        
        // Create placeholder nodes for inputs
        tensorflow::NodeDef* data_placeholder = graph_def.add_node();
        data_placeholder->set_name("data");
        data_placeholder->set_op("Placeholder");
        tensorflow::AttrValue data_placeholder_type;
        data_placeholder_type.set_type(tensorflow::DT_FLOAT);
        (*data_placeholder->mutable_attr())["dtype"] = data_placeholder_type;
        
        tensorflow::NodeDef* segment_placeholder = graph_def.add_node();
        segment_placeholder->set_name("segment_ids");
        segment_placeholder->set_op("Placeholder");
        tensorflow::AttrValue segment_placeholder_type;
        segment_placeholder_type.set_type(tensorflow::DT_INT32);
        (*segment_placeholder->mutable_attr())["dtype"] = segment_placeholder_type;
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"data:0", data_tensor},
            {"segment_ids:0", segment_ids_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"segment_prod:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed SegmentProd operation
            auto output_shape = outputs[0].shape();
            auto output_flat = outputs[0].flat<float>();
            
            // Basic validation - check output is finite
            for (int i = 0; i < output_flat.size(); i++) {
                if (!std::isfinite(output_flat(i))) {
                    break;
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