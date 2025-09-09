#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/attr_value.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        // Extract dtype index
        uint8_t dtype_idx = data[offset++] % 19;
        tensorflow::DataType dtypes[] = {
            tensorflow::DT_FLOAT, tensorflow::DT_DOUBLE, tensorflow::DT_INT32,
            tensorflow::DT_UINT8, tensorflow::DT_INT16, tensorflow::DT_INT8,
            tensorflow::DT_COMPLEX64, tensorflow::DT_INT64, tensorflow::DT_QINT8,
            tensorflow::DT_QUINT8, tensorflow::DT_QINT32, tensorflow::DT_BFLOAT16,
            tensorflow::DT_QINT16, tensorflow::DT_QUINT16, tensorflow::DT_UINT16,
            tensorflow::DT_COMPLEX128, tensorflow::DT_HALF, tensorflow::DT_UINT32,
            tensorflow::DT_UINT64
        };
        tensorflow::DataType dtype = dtypes[dtype_idx];
        
        // Extract shape dimensions
        int num_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        std::vector<int64_t> shape_dims;
        for (int i = 0; i < num_dims && offset < size; i++) {
            int64_t dim = (data[offset++] % 10) + 1; // 1-10 size per dimension
            shape_dims.push_back(dim);
        }
        
        if (offset >= size) return 0;
        
        // Extract container string length and content
        uint8_t container_len = data[offset++] % 32;
        std::string container = "";
        if (container_len > 0 && offset + container_len <= size) {
            container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
            offset += container_len;
        }
        
        if (offset >= size) return 0;
        
        // Extract shared_name string length and content
        uint8_t shared_name_len = data[offset++] % 32;
        std::string shared_name = "";
        if (shared_name_len > 0 && offset + shared_name_len <= size) {
            shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
            offset += shared_name_len;
        }
        
        if (offset >= size) return 0;
        
        // Extract reduction_type
        std::string reduction_type = (data[offset++] % 2 == 0) ? "MEAN" : "SUM";
        
        // Create GraphDef
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("sparse_conditional_accumulator");
        node_def->set_op("SparseConditionalAccumulator");
        
        // Set dtype attribute
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(dtype);
        (*node_def->mutable_attr())["dtype"] = dtype_attr;
        
        // Set shape attribute
        tensorflow::AttrValue shape_attr;
        tensorflow::TensorShapeProto* shape_proto = shape_attr.mutable_shape();
        for (int64_t dim : shape_dims) {
            shape_proto->add_dim()->set_size(dim);
        }
        (*node_def->mutable_attr())["shape"] = shape_attr;
        
        // Set container attribute
        tensorflow::AttrValue container_attr;
        container_attr.set_s(container);
        (*node_def->mutable_attr())["container"] = container_attr;
        
        // Set shared_name attribute
        tensorflow::AttrValue shared_name_attr;
        shared_name_attr.set_s(shared_name);
        (*node_def->mutable_attr())["shared_name"] = shared_name_attr;
        
        // Set reduction_type attribute
        tensorflow::AttrValue reduction_type_attr;
        reduction_type_attr.set_s(reduction_type);
        (*node_def->mutable_attr())["reduction_type"] = reduction_type_attr;
        
        // Create session and run
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"sparse_conditional_accumulator:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully created accumulator
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}