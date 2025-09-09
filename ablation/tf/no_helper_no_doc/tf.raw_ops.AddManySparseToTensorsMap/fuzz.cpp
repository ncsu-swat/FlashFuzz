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
        
        // Extract basic parameters from fuzz input
        int32_t batch_size = (data[offset] % 10) + 1;
        offset += 1;
        
        int32_t num_features = (data[offset] % 20) + 1;
        offset += 1;
        
        int32_t nnz = (data[offset] % 50) + 1;
        offset += 1;
        
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>((data[offset] % 4) + 1); // DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64
        offset += 1;
        
        if (dtype != tensorflow::DT_FLOAT && dtype != tensorflow::DT_DOUBLE && 
            dtype != tensorflow::DT_INT32 && dtype != tensorflow::DT_INT64) {
            dtype = tensorflow::DT_FLOAT;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create sparse_indices tensor (nnz x 2)
        tensorflow::NodeDef sparse_indices_node;
        sparse_indices_node.set_name("sparse_indices");
        sparse_indices_node.set_op("Const");
        sparse_indices_node.mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
        sparse_indices_node.mutable_attr()->at("dtype").set_type(tensorflow::DT_INT64);
        
        tensorflow::TensorProto sparse_indices_proto;
        sparse_indices_proto.set_dtype(tensorflow::DT_INT64);
        sparse_indices_proto.mutable_tensor_shape()->add_dim()->set_size(nnz);
        sparse_indices_proto.mutable_tensor_shape()->add_dim()->set_size(2);
        
        for (int i = 0; i < nnz * 2 && offset < size; ++i) {
            int64_t val = static_cast<int64_t>(data[offset] % (i < nnz ? batch_size : num_features));
            sparse_indices_proto.add_int64_val(val);
            offset++;
        }
        
        sparse_indices_node.mutable_attr()->insert({"value", tensorflow::AttrValue()});
        *sparse_indices_node.mutable_attr()->at("value").mutable_tensor() = sparse_indices_proto;
        *graph_def.add_node() = sparse_indices_node;
        
        // Create sparse_values tensor
        tensorflow::NodeDef sparse_values_node;
        sparse_values_node.set_name("sparse_values");
        sparse_values_node.set_op("Const");
        sparse_values_node.mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
        sparse_values_node.mutable_attr()->at("dtype").set_type(dtype);
        
        tensorflow::TensorProto sparse_values_proto;
        sparse_values_proto.set_dtype(dtype);
        sparse_values_proto.mutable_tensor_shape()->add_dim()->set_size(nnz);
        
        for (int i = 0; i < nnz && offset < size; ++i) {
            if (dtype == tensorflow::DT_FLOAT) {
                float val = static_cast<float>(data[offset]) / 255.0f;
                sparse_values_proto.add_float_val(val);
            } else if (dtype == tensorflow::DT_DOUBLE) {
                double val = static_cast<double>(data[offset]) / 255.0;
                sparse_values_proto.add_double_val(val);
            } else if (dtype == tensorflow::DT_INT32) {
                int32_t val = static_cast<int32_t>(data[offset]);
                sparse_values_proto.add_int_val(val);
            } else if (dtype == tensorflow::DT_INT64) {
                int64_t val = static_cast<int64_t>(data[offset]);
                sparse_values_proto.add_int64_val(val);
            }
            offset++;
        }
        
        sparse_values_node.mutable_attr()->insert({"value", tensorflow::AttrValue()});
        *sparse_values_node.mutable_attr()->at("value").mutable_tensor() = sparse_values_proto;
        *graph_def.add_node() = sparse_values_node;
        
        // Create sparse_shape tensor
        tensorflow::NodeDef sparse_shape_node;
        sparse_shape_node.set_name("sparse_shape");
        sparse_shape_node.set_op("Const");
        sparse_shape_node.mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
        sparse_shape_node.mutable_attr()->at("dtype").set_type(tensorflow::DT_INT64);
        
        tensorflow::TensorProto sparse_shape_proto;
        sparse_shape_proto.set_dtype(tensorflow::DT_INT64);
        sparse_shape_proto.mutable_tensor_shape()->add_dim()->set_size(2);
        sparse_shape_proto.add_int64_val(batch_size);
        sparse_shape_proto.add_int64_val(num_features);
        
        sparse_shape_node.mutable_attr()->insert({"value", tensorflow::AttrValue()});
        *sparse_shape_node.mutable_attr()->at("value").mutable_tensor() = sparse_shape_proto;
        *graph_def.add_node() = sparse_shape_node;
        
        // Create AddManySparseToTensorsMap node
        tensorflow::NodeDef add_many_node;
        add_many_node.set_name("add_many_sparse");
        add_many_node.set_op("AddManySparseToTensorsMap");
        add_many_node.add_input("sparse_indices");
        add_many_node.add_input("sparse_values");
        add_many_node.add_input("sparse_shape");
        
        add_many_node.mutable_attr()->insert({"T", tensorflow::AttrValue()});
        add_many_node.mutable_attr()->at("T").set_type(dtype);
        
        if (offset < size) {
            std::string container_str(reinterpret_cast<const char*>(data + offset), std::min(size - offset, size_t(10)));
            add_many_node.mutable_attr()->insert({"container", tensorflow::AttrValue()});
            add_many_node.mutable_attr()->at("container").set_s(container_str);
        }
        
        *graph_def.add_node() = add_many_node;
        
        // Create and run session
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"add_many_sparse:0"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}