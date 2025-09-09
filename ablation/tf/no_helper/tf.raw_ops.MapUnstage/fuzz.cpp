#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract key value
        int64_t key_val = 0;
        if (offset + sizeof(int64_t) <= size) {
            memcpy(&key_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract indices size and values
        int32_t indices_size = 1;
        if (offset + sizeof(int32_t) <= size) {
            memcpy(&indices_size, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            indices_size = std::abs(indices_size) % 10 + 1; // Limit size
        }
        
        std::vector<int32_t> indices_data;
        for (int i = 0; i < indices_size && offset + sizeof(int32_t) <= size; ++i) {
            int32_t idx = 0;
            memcpy(&idx, data + offset, sizeof(int32_t));
            indices_data.push_back(idx);
            offset += sizeof(int32_t);
        }
        
        // Extract dtypes count
        int32_t dtypes_count = 1;
        if (offset + sizeof(int32_t) <= size) {
            memcpy(&dtypes_count, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            dtypes_count = std::abs(dtypes_count) % 5 + 1; // Limit to reasonable size
        }
        
        // Extract optional parameters
        int32_t capacity = 0;
        if (offset + sizeof(int32_t) <= size) {
            memcpy(&capacity, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            capacity = std::abs(capacity) % 1000;
        }
        
        int32_t memory_limit = 0;
        if (offset + sizeof(int32_t) <= size) {
            memcpy(&memory_limit, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            memory_limit = std::abs(memory_limit) % 1000;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create key tensor
        auto key_tensor = tensorflow::ops::Const(root, key_val);
        
        // Create indices tensor
        tensorflow::Tensor indices_tf(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(indices_data.size())}));
        auto indices_flat = indices_tf.flat<int32_t>();
        for (size_t i = 0; i < indices_data.size(); ++i) {
            indices_flat(i) = indices_data[i];
        }
        auto indices_tensor = tensorflow::ops::Const(root, indices_tf);
        
        // Create dtypes list
        std::vector<tensorflow::DataType> dtypes;
        for (int i = 0; i < dtypes_count; ++i) {
            // Use common data types
            tensorflow::DataType dt = static_cast<tensorflow::DataType>((i % 4) + 1); // DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8
            if (dt == tensorflow::DT_FLOAT || dt == tensorflow::DT_DOUBLE || 
                dt == tensorflow::DT_INT32 || dt == tensorflow::DT_UINT8) {
                dtypes.push_back(dt);
            } else {
                dtypes.push_back(tensorflow::DT_FLOAT);
            }
        }
        
        // Create container and shared_name strings
        std::string container = "test_container";
        std::string shared_name = "test_shared";
        
        // Create MapUnstage operation
        auto map_unstage = tensorflow::ops::MapUnstage(
            root,
            key_tensor,
            indices_tensor,
            dtypes,
            tensorflow::ops::MapUnstage::Attrs()
                .Capacity(capacity)
                .MemoryLimit(memory_limit)
                .Container(container)
                .SharedName(shared_name)
        );
        
        // Create session and run (this will likely fail since no data was staged)
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // The operation will likely block or fail since no data is staged
        // We just want to test that the operation can be created without crashing
        auto status = session.Run({map_unstage}, &outputs);
        
        // Don't check status as this operation is expected to fail/block
        // when no corresponding MapStage operation has been run
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}