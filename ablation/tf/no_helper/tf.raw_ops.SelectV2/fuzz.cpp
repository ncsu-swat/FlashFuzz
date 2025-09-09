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
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type info from fuzzer input
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t data_type = *reinterpret_cast<const uint32_t*>(data + offset) % 3;
        offset += 4;
        uint32_t remaining_size = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        if (offset >= size) return 0;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create tensor shapes
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create condition tensor (bool)
        tensorflow::Tensor condition_tensor(tensorflow::DT_BOOL, shape);
        auto condition_flat = condition_tensor.flat<bool>();
        for (int i = 0; i < condition_flat.size() && offset < size; ++i) {
            condition_flat(i) = (data[offset] % 2) == 1;
            offset++;
        }
        
        tensorflow::DataType dt;
        switch (data_type) {
            case 0: dt = tensorflow::DT_FLOAT; break;
            case 1: dt = tensorflow::DT_INT32; break;
            default: dt = tensorflow::DT_DOUBLE; break;
        }
        
        // Create input tensors based on data type
        if (dt == tensorflow::DT_FLOAT) {
            tensorflow::Tensor t_tensor(tensorflow::DT_FLOAT, shape);
            tensorflow::Tensor e_tensor(tensorflow::DT_FLOAT, shape);
            
            auto t_flat = t_tensor.flat<float>();
            auto e_flat = e_tensor.flat<float>();
            
            for (int i = 0; i < t_flat.size() && offset < size; ++i) {
                t_flat(i) = static_cast<float>(data[offset % size]);
                e_flat(i) = static_cast<float>(data[(offset + 1) % size]);
                offset += 2;
            }
            
            auto condition_op = tensorflow::ops::Const(root, condition_tensor);
            auto t_op = tensorflow::ops::Const(root, t_tensor);
            auto e_op = tensorflow::ops::Const(root, e_tensor);
            
            auto select_op = tensorflow::ops::SelectV2(root, condition_op, t_op, e_op);
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session.Run({select_op}, &outputs);
            
        } else if (dt == tensorflow::DT_INT32) {
            tensorflow::Tensor t_tensor(tensorflow::DT_INT32, shape);
            tensorflow::Tensor e_tensor(tensorflow::DT_INT32, shape);
            
            auto t_flat = t_tensor.flat<int32_t>();
            auto e_flat = e_tensor.flat<int32_t>();
            
            for (int i = 0; i < t_flat.size() && offset < size; ++i) {
                t_flat(i) = static_cast<int32_t>(data[offset % size]);
                e_flat(i) = static_cast<int32_t>(data[(offset + 1) % size]);
                offset += 2;
            }
            
            auto condition_op = tensorflow::ops::Const(root, condition_tensor);
            auto t_op = tensorflow::ops::Const(root, t_tensor);
            auto e_op = tensorflow::ops::Const(root, e_tensor);
            
            auto select_op = tensorflow::ops::SelectV2(root, condition_op, t_op, e_op);
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session.Run({select_op}, &outputs);
            
        } else {
            tensorflow::Tensor t_tensor(tensorflow::DT_DOUBLE, shape);
            tensorflow::Tensor e_tensor(tensorflow::DT_DOUBLE, shape);
            
            auto t_flat = t_tensor.flat<double>();
            auto e_flat = e_tensor.flat<double>();
            
            for (int i = 0; i < t_flat.size() && offset < size; ++i) {
                t_flat(i) = static_cast<double>(data[offset % size]);
                e_flat(i) = static_cast<double>(data[(offset + 1) % size]);
                offset += 2;
            }
            
            auto condition_op = tensorflow::ops::Const(root, condition_tensor);
            auto t_op = tensorflow::ops::Const(root, t_tensor);
            auto e_op = tensorflow::ops::Const(root, e_tensor);
            
            auto select_op = tensorflow::ops::SelectV2(root, condition_op, t_op, e_op);
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session.Run({select_op}, &outputs);
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}