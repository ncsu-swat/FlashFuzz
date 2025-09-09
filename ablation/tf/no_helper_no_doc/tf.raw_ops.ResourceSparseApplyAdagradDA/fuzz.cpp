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
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_handle.h>
#include <tensorflow/core/kernels/variable_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract basic parameters from fuzz data
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float l1 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float l2 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        int64_t global_step = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        
        if (offset + 16 > size) return 0;
        
        // Extract dimensions
        int var_dim0 = (data[offset] % 10) + 1;
        int var_dim1 = (data[offset + 1] % 10) + 1;
        int indices_size = (data[offset + 2] % 5) + 1;
        offset += 3;
        
        // Create test environment
        tensorflow::test::OpsTestBase test;
        
        // Build node definition
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("resource_sparse_apply_adagrad_da", "ResourceSparseApplyAdagradDA")
            .Input(tensorflow::FakeInput(tensorflow::DT_RESOURCE))  // var
            .Input(tensorflow::FakeInput(tensorflow::DT_RESOURCE))  // gradient_accumulator
            .Input(tensorflow::FakeInput(tensorflow::DT_RESOURCE))  // gradient_squared_accumulator
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // grad
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))     // indices
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // lr
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // l1
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // l2
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))     // global_step
            .Attr("T", tensorflow::DT_FLOAT)
            .Attr("Tindices", tensorflow::DT_INT32)
            .Finalize(&node_def);
            
        if (!status.ok()) return 0;
        
        test.InitOp(node_def);
        
        // Create resource handles
        tensorflow::ResourceHandle var_handle;
        var_handle.set_device("CPU:0");
        var_handle.set_container("test");
        var_handle.set_name("var");
        var_handle.set_hash_code(1);
        var_handle.set_maybe_type_name("VarHandleOp");
        
        tensorflow::ResourceHandle grad_acc_handle;
        grad_acc_handle.set_device("CPU:0");
        grad_acc_handle.set_container("test");
        grad_acc_handle.set_name("grad_acc");
        grad_acc_handle.set_hash_code(2);
        grad_acc_handle.set_maybe_type_name("VarHandleOp");
        
        tensorflow::ResourceHandle grad_sq_acc_handle;
        grad_sq_acc_handle.set_device("CPU:0");
        grad_sq_acc_handle.set_container("test");
        grad_sq_acc_handle.set_name("grad_sq_acc");
        grad_sq_acc_handle.set_hash_code(3);
        grad_sq_acc_handle.set_maybe_type_name("VarHandleOp");
        
        // Add inputs
        test.AddInputFromArray<tensorflow::ResourceHandle>(
            tensorflow::TensorShape({}), {var_handle});
        test.AddInputFromArray<tensorflow::ResourceHandle>(
            tensorflow::TensorShape({}), {grad_acc_handle});
        test.AddInputFromArray<tensorflow::ResourceHandle>(
            tensorflow::TensorShape({}), {grad_sq_acc_handle});
        
        // Create gradient tensor
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({indices_size, var_dim1}));
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < grad_flat.size() && offset < size; ++i) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        test.AddInputFromArray<float>(grad_tensor.shape(), grad_flat.data());
        
        // Create indices tensor
        std::vector<int32_t> indices_data;
        for (int i = 0; i < indices_size; ++i) {
            indices_data.push_back(i % var_dim0);
        }
        test.AddInputFromArray<int32_t>(
            tensorflow::TensorShape({indices_size}), indices_data);
        
        // Add scalar inputs
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {lr});
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {l1});
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {l2});
        test.AddInputFromArray<int64_t>(tensorflow::TensorShape({}), {global_step});
        
        // Run the operation
        status = test.RunOpKernel();
        
        // The operation may fail due to resource not found, which is expected in fuzzing
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}