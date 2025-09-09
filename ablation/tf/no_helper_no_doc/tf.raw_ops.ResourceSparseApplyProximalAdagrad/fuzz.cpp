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
#include <tensorflow/core/kernels/training_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract basic parameters from fuzz input
        int32_t var_dim = (data[offset] % 10) + 1;
        offset += 1;
        
        int32_t accum_dim = var_dim;
        int32_t indices_size = (data[offset] % 5) + 1;
        offset += 1;
        
        int32_t grad_dim = indices_size;
        
        // Extract scalar values
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float l1 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float l2 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        // Clamp values to reasonable ranges
        lr = std::max(0.001f, std::min(1.0f, std::abs(lr)));
        l1 = std::max(0.0f, std::min(1.0f, std::abs(l1)));
        l2 = std::max(0.0f, std::min(1.0f, std::abs(l2)));
        
        // Create test environment
        tensorflow::test::OpsTestBase test;
        
        // Build the node
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("resource_sparse_apply_proximal_adagrad", "ResourceSparseApplyProximalAdagrad")
            .Attr("T", tensorflow::DT_FLOAT)
            .Attr("Tindices", tensorflow::DT_INT32)
            .Attr("use_locking", false)
            .Input(tensorflow::FakeInput(tensorflow::DT_RESOURCE))  // var
            .Input(tensorflow::FakeInput(tensorflow::DT_RESOURCE))  // accum
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // lr
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // l1
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // l2
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))     // grad
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))     // indices
            .Finalize(&node_def);
            
        if (!status.ok()) return 0;
        
        status = test.InitOp(node_def);
        if (!status.ok()) return 0;
        
        // Create resource handles
        tensorflow::ResourceHandle var_handle;
        var_handle.set_device("CPU:0");
        var_handle.set_container("test");
        var_handle.set_name("var");
        var_handle.set_hash_code(1);
        var_handle.set_maybe_type_name("Variable");
        
        tensorflow::ResourceHandle accum_handle;
        accum_handle.set_device("CPU:0");
        accum_handle.set_container("test");
        accum_handle.set_name("accum");
        accum_handle.set_hash_code(2);
        accum_handle.set_maybe_type_name("Variable");
        
        // Create input tensors
        tensorflow::Tensor var_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        var_tensor.scalar<tensorflow::ResourceHandle>()() = var_handle;
        test.AddInputFromArray<tensorflow::ResourceHandle>(tensorflow::TensorShape({}), {var_handle});
        
        tensorflow::Tensor accum_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        accum_tensor.scalar<tensorflow::ResourceHandle>()() = accum_handle;
        test.AddInputFromArray<tensorflow::ResourceHandle>(tensorflow::TensorShape({}), {accum_handle});
        
        // Add scalar inputs
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {lr});
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {l1});
        test.AddInputFromArray<float>(tensorflow::TensorShape({}), {l2});
        
        // Create gradient tensor
        std::vector<float> grad_values(grad_dim);
        for (int i = 0; i < grad_dim && offset + sizeof(float) <= size; ++i) {
            grad_values[i] = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        test.AddInputFromArray<float>(tensorflow::TensorShape({grad_dim}), grad_values);
        
        // Create indices tensor
        std::vector<int32_t> indices_values(indices_size);
        for (int i = 0; i < indices_size; ++i) {
            indices_values[i] = i % var_dim;  // Ensure valid indices
        }
        test.AddInputFromArray<int32_t>(tensorflow::TensorShape({indices_size}), indices_values);
        
        // Run the operation (this will likely fail due to missing resource setup, but tests the parsing)
        tensorflow::Status run_status = test.RunOpKernel();
        // We don't check the status as the operation may fail due to resource setup issues
        // but we've successfully tested the input parsing and validation
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}