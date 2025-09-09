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
        int64_t num_indices = (data[offset] % 10) + 1;
        offset++;
        int64_t num_dims = (data[offset] % 5) + 2;
        offset++;
        int64_t num_values = num_indices;
        bool keep_dims = data[offset] % 2;
        offset++;
        
        if (offset + num_indices * num_dims * 8 + num_values * 4 + num_dims * 8 > size) {
            return 0;
        }
        
        // Create input_indices tensor (int64)
        tensorflow::Tensor input_indices(tensorflow::DT_INT64, 
            tensorflow::TensorShape({num_indices, num_dims}));
        auto indices_flat = input_indices.flat<int64_t>();
        for (int i = 0; i < num_indices * num_dims; i++) {
            int64_t val = 0;
            memcpy(&val, data + offset, sizeof(int64_t));
            indices_flat(i) = std::abs(val) % 100;  // Keep values reasonable
            offset += sizeof(int64_t);
        }
        
        // Create input_values tensor (float)
        tensorflow::Tensor input_values(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({num_values}));
        auto values_flat = input_values.flat<float>();
        for (int i = 0; i < num_values; i++) {
            float val = 0.0f;
            memcpy(&val, data + offset, sizeof(float));
            if (std::isnan(val) || std::isinf(val)) {
                val = 1.0f;
            }
            values_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create input_shape tensor (int64)
        tensorflow::Tensor input_shape(tensorflow::DT_INT64, 
            tensorflow::TensorShape({num_dims}));
        auto shape_flat = input_shape.flat<int64_t>();
        for (int i = 0; i < num_dims; i++) {
            int64_t val = 0;
            memcpy(&val, data + offset, sizeof(int64_t));
            shape_flat(i) = std::abs(val) % 100 + 1;  // Ensure positive shape
            offset += sizeof(int64_t);
        }
        
        // Create reduction_axes tensor (int32)
        int32_t num_reduction_axes = (data[offset % size] % num_dims) + 1;
        tensorflow::Tensor reduction_axes(tensorflow::DT_INT32, 
            tensorflow::TensorShape({num_reduction_axes}));
        auto axes_flat = reduction_axes.flat<int32_t>();
        for (int i = 0; i < num_reduction_axes; i++) {
            axes_flat(i) = i % num_dims;
        }
        
        // Create a simple test using OpsTestBase
        class SparseReduceSumSparseTest : public tensorflow::OpsTestBase {};
        SparseReduceSumSparseTest test;
        
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("sparse_reduce_sum_sparse", "SparseReduceSumSparse")
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))    // input_indices
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))    // input_values  
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))    // input_shape
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))    // reduction_axes
            .Attr("keep_dims", keep_dims)
            .Finalize(&node_def);
            
        if (!status.ok()) {
            return 0;
        }
        
        status = test.InitOp(node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add inputs
        test.AddInputFromArray<int64_t>(input_indices.shape(), 
            input_indices.flat<int64_t>());
        test.AddInputFromArray<float>(input_values.shape(), 
            input_values.flat<float>());
        test.AddInputFromArray<int64_t>(input_shape.shape(), 
            input_shape.flat<int64_t>());
        test.AddInputFromArray<int32_t>(reduction_axes.shape(), 
            reduction_axes.flat<int32_t>());
        
        // Run the operation
        status = test.RunOpKernel();
        if (!status.ok()) {
            // Operation failed, but this is acceptable for fuzzing
            return 0;
        }
        
        // Check outputs exist
        if (test.GetOutput(0) != nullptr && 
            test.GetOutput(1) != nullptr && 
            test.GetOutput(2) != nullptr) {
            // Successfully executed
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}