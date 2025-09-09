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
        
        // Extract shape dimensions
        int32_t shape_dims = (data[offset] % 4) + 1;
        offset++;
        
        if (offset + shape_dims * 4 + 16 > size) return 0;
        
        // Create shape tensor
        tensorflow::TensorShape shape_tensor_shape({shape_dims});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int32_t>();
        
        for (int i = 0; i < shape_dims; i++) {
            int32_t dim_size = 1 + (*(reinterpret_cast<const uint32_t*>(data + offset)) % 10);
            shape_flat(i) = dim_size;
            offset += 4;
        }
        
        // Extract minval and maxval
        int64_t minval = *(reinterpret_cast<const int64_t*>(data + offset));
        offset += 8;
        int64_t maxval = *(reinterpret_cast<const int64_t*>(data + offset));
        offset += 8;
        
        // Ensure maxval > minval
        if (maxval <= minval) {
            maxval = minval + 1;
        }
        
        // Create minval and maxval tensors
        tensorflow::Tensor minval_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        minval_tensor.scalar<int64_t>()() = minval;
        
        tensorflow::Tensor maxval_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        maxval_tensor.scalar<int64_t>()() = maxval;
        
        // Extract seed values if available
        int64_t seed = 0;
        int64_t seed2 = 0;
        if (offset + 16 <= size) {
            seed = *(reinterpret_cast<const int64_t*>(data + offset));
            offset += 8;
            seed2 = *(reinterpret_cast<const int64_t*>(data + offset));
            offset += 8;
        }
        
        // Create test utility
        tensorflow::OpsTestBase test;
        
        // Build the node
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("random_uniform_int", "RandomUniformInt")
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))
            .Attr("seed", seed)
            .Attr("seed2", seed2)
            .Attr("Tout", tensorflow::DT_INT64)
            .Finalize(&node_def);
        
        if (!status.ok()) return 0;
        
        status = test.InitOp(node_def);
        if (!status.ok()) return 0;
        
        // Add inputs
        test.AddInputFromArray<int32_t>(shape_tensor.shape(), shape_flat.data());
        test.AddInputFromArray<int64_t>(minval_tensor.shape(), {minval});
        test.AddInputFromArray<int64_t>(maxval_tensor.shape(), {maxval});
        
        // Run the operation
        status = test.RunOpKernel();
        if (!status.ok()) return 0;
        
        // Get output
        tensorflow::Tensor* output = test.GetOutput(0);
        if (output == nullptr) return 0;
        
        // Verify output shape matches input shape
        if (output->dims() != shape_dims) return 0;
        
        for (int i = 0; i < shape_dims; i++) {
            if (output->dim_size(i) != shape_flat(i)) return 0;
        }
        
        // Verify output values are in range [minval, maxval)
        auto output_flat = output->flat<int64_t>();
        for (int i = 0; i < output_flat.size(); i++) {
            int64_t val = output_flat(i);
            if (val < minval || val >= maxval) return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}