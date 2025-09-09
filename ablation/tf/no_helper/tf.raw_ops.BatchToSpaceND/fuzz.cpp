#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for input tensor
        uint32_t batch_size = (data[offset] % 8) + 1;
        offset++;
        uint32_t spatial_dim1 = (data[offset] % 4) + 1;
        offset++;
        uint32_t spatial_dim2 = (data[offset] % 4) + 1;
        offset++;
        uint32_t remaining_dim = (data[offset] % 4) + 1;
        offset++;
        
        // Extract block_shape values
        uint32_t block_shape_0 = (data[offset] % 4) + 1;
        offset++;
        uint32_t block_shape_1 = (data[offset] % 4) + 1;
        offset++;
        
        // Ensure batch_size is divisible by product of block_shape
        uint32_t block_prod = block_shape_0 * block_shape_1;
        batch_size = ((batch_size / block_prod) + 1) * block_prod;
        
        // Extract crop values
        uint32_t crop_0_start = data[offset] % (block_shape_0 * spatial_dim1);
        offset++;
        uint32_t crop_0_end = data[offset] % (block_shape_0 * spatial_dim1 - crop_0_start);
        offset++;
        uint32_t crop_1_start = data[offset] % (block_shape_1 * spatial_dim2);
        offset++;
        uint32_t crop_1_end = data[offset] % (block_shape_1 * spatial_dim2 - crop_1_start);
        offset++;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch_size, spatial_dim1, spatial_dim2, remaining_dim});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t tensor_size = input_tensor.NumElements();
        for (size_t i = 0; i < tensor_size && offset < size; i++) {
            input_flat(i) = static_cast<float>(data[offset % size]);
            offset++;
        }
        
        // Create block_shape tensor
        tensorflow::Tensor block_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto block_shape_flat = block_shape_tensor.flat<int32_t>();
        block_shape_flat(0) = static_cast<int32_t>(block_shape_0);
        block_shape_flat(1) = static_cast<int32_t>(block_shape_1);
        
        // Create crops tensor
        tensorflow::Tensor crops_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2, 2}));
        auto crops_flat = crops_tensor.flat<int32_t>();
        crops_flat(0) = static_cast<int32_t>(crop_0_start);
        crops_flat(1) = static_cast<int32_t>(crop_0_end);
        crops_flat(2) = static_cast<int32_t>(crop_1_start);
        crops_flat(3) = static_cast<int32_t>(crop_1_end);
        
        // Create constant ops
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto block_shape_op = tensorflow::ops::Const(root, block_shape_tensor);
        auto crops_op = tensorflow::ops::Const(root, crops_tensor);
        
        // Create BatchToSpaceND operation
        auto batch_to_space_nd = tensorflow::ops::BatchToSpaceND(root, input_op, block_shape_op, crops_op);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({batch_to_space_nd}, &outputs);
        
        if (!status.ok()) {
            std::cout << "BatchToSpaceND operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != tensorflow::DT_FLOAT) {
                std::cout << "Output dtype mismatch" << std::endl;
                return 0;
            }
            
            // Check output shape validity
            auto output_shape = output.shape();
            if (output_shape.dims() != 4) {
                std::cout << "Output dimension mismatch" << std::endl;
                return 0;
            }
            
            // Verify batch dimension
            int64_t expected_batch = batch_size / block_prod;
            if (output_shape.dim_size(0) != expected_batch) {
                std::cout << "Output batch size mismatch" << std::endl;
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}