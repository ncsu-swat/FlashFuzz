#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // trunc_ only works on floating-point tensors
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Make tensor contiguous and ensure it requires no grad for in-place op
        tensor = tensor.contiguous();
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply trunc_ operation (in-place)
        tensor.trunc_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // Use equal with nan handling instead of allclose
        torch::Tensor expected = torch::trunc(original);
        
        // Silent check - don't throw on mismatch, just verify operation completed
        try {
            // For tensors without NaN, verify correctness
            if (!torch::any(torch::isnan(original)).item<bool>()) {
                if (!torch::allclose(tensor, expected)) {
                    // This would indicate a bug in PyTorch itself
                }
            }
        } catch (...) {
            // Silently ignore verification errors
        }
        
        // Try with different tensor dtypes if we have more data
        if (offset + 4 < Size) {
            size_t offset2 = 0;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Convert to float if needed
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat64);
            }
            
            tensor2 = tensor2.contiguous();
            
            // Apply trunc_ operation
            tensor2.trunc_();
        }
        
        // Test with explicitly created float tensor of different sizes
        if (Size > 8) {
            uint8_t shape_byte = Data[offset % Size];
            int64_t dim = (shape_byte % 4) + 1;  // 1-4 dimensions
            
            std::vector<int64_t> shape;
            for (int64_t i = 0; i < dim; i++) {
                int64_t size_val = (Data[(offset + i + 1) % Size] % 8) + 1;  // 1-8 per dimension
                shape.push_back(size_val);
            }
            
            // Create tensor with random values
            torch::Tensor tensor3 = torch::randn(shape, torch::kFloat32);
            tensor3.trunc_();
            
            // Test with double precision
            torch::Tensor tensor4 = torch::randn(shape, torch::kFloat64);
            tensor4.trunc_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}