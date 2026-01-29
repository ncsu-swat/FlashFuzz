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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Data[offset] % 4) + 1;
        offset++;
        
        // For hstack to work, tensors must have compatible shapes:
        // - 1-D tensors can be stacked together
        // - For higher dimensions, all dims except dim 1 must match
        
        // Use remaining bytes to determine common dimensions and sizes
        uint8_t mode = (offset < Size) ? Data[offset] % 3 : 0;
        offset++;
        
        std::vector<torch::Tensor> tensors;
        
        if (mode == 0) {
            // Mode 0: Create 1-D tensors (simplest case for hstack)
            for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
                int64_t len = (offset < Size) ? (Data[offset] % 16) + 1 : 4;
                offset++;
                torch::Tensor tensor = torch::randn({len});
                tensors.push_back(tensor);
            }
        } else if (mode == 1) {
            // Mode 1: Create 2-D tensors with same number of rows
            int64_t rows = (offset < Size) ? (Data[offset] % 8) + 1 : 4;
            offset++;
            for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
                int64_t cols = (offset < Size) ? (Data[offset] % 8) + 1 : 4;
                offset++;
                torch::Tensor tensor = torch::randn({rows, cols});
                tensors.push_back(tensor);
            }
        } else {
            // Mode 2: Create 3-D tensors with matching first and last dimensions
            int64_t dim0 = (offset < Size) ? (Data[offset] % 4) + 1 : 2;
            offset++;
            int64_t dim2 = (offset < Size) ? (Data[offset] % 4) + 1 : 2;
            offset++;
            for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
                int64_t dim1 = (offset < Size) ? (Data[offset] % 4) + 1 : 2;
                offset++;
                torch::Tensor tensor = torch::randn({dim0, dim1, dim2});
                tensors.push_back(tensor);
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply torch.hstack operation
        // hstack stacks along axis 1 for 2+D tensors, axis 0 for 1-D tensors
        torch::Tensor result = torch::hstack(tensors);
        
        // Test some properties of the result to exercise more code paths
        auto result_numel = result.numel();
        auto result_sizes = result.sizes();
        auto result_dim = result.dim();
        
        // Verify result makes sense
        if (result_numel > 0) {
            // Access element to ensure computation completed
            try {
                volatile float val = result.flatten().index({0}).item<float>();
                (void)val;
            } catch (...) {
                // Silently ignore access errors
            }
        }
        
        // Test with different dtypes
        if (offset < Size && (Data[offset] % 2 == 0)) {
            std::vector<torch::Tensor> int_tensors;
            for (const auto& t : tensors) {
                int_tensors.push_back(t.to(torch::kInt32));
            }
            try {
                torch::Tensor int_result = torch::hstack(int_tensors);
                (void)int_result.numel();
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}