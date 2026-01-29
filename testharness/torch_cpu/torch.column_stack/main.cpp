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
        if (Size < 8) {
            return 0;
        }
        
        // Determine number of tensors to create (2-4, need at least 2 for meaningful stacking)
        uint8_t num_tensors = (Data[offset] % 3) + 2;
        offset++;
        
        // Determine common number of rows (1-8)
        int64_t num_rows = (Data[offset] % 8) + 1;
        offset++;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors with compatible shapes for column_stack
        // column_stack treats 1D tensors as column vectors
        // All tensors must have same number of rows (or be 1D with same length)
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            if (offset >= Size) break;
            
            // Decide tensor type: 1D or 2D
            bool is_1d = (Data[offset] % 2) == 0;
            offset++;
            
            if (offset >= Size) break;
            
            // Determine number of columns for 2D tensor (1-4)
            int64_t num_cols = (Data[offset] % 4) + 1;
            offset++;
            
            // Determine dtype
            if (offset >= Size) break;
            torch::ScalarType dtype;
            switch (Data[offset] % 4) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                default: dtype = torch::kInt64; break;
            }
            offset++;
            
            torch::Tensor tensor;
            if (is_1d) {
                // 1D tensor with length = num_rows (will be treated as column)
                tensor = torch::randn({num_rows}, torch::TensorOptions().dtype(torch::kFloat32));
            } else {
                // 2D tensor with shape [num_rows, num_cols]
                tensor = torch::randn({num_rows, num_cols}, torch::TensorOptions().dtype(torch::kFloat32));
            }
            
            // Convert to target dtype if numeric
            if (dtype == torch::kInt32 || dtype == torch::kInt64) {
                tensor = tensor.to(dtype);
            }
            
            tensors.push_back(tensor);
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply column_stack operation
        torch::Tensor result = torch::column_stack(tensors);
        
        // Verify result shape - should have num_rows rows
        if (result.dim() >= 1 && result.size(0) != num_rows) {
            std::cerr << "Unexpected result shape" << std::endl;
        }
        
        // Perform operations on the result to ensure it's used
        auto sum = result.sum();
        auto mean = result.mean();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}