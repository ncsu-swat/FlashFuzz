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
        
        // Need at least some data to proceed
        if (Size < 3) {
            return 0;
        }
        
        // Create condition tensor and convert to boolean
        torch::Tensor condition = fuzzer_utils::createTensor(Data, Size, offset);
        condition = condition.to(torch::kBool);
        
        // Check if we have enough data left for x tensor
        if (offset >= Size) {
            // Test with default tensors if not enough data
            torch::Tensor x = torch::ones({condition.sizes().vec()});
            torch::Tensor y = torch::zeros({condition.sizes().vec()});
            try {
                torch::Tensor result = torch::where(condition, x, y);
            } catch (const std::exception&) {
                // Shape mismatch possible, ignore
            }
            return 0;
        }
        
        // Create x tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for y tensor
        if (offset >= Size) {
            // Test with default tensor for y matching x's dtype
            torch::Tensor y = torch::zeros_like(x);
            try {
                // Broadcast condition to match x/y shapes
                torch::Tensor result = torch::where(condition, x, y);
            } catch (const std::exception&) {
                // Shape/broadcast mismatch possible, ignore
            }
            return 0;
        }
        
        // Create y tensor
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.where operation with broadcasting
        try {
            torch::Tensor result = torch::where(condition, x, y);
        } catch (const std::exception&) {
            // Shape/dtype mismatch possible, ignore
        }
        
        // Test single-argument variant: torch::where(condition) returns nonzero indices
        try {
            std::vector<torch::Tensor> indices = torch::where(condition);
            // indices contains one tensor per dimension
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with x and y having same shape as condition
        try {
            torch::Tensor x_broadcast = x.expand_as(condition);
            torch::Tensor y_broadcast = y.expand_as(condition);
            torch::Tensor result2 = torch::where(condition, x_broadcast, y_broadcast);
        } catch (const std::exception&) {
            // Broadcasting may fail, ignore
        }
        
        // Test with scalar tensors
        try {
            torch::Tensor scalar_x = torch::tensor(1.0);
            torch::Tensor scalar_y = torch::tensor(0.0);
            torch::Tensor result3 = torch::where(condition, scalar_x, scalar_y);
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with different dtypes - ensure x and y can be promoted
        try {
            torch::Tensor x_float = x.to(torch::kFloat32);
            torch::Tensor y_double = y.to(torch::kFloat64);
            torch::Tensor result4 = torch::where(condition, x_float, y_double);
        } catch (const std::exception&) {
            // Type promotion may fail, ignore
        }
        
        // Test with integer tensors
        try {
            torch::Tensor x_int = x.to(torch::kInt32);
            torch::Tensor y_int = y.to(torch::kInt32);
            torch::Tensor result5 = torch::where(condition, x_int, y_int);
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with different condition created from comparison
        try {
            torch::Tensor cond_from_compare = x > y;
            torch::Tensor result6 = torch::where(cond_from_compare, x, y);
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with all-true and all-false conditions
        try {
            torch::Tensor all_true = torch::ones_like(condition, torch::kBool);
            torch::Tensor all_false = torch::zeros_like(condition, torch::kBool);
            torch::Tensor result7 = torch::where(all_true, x, y);
            torch::Tensor result8 = torch::where(all_false, x, y);
        } catch (const std::exception&) {
            // Ignore errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}