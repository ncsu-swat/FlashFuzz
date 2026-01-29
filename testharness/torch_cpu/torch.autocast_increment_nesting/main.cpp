#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream> // For cerr
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Check if we have enough data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use in the context
        torch::Tensor tensor;
        try {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (...) {
            tensor = torch::ones({2, 2});
        }
        
        // Ensure tensor is valid for operations
        if (!tensor.defined() || tensor.numel() == 0) {
            tensor = torch::ones({2, 2});
        }
        
        // Extract a boolean from the data to determine if we should use enabled=true/false
        bool enabled = false;
        if (offset < Size) {
            enabled = static_cast<bool>(Data[offset++] & 0x1);
        }
        
        // Test autocast_increment_nesting on CPU
        // Set autocast state for CPU using the new API
        at::autocast::set_autocast_enabled(at::kCPU, enabled);
        
        // Call autocast_increment_nesting - this is the main API under test
        int nesting_before = at::autocast::increment_nesting();
        
        // Perform some operations inside the autocast context
        try {
            torch::Tensor result = tensor + 1.0f;
            (void)result.sum().item<float>(); // ensure the result is materialized
        } catch (...) {
            // Shape/dtype mismatches are expected
        }
        
        // Decrement nesting to balance the increment
        int nesting_after = at::autocast::decrement_nesting();
        (void)nesting_before;
        (void)nesting_after;
        
        // Try with different enabled values
        at::autocast::set_autocast_enabled(at::kCPU, !enabled);
        at::autocast::increment_nesting();
        try {
            torch::Tensor another_result = tensor * 2.0f;
            (void)another_result.sum().item<float>();
        } catch (...) {
            // Expected failures
        }
        at::autocast::decrement_nesting();
        
        // Try nested calls to test nesting level tracking
        at::autocast::increment_nesting();
        at::autocast::increment_nesting();
        try {
            torch::Tensor nested_result = tensor.pow(2);
            (void)nested_result.sum().item<float>();
        } catch (...) {
            // Expected failures
        }
        at::autocast::decrement_nesting();
        at::autocast::decrement_nesting();
        
        // Try with different tensor operations
        try {
            if (offset + 1 < Size) {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (another_tensor.defined() && another_tensor.numel() > 0) {
                    at::autocast::increment_nesting();
                    // Use matmul which is more likely to trigger autocast behavior
                    auto t1 = tensor.to(torch::kFloat32).reshape({-1, 1});
                    auto t2 = another_tensor.to(torch::kFloat32).reshape({1, -1});
                    torch::Tensor mixed_result = torch::matmul(t1, t2);
                    (void)mixed_result.sum().item<float>();
                    at::autocast::decrement_nesting();
                }
            }
        } catch (...) {
            // Shape mismatches are expected
        }
        
        // Try with varying nesting levels
        uint8_t nesting_level = 1;
        if (offset < Size) {
            nesting_level = (Data[offset++] % 5) + 1; // 1-5 levels
        }
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            at::autocast::increment_nesting();
        }
        
        try {
            torch::Tensor deep_nested_result = tensor.sin().cos().exp();
            (void)deep_nested_result.sum().item<float>();
        } catch (...) {
            // Expected failures
        }
        
        for (uint8_t i = 0; i < nesting_level; i++) {
            at::autocast::decrement_nesting();
        }
        
        // Test autocast dtype setting (exercises related APIs)
        try {
            at::autocast::set_autocast_dtype(at::kCPU, torch::kBFloat16);
            at::autocast::increment_nesting();
            torch::Tensor dtype_test = tensor.to(torch::kFloat32) + tensor.to(torch::kFloat32);
            (void)dtype_test.sum().item<float>();
            at::autocast::decrement_nesting();
            at::autocast::set_autocast_dtype(at::kCPU, torch::kFloat16);
        } catch (...) {
            // BFloat16 might not be supported on all systems
        }
        
        // Reset autocast state to clean up
        at::autocast::set_autocast_enabled(at::kCPU, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}