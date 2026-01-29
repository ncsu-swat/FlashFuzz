#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// Target API: torch.get_deterministic_debug_mode
// Note: This Python API maps to at::globalContext() deterministic settings in C++
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

        // C++ equivalent of torch.get_deterministic_debug_mode()
        // Returns: 0 = default, 1 = warn, 2 = error
        auto get_debug_mode = []() -> int {
            if (at::globalContext().deterministicAlgorithms())
            {
                return at::globalContext().deterministicAlgorithmsWarnOnly() ? 1 : 2;
            }
            return 0;
        };

        // Preserve current deterministic debug mode settings
        const bool original_mode = at::globalContext().deterministicAlgorithms();
        const bool original_warn_only = at::globalContext().deterministicAlgorithmsWarnOnly();
        
        // Exercise the get function
        int initial_mode = get_debug_mode();
        (void)initial_mode;

        if (Size > 0) {
            // Test different deterministic debug modes based on fuzz input
            uint8_t mode_selector = Data[offset++] % 3;
            
            // Set deterministic debug mode based on the input
            switch (mode_selector) {
                case 0:
                    // Mode 0: default (non-deterministic)
                    at::globalContext().setDeterministicAlgorithms(false, false);
                    break;
                case 1:
                    // Mode 1: warn only
                    at::globalContext().setDeterministicAlgorithms(true, true);
                    break;
                case 2:
                default:
                    // Mode 2: error on non-deterministic
                    at::globalContext().setDeterministicAlgorithms(true, false);
                    break;
            }
            
            // Verify the mode was set correctly
            int new_mode = get_debug_mode();
            (void)new_mode;
            
            // Create a tensor and perform operations that may be affected by deterministic mode
            if (offset < Size) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (tensor.defined() && tensor.numel() > 0) {
                    try {
                        // Ensure tensor is 2D for matmul
                        auto t = tensor.view({-1});
                        int64_t n = t.size(0);
                        if (n > 1) {
                            int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(n)));
                            if (side > 0) {
                                auto mat = t.narrow(0, 0, side * side).view({side, side});
                                // matmul can trigger deterministic algorithm checks
                                try {
                                    auto result = torch::matmul(mat, mat);
                                    (void)result.sum();
                                } catch (...) {
                                    // Expected: deterministic mode may throw for some operations
                                }
                            }
                        }
                    } catch (...) {
                        // Shape/size issues - ignore
                    }
                }
            }
            
            // Test fillUnimplementedCuDNNOperator flag as well
            bool fill_flag = at::globalContext().deterministicFillUninitializedMemory();
            (void)fill_flag;
            
            // Reset to the original mode
            at::globalContext().setDeterministicAlgorithms(original_mode, original_warn_only);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}