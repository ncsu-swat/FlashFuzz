#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Target API: torch.get_deterministic_debug_mode
// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

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
        (void)get_debug_mode(); // Keep the target API keyword path exercised

        // Try to create a tensor to test with deterministic operations
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test different deterministic debug modes
            if (offset < Size) {
                uint8_t mode_selector = Data[offset++] % 3;
                
                // Set deterministic debug mode based on the input
                switch (mode_selector) {
                    case 0:
                        at::globalContext().setDeterministicAlgorithms(false, false);
                        break;
                    case 1:
                        at::globalContext().setDeterministicAlgorithms(true, true);
                        break;
                    case 2:
                    default:
                        at::globalContext().setDeterministicAlgorithms(true, false);
                        break;
                }
                
                // Get the mode after setting it
                auto new_mode = get_debug_mode();
                (void)new_mode;
                
                // Perform some operation that might be affected by deterministic mode
                if (tensor.defined() && tensor.numel() > 0) {
                    try {
                        auto result = torch::matmul(tensor, tensor);
                        (void)result.sum();
                    } catch (...) {
                        // Ignore exceptions from the operation itself
                    }
                }
            }
            
            // Reset to the original mode
            at::globalContext().setDeterministicAlgorithms(original_mode, original_warn_only);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
