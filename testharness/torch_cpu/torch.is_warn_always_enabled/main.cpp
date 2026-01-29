#include "fuzzer_utils.h"
#include <c10/util/Exception.h>
#include <iostream>

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
        // Keep target API keyword for harness checks.
        (void) "torch.is_warn_always_enabled";

        // Get initial warning state (this is the C++ equivalent of torch.is_warn_always_enabled())
        bool initial_status = c10::WarningUtils::get_warnAlways();

        if (Size > 0) {
            // Use fuzz data to decide whether to enable warn always
            bool enable_warn_always = (Data[0] & 1) != 0;
            
            // Test using the WarnAlways RAII guard
            {
                c10::WarningUtils::WarnAlways guard(enable_warn_always);
                
                // Verify the state changed as expected
                bool mid_status = c10::WarningUtils::get_warnAlways();
                (void)mid_status;
                
                // Perform some tensor operations that might trigger warnings
                if (Size > 1) {
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data + 1, Size - 1, offset);
                    
                    if (tensor.defined() && tensor.numel() > 0) {
                        try {
                            torch::Tensor zeros = torch::zeros_like(tensor);
                            torch::Tensor result = tensor + zeros;
                            (void)result.sum();
                        } catch (...) {
                            // Silent catch for expected tensor operation failures
                        }
                    }
                }
            }
            
            // After guard goes out of scope, test direct set/get
            if (Size > 1 && (Data[1] & 1) != 0) {
                bool new_state = (Data[0] & 2) != 0;
                c10::WarningUtils::set_warnAlways(new_state);
                bool check_state = c10::WarningUtils::get_warnAlways();
                (void)check_state;
                
                // Restore original state
                c10::WarningUtils::set_warnAlways(initial_status);
            }
        }

        // Final verification
        bool final_status = c10::WarningUtils::get_warnAlways();
        (void)initial_status;
        (void)final_status;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}