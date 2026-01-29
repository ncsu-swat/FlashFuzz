#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream> // For cerr

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
        
        // Need at least 1 byte for the enabled flag
        if (Size < 1) {
            return 0;
        }
        
        // Parse the enabled flag from the first byte
        bool enabled = Data[0] & 0x01;
        offset++;
        
        // Parse the dtype_selector if there's enough data
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            dtype = fuzzer_utils::parseDataType(dtype_selector);
        }

        // torch.get_autocast_cpu_dtype target API
        auto legacy_dtype = at::autocast::get_autocast_cpu_dtype();

        auto original_dtype = at::autocast::get_autocast_dtype(at::kCPU);
        bool original_enabled = at::autocast::is_autocast_enabled(at::kCPU);

        // Update dtype and enabled flag based on fuzz input
        at::autocast::set_autocast_dtype(at::kCPU, dtype);
        at::autocast::set_autocast_enabled(at::kCPU, enabled);

        // Get the dtype again after setting
        auto new_result_dtype = at::autocast::get_autocast_dtype(at::kCPU);

        // Create a tensor with the autocast dtype
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            torch::Tensor result = tensor.to(new_result_dtype);
            (void)result.sum();
        }

        // Try with different enabled states
        at::autocast::set_autocast_enabled(at::kCPU, true);
        auto dtype1 = at::autocast::get_autocast_dtype(at::kCPU);
        at::autocast::set_autocast_enabled(at::kCPU, false);
        auto dtype2 = at::autocast::get_autocast_dtype(at::kCPU);

        // Flip back to original settings
        at::autocast::set_autocast_dtype(at::kCPU, original_dtype);
        at::autocast::set_autocast_enabled(at::kCPU, original_enabled);

        (void)legacy_dtype;
        (void)new_result_dtype;
        (void)dtype1;
        (void)dtype2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}