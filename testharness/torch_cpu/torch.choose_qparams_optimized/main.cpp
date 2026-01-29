#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <tuple>          // For std::get

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
        
        // Need at least a few bytes for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }
        
        // Create input tensor - must be float type for this API
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float type (API requires float)
        input_tensor = input_tensor.to(torch::kFloat32).contiguous();
        
        // Ensure tensor has at least 1 element
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Extract numel parameter - must be <= input_tensor.numel() and > 0
        uint8_t numel_byte = 0;
        if (offset < Size) {
            numel_byte = Data[offset++];
        }
        // Scale numel to be within valid range [1, input_tensor.numel()]
        int64_t max_numel = input_tensor.numel();
        int64_t numel = (numel_byte % max_numel) + 1;
        
        // Extract n_bins parameter (must be > 0)
        uint8_t n_bins_byte = 0;
        if (offset < Size) {
            n_bins_byte = Data[offset++];
        }
        int64_t n_bins = (n_bins_byte % 255) + 1; // Range: 1-255
        
        // Extract ratio parameter (0.0 to 1.0 range makes most sense)
        uint8_t ratio_byte = 0;
        if (offset < Size) {
            ratio_byte = Data[offset++];
        }
        double ratio = ratio_byte / 255.0; // Range: 0.0-1.0
        
        // Extract bit_width parameter (common values: 4, 8)
        uint8_t bit_width_byte = 0;
        if (offset < Size) {
            bit_width_byte = Data[offset++];
        }
        // Common quantization bit widths are 2, 4, 8
        int64_t bit_widths[] = {2, 4, 8};
        int64_t bit_width = bit_widths[bit_width_byte % 3];
        
        // Apply the operation
        // Returns (xmax_tensor, xmin_tensor) as per implementation
        auto result = torch::choose_qparams_optimized(
            input_tensor,
            numel,
            n_bins,
            ratio,
            bit_width
        );
        
        // Access the result to ensure it's computed
        auto xmax_tensor = std::get<0>(result);
        auto xmin_tensor = std::get<1>(result);
        
        // Verify results are valid tensors
        if (xmax_tensor.defined() && xmin_tensor.defined()) {
            // Access values to ensure computation completed
            float xmax = xmax_tensor.item<float>();
            float xmin = xmin_tensor.item<float>();
            (void)xmax;
            (void)xmin;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}