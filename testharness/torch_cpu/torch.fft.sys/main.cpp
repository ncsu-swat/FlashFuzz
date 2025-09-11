#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use the input tensor for FFT operations
        if (input_tensor.is_complex()) {
            // For complex tensors, try to perform an FFT operation
            if (input_tensor.dim() > 0) {
                int64_t dim = input_tensor.dim() - 1;
                try {
                    auto result = torch::fft::fftn(input_tensor, {}, {dim});
                } catch (...) {
                    // Ignore exceptions from the FFT operation itself
                }
            }
        } else {
            // For real tensors, try to perform a real FFT operation
            if (input_tensor.dim() > 0) {
                int64_t dim = input_tensor.dim() - 1;
                try {
                    auto result = torch::fft::rfftn(input_tensor, {}, {dim});
                } catch (...) {
                    // Ignore exceptions from the FFT operation itself
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
