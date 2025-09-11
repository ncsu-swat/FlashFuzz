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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 bytes left for padding values
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract padding values from the remaining data
        int64_t left = static_cast<int64_t>(Data[offset++]);
        int64_t right = static_cast<int64_t>(Data[offset++]);
        int64_t top = static_cast<int64_t>(Data[offset++]);
        int64_t bottom = static_cast<int64_t>(Data[offset++]);
        
        // Create ZeroPad2d module
        // Try different ways to specify padding
        if (offset % 3 == 0) {
            // Single value for all sides
            int64_t padding = left;
            auto zeropad = torch::nn::ZeroPad2d(padding);
            auto output = zeropad->forward(input_tensor);
        } 
        else if (offset % 3 == 1) {
            // Vector of 4 values (left, right, top, bottom)
            std::vector<int64_t> padding = {left, right, top, bottom};
            auto zeropad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(padding));
            auto output = zeropad->forward(input_tensor);
        }
        else {
            // Vector of 4 values
            std::vector<int64_t> padding = {left, right, top, bottom};
            auto zeropad = torch::nn::ZeroPad2d(padding);
            auto output = zeropad->forward(input_tensor);
        }
        
        // Try functional version as well
        std::vector<int64_t> pad_vec = {left, right, top, bottom};
        auto functional_output = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions(pad_vec).mode(torch::kConstant).value(0.0)
        );
        
        // Try with negative padding values
        if (offset < Size) {
            int64_t neg_padding = -static_cast<int64_t>(Data[offset++] % 5);
            try {
                auto neg_zeropad = torch::nn::ZeroPad2d(neg_padding);
                auto neg_output = neg_zeropad->forward(input_tensor);
            } catch (...) {
                // Expected to potentially fail with negative padding
            }
        }
        
        // Try with very large padding values
        if (offset < Size) {
            int64_t large_padding = static_cast<int64_t>(Data[offset++]) * 1000;
            try {
                auto large_zeropad = torch::nn::ZeroPad2d(large_padding);
                auto large_output = large_zeropad->forward(input_tensor);
            } catch (...) {
                // Expected to potentially fail with very large padding
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
