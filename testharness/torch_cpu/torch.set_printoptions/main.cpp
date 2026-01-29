#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>
#include <cstring>

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
        
        // Need at least a few bytes for the options
        if (Size < 4) {
            return 0;
        }
        
        // Extract parameters for print options with reasonable bounds
        int precision = 4;
        int64_t threshold = 1000;
        int edgeitems = 3;
        int linewidth = 80;
        
        // Parse precision (number of digits to print) - keep it reasonable
        if (offset + 1 <= Size) {
            precision = static_cast<int>(Data[offset] % 20) + 1;  // 1-20
            offset += 1;
        }
        
        // Parse threshold (total number of elements before switching to summary)
        if (offset + 2 <= Size) {
            uint16_t val;
            std::memcpy(&val, Data + offset, sizeof(uint16_t));
            threshold = static_cast<int64_t>(val % 10000) + 1;  // 1-10000
            offset += 2;
        }
        
        // Parse edgeitems (number of edge items to show)
        if (offset + 1 <= Size) {
            edgeitems = static_cast<int>(Data[offset] % 10) + 1;  // 1-10
            offset += 1;
        }
        
        // Parse linewidth (number of characters per line)
        if (offset + 1 <= Size) {
            linewidth = static_cast<int>(Data[offset] % 200) + 20;  // 20-219
            offset += 1;
        }

        // Create a tensor to test printing
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we've consumed all input data
            tensor = torch::randn({3, 4, 5});
        }

        // Inner try-catch for expected failures during print options setting
        try
        {
            // In PyTorch C++, print options are typically set through torch::PrintOptions
            // or the stream directly. The actual C++ API uses c10::print_tensor or 
            // streaming operators with options.
            
            // Set print options using the C10 API
            c10::TensorPrintOptions options;
            options.precision = precision;
            options.threshold = threshold;
            options.edgeitems = edgeitems;
            options.linewidth = linewidth;
            c10::set_print_options(options);
            
            // Test printing the tensor to a string stream
            std::stringstream ss;
            ss << tensor;
            
            // Also test with different tensor types
            torch::Tensor int_tensor = torch::randint(0, 100, {4, 4}, torch::kInt32);
            std::stringstream ss2;
            ss2 << int_tensor;
            
            // Test with a complex tensor if supported
            try {
                torch::Tensor complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
                std::stringstream ss3;
                ss3 << complex_tensor;
            } catch (...) {
                // Complex might not be supported, ignore
            }
            
            // Reset to default options to avoid affecting other tests
            c10::TensorPrintOptions default_options;
            c10::set_print_options(default_options);
        }
        catch (...)
        {
            // Silently catch expected failures (e.g., invalid option combinations)
            // Reset to defaults on failure
            try {
                c10::TensorPrintOptions default_options;
                c10::set_print_options(default_options);
            } catch (...) {}
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}