#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the options
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for set_printoptions
        int64_t precision = 0;
        int64_t threshold = 0;
        int64_t edgeitems = 0;
        int64_t linewidth = 0;
        
        // Parse precision (number of digits to print)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&precision, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse threshold (total number of elements before switching to summary)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse edgeitems (number of edge items to show)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&edgeitems, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse linewidth (number of characters per line)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&linewidth, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create a tensor to test printing
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we've consumed all input data
            tensor = torch::randn({3, 4, 5});
        }
        
        // Apply set_printoptions with the parsed parameters
        c10::TensorPrintOptions options;
        options.precision = precision;
        options.threshold = threshold;
        options.edgeitems = edgeitems;
        options.linewidth = linewidth;
        c10::set_print_options(options);
        
        // Test printing the tensor to a string
        std::stringstream ss;
        ss << tensor;
        
        // Reset to default options
        c10::TensorPrintOptions default_options;
        c10::set_print_options(default_options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}