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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create index tensor with appropriate dtype (long)
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert index to long dtype as required by scatter
            index = index.to(torch::kLong);
        } else {
            // If we don't have enough data, create a simple index tensor
            if (input.dim() > 0) {
                index = torch::zeros_like(input, torch::kLong);
            } else {
                // For scalar input, create a simple index
                index = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Create src tensor
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple src tensor
            src = torch::ones_like(input);
        }
        
        // Get a dimension to scatter along
        int64_t dim = 0;
        if (input.dim() > 0 && offset < Size) {
            // Use some bytes from the input to determine the dimension
            dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
        }
        
        // Try different scatter operations
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            // Try scatter
            try {
                torch::Tensor result = input.scatter(dim, index, src);
            } catch (const std::exception&) {
                // Catch and continue - we expect some invalid inputs
            }
            
            // Try scatter with scalar value
            try {
                double value = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::Tensor result = input.scatter(dim, index, value);
            } catch (const std::exception&) {
                // Catch and continue
            }
            
            // Try scatter_ (in-place)
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.scatter_(dim, index, src);
            } catch (const std::exception&) {
                // Catch and continue
            }
            
            // Try scatter_ with scalar value (in-place)
            try {
                double value = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::Tensor input_copy = input.clone();
                input_copy.scatter_(dim, index, value);
            } catch (const std::exception&) {
                // Catch and continue
            }
            
            // Try different reduction modes if we have more data
            if (offset < Size) {
                uint8_t reduce_selector = Data[offset++] % 3;
                std::string reduce_mode;
                
                switch (reduce_selector) {
                    case 0: reduce_mode = "add"; break;
                    case 1: reduce_mode = "multiply"; break;
                    default: reduce_mode = ""; break;
                }
                
                if (!reduce_mode.empty()) {
                    try {
                        torch::Tensor input_copy = input.clone();
                        input_copy.scatter_(dim, index, src, reduce_mode);
                    } catch (const std::exception&) {
                        // Catch and continue
                    }
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
