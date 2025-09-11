#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <glob.h>         // For glob functionality
#include <vector>
#include <string>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a pattern string
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input
        torch::Tensor input_tensor;
        if (offset < Size) {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Extract a pattern string from the remaining data
        std::string pattern;
        if (offset < Size) {
            // Use remaining bytes to create a pattern string
            size_t pattern_length = std::min(Size - offset, static_cast<size_t>(32));
            pattern.reserve(pattern_length);
            
            for (size_t i = 0; i < pattern_length; i++) {
                char c = static_cast<char>(Data[offset + i]);
                // Ensure we have some valid glob characters
                if (i % 5 == 0) {
                    pattern.push_back('*');
                } else if (i % 7 == 0) {
                    pattern.push_back('?');
                } else if (c >= 32 && c <= 126) {
                    pattern.push_back(c);
                } else {
                    pattern.push_back('a');
                }
            }
            offset += pattern_length;
        } else {
            pattern = "*";  // Default pattern if no data left
        }
        
        // Try different glob operations using standard glob
        try {
            glob_t glob_result;
            int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
            if (ret == 0) {
                // Process results if needed
                globfree(&glob_result);
            }
        } catch (const std::exception& e) {
            // Standard exceptions are expected and OK
        }
        
        // Try with different flags if we have more data
        if (offset < Size) {
            bool use_nosort = Data[offset++] % 2 == 0;
            try {
                glob_t glob_result;
                int flags = GLOB_TILDE;
                if (use_nosort) {
                    flags |= GLOB_NOSORT;
                }
                int ret = glob(pattern.c_str(), flags, NULL, &glob_result);
                if (ret == 0) {
                    globfree(&glob_result);
                }
            } catch (const std::exception& e) {
                // Standard exceptions are expected and OK
            }
        }
        
        // Try with different path combinations
        if (offset + 1 < Size) {
            std::string root_dir;
            size_t root_dir_length = std::min(Size - offset, static_cast<size_t>(16));
            root_dir.reserve(root_dir_length);
            
            for (size_t i = 0; i < root_dir_length; i++) {
                char c = static_cast<char>(Data[offset + i]);
                if (c >= 32 && c <= 126 && c != '\0') {
                    root_dir.push_back(c);
                } else {
                    root_dir.push_back('/');
                }
            }
            offset += root_dir_length;
            
            std::string full_pattern = root_dir + "/" + pattern;
            
            try {
                glob_t glob_result;
                int ret = glob(full_pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
                if (ret == 0) {
                    globfree(&glob_result);
                }
            } catch (const std::exception& e) {
                // Standard exceptions are expected and OK
            }
        }
        
        // Use the tensor in some way to avoid unused variable warning
        if (input_tensor.defined()) {
            auto sum = torch::sum(input_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
