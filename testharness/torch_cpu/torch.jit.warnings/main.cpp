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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with torch.jit operations
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract warning message from remaining data
        std::string warning_message;
        if (offset < Size) {
            size_t message_length = std::min(Size - offset, static_cast<size_t>(100));
            warning_message = std::string(reinterpret_cast<const char*>(Data + offset), message_length);
            offset += message_length;
        } else {
            warning_message = "Test warning message";
        }
        
        // Extract warning verbosity from remaining data
        bool verbosity = false;
        if (offset < Size) {
            verbosity = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Test torch.jit functionality with warnings context
        c10::WarningUtils::set_enabled(verbosity);
        
        // Test warning functionality
        TORCH_WARN(warning_message);
        
        // Test with source location
        if (offset < Size) {
            std::string source_location;
            size_t loc_length = std::min(Size - offset, static_cast<size_t>(50));
            source_location = std::string(reinterpret_cast<const char*>(Data + offset), loc_length);
            TORCH_WARN_ONCE(warning_message);
        }
        
        // Test warning state
        bool current_state = c10::WarningUtils::is_enabled();
        c10::WarningUtils::set_enabled(!current_state);
        TORCH_WARN(warning_message);
        c10::WarningUtils::set_enabled(current_state);
        
        // Test with tensor operations that might trigger warnings
        try {
            auto result = tensor.clone();
            if (tensor.dim() > 0 && tensor.size(0) > 0) {
                result = tensor[0];
            }
            
            // Try operations that might trigger warnings
            if (tensor.numel() > 0) {
                auto mean = tensor.mean();
                auto sum = tensor.sum();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from tensor operations are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
