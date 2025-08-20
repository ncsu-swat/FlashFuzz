#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use with sym_fresh_size
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a name for the symbolic size if there's data left
        std::string name = "dim";
        if (offset + 1 < Size) {
            // Use remaining data to create a name suffix
            uint8_t name_suffix = Data[offset++];
            name += std::to_string(name_suffix);
        }
        
        // Call sym_fresh_size with different parameters based on available data
        if (offset < Size) {
            uint8_t param_selector = Data[offset++];
            
            // Different ways to call sym_fresh_size based on the selector
            switch (param_selector % 3) {
                case 0:
                {
                    // Call with just the name
                    c10::SymInt sym_size = c10::SymInt::fresh_symbol(name);
                    break;
                }
                    
                case 1:
                {
                    // Call with name and min value
                    int64_t min_value = 1;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&min_value, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    c10::SymInt sym_size = c10::SymInt::fresh_symbol(name, min_value);
                    break;
                }
                    
                case 2:
                {
                    // Call with name, min value, and max value
                    int64_t min_val = 1;
                    int64_t max_val = 100;
                    
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&min_val, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&max_val, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    
                    c10::SymInt sym_size = c10::SymInt::fresh_symbol(name, min_val, max_val);
                    break;
                }
            }
        } else {
            // Default call if no more data
            c10::SymInt sym_size = c10::SymInt::fresh_symbol(name);
        }
        
        // Try to use the symbolic size in a tensor creation
        try {
            auto sym_size = c10::SymInt::fresh_symbol(name);
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto new_tensor = torch::empty({sym_size}, options);
        } catch (const std::exception& e) {
            // Catch but don't discard - we want to test if this throws
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}