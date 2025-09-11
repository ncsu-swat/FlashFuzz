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
        
        // Call the get_autocast_cpu_dtype function
        torch::ScalarType result_dtype = torch::autocast::get_autocast_cpu_dtype();
        
        // Try setting the autocast CPU dtype
        torch::autocast::set_autocast_cpu_dtype(dtype);
        
        // Try enabling/disabling autocast
        torch::autocast::AutocastCPU guard(enabled);
        
        // Get the dtype again after setting
        torch::ScalarType new_result_dtype = torch::autocast::get_autocast_cpu_dtype();
        
        // Create a tensor with the autocast dtype
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try using the tensor with autocast
            torch::Tensor result;
            {
                torch::autocast::AutocastCPU inner_guard(enabled);
                result = tensor.to(torch::autocast::get_autocast_cpu_dtype());
            }
        }
        
        // Try with different enabled states
        {
            torch::autocast::AutocastCPU guard1(true);
            torch::ScalarType dtype1 = torch::autocast::get_autocast_cpu_dtype();
        }
        
        {
            torch::autocast::AutocastCPU guard2(false);
            torch::ScalarType dtype2 = torch::autocast::get_autocast_cpu_dtype();
        }
        
        // Try with nested autocast contexts
        {
            torch::autocast::AutocastCPU outer_guard(true);
            torch::ScalarType outer_dtype = torch::autocast::get_autocast_cpu_dtype();
            
            {
                torch::autocast::AutocastCPU inner_guard(false);
                torch::ScalarType inner_dtype = torch::autocast::get_autocast_cpu_dtype();
            }
            
            torch::ScalarType after_inner_dtype = torch::autocast::get_autocast_cpu_dtype();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
