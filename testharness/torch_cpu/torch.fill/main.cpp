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
        
        // Need at least a few bytes for the tensor and fill value
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to fill
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a fill value from the remaining data
        if (offset + sizeof(float) <= Size) {
            float fill_value;
            std::memcpy(&fill_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Apply the fill operation
            tensor.fill_(fill_value);
        } else if (offset < Size) {
            // If we don't have enough bytes for a float, use whatever is left as a byte
            uint8_t byte_value = Data[offset++];
            tensor.fill_(static_cast<float>(byte_value));
        } else {
            // If we've consumed all data, use a default value
            tensor.fill_(0.0f);
        }
        
        // Try filling with a scalar tensor
        if (offset + sizeof(float) <= Size) {
            float scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Create a scalar tensor and use it for filling
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            tensor.fill_(scalar_tensor);
        }
        
        // Try filling with different types
        if (offset < Size) {
            uint8_t type_selector = Data[offset++];
            
            switch (type_selector % 4) {
                case 0:
                    tensor.fill_(42);  // int
                    break;
                case 1:
                    tensor.fill_(3.14);  // double
                    break;
                case 2:
                    tensor.fill_(true);  // bool
                    break;
                case 3: {
                    // Try with a potentially incompatible tensor
                    if (offset < Size) {
                        torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        try {
                            tensor.fill_(another_tensor);
                        } catch (...) {
                            // Expected to fail in some cases, but we want to test it
                        }
                    }
                    break;
                }
            }
        }
        
        // Try filling with extreme values
        if (offset < Size) {
            uint8_t extreme_selector = Data[offset++];
            
            switch (extreme_selector % 4) {
                case 0:
                    tensor.fill_(std::numeric_limits<float>::infinity());
                    break;
                case 1:
                    tensor.fill_(-std::numeric_limits<float>::infinity());
                    break;
                case 2:
                    tensor.fill_(std::numeric_limits<float>::quiet_NaN());
                    break;
                case 3:
                    tensor.fill_(std::numeric_limits<float>::min());
                    break;
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
