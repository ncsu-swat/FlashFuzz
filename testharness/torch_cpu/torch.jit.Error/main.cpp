#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        uint8_t error_type = 0;
        if (offset < Size) {
            error_type = Data[offset++] % 5;
        }
        
        std::string error_message = "Test error message";
        if (offset < Size) {
            size_t msg_len = std::min(Size - offset, static_cast<size_t>(20));
            if (msg_len > 0) {
                error_message = std::string(reinterpret_cast<const char*>(Data + offset), msg_len);
                offset += msg_len;
            }
        }
        
        try {
            switch (error_type) {
                case 0:
                    throw torch::Error(error_message);
                case 1:
                    throw torch::Error(error_message);
                case 2: {
                    std::vector<torch::Tensor> tensors = {input_tensor};
                    if (offset < Size) {
                        torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        tensors.push_back(second_tensor);
                    }
                    throw torch::Error(error_message);
                }
                case 3: {
                    std::vector<torch::IValue> values = {input_tensor};
                    if (offset < Size) {
                        torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        values.push_back(second_tensor);
                    }
                    if (offset < Size && Data[offset] % 2 == 0) {
                        values.push_back(error_message);
                    }
                    throw torch::Error(error_message);
                }
                case 4: {
                    torch::Error err(error_message);
                    throw err;
                }
            }
        } catch (const torch::Error& e) {
            std::string what_message = e.what();
            
            if (error_type == 1 || error_type == 2) {
                auto sizes = input_tensor.sizes();
                auto dtype = input_tensor.dtype();
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
