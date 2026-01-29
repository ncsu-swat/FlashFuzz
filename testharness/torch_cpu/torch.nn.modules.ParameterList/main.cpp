#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Ensure we have enough data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a ParameterList
        torch::nn::ParameterList paramList;
        
        // Determine how many parameters to add to the list
        uint8_t numParams = Data[offset++] % 10 + 1; // 1-10 parameters
        
        // Create and add parameters to the list
        for (uint8_t i = 0; i < numParams && offset < Size; ++i) {
            torch::Tensor tensor;
            try {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // Silent catch - create a simple tensor as fallback
                tensor = torch::ones({1});
            }
            
            // Add the tensor as a parameter to the list
            tensor.requires_grad_(true);
            paramList->append(tensor);
        }
        
        // Test various operations on the ParameterList
        if (paramList->size() > 0) {
            // Access parameters by index
            auto firstParam = (*paramList)[0];
            
            // Test iteration over named_parameters
            for (const auto& param : paramList->named_parameters()) {
                auto shape = param.value().sizes();
                (void)shape; // Suppress unused variable warning
            }
            
            // Test append functionality with new tensors
            torch::Tensor tensor3 = torch::zeros({3, 3});
            tensor3.requires_grad_(true);
            paramList->append(tensor3);
            
            // Test named_parameters
            auto namedParams = paramList->named_parameters();
            for (const auto& pair : namedParams) {
                auto name = pair.key();
                auto param = pair.value();
                (void)name;
                (void)param;
            }
            
            // Test parameters
            auto params = paramList->parameters();
            (void)params;
            
            // Test to(device) functionality
            paramList->to(torch::kCPU);
            
            // Test to(dtype) functionality - only float types can have gradients
            if (offset < Size) {
                uint8_t dtypeSelector = Data[offset++] % 4;
                torch::Dtype dtype;
                switch (dtypeSelector) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kFloat16; break;
                    default: dtype = torch::kFloat32; break;
                }
                try {
                    paramList->to(dtype);
                } catch (const std::exception& e) {
                    // Silent catch - some dtype conversions may fail
                }
            }
            
            // Test clone functionality
            torch::nn::ParameterList clonedList;
            for (const auto& param : paramList->parameters()) {
                auto cloned = param.clone();
                cloned.requires_grad_(true);
                clonedList->append(cloned);
            }
            
            // Test zero_grad on parameters
            for (auto& param : paramList->parameters()) {
                if (param.grad().defined()) {
                    param.grad().zero_();
                }
            }
        }
        
        // Test empty ParameterList
        torch::nn::ParameterList emptyList;
        auto emptyParams = emptyList->parameters();
        auto emptySize = emptyList->size();
        (void)emptyParams;
        (void)emptySize;
        
        // Test ParameterList with tensor from fuzzer data
        if (offset < Size) {
            try {
                torch::Tensor largeTensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::nn::ParameterList singleParamList;
                largeTensor.requires_grad_(true);
                singleParamList->append(largeTensor);
                
                // Exercise operations on the single-param list
                auto singleParams = singleParamList->parameters();
                singleParamList->to(torch::kCPU);
            } catch (const std::exception& e) {
                // Silent catch for tensor creation failures
            }
        }
        
        // Test ParameterList construction with initializer
        {
            torch::Tensor t1 = torch::randn({2, 2});
            torch::Tensor t2 = torch::randn({3, 3});
            t1.requires_grad_(true);
            t2.requires_grad_(true);
            
            torch::nn::ParameterList initList;
            initList->append(t1);
            initList->append(t2);
            
            // Verify size
            auto listSize = initList->size();
            (void)listSize;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}