#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Define a simple class with a classproperty
class TestClass {
private:
    static torch::Tensor _tensor;

public:
    static void set_tensor(const torch::Tensor& t) {
        _tensor = t;
    }

    static torch::Tensor get_tensor() {
        return _tensor;
    }
};

torch::Tensor TestClass::_tensor = torch::empty({0});

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Set the tensor as a class property
        TestClass::set_tensor(input_tensor);
        
        // Get the tensor back using the class property
        torch::Tensor retrieved_tensor = TestClass::get_tensor();
        
        // Verify the tensor was properly stored and retrieved
        if (!torch::equal(input_tensor, retrieved_tensor)) {
            throw std::runtime_error("Retrieved tensor does not match input tensor");
        }
        
        // Try some operations on the retrieved tensor
        if (retrieved_tensor.numel() > 0) {
            torch::Tensor result = retrieved_tensor + 1;
            result = result * 2;
            result = torch::softmax(result, 0);
        }
        
        // Test with different tensor types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Create a new tensor with the selected dtype
            torch::Tensor typed_tensor = input_tensor.to(dtype);
            
            // Set and retrieve using class property
            TestClass::set_tensor(typed_tensor);
            torch::Tensor retrieved_typed = TestClass::get_tensor();
            
            // Verify dtype is preserved
            if (retrieved_typed.dtype() != dtype) {
                throw std::runtime_error("Data type not preserved in class property");
            }
        }
        
        // Test with empty tensor
        TestClass::set_tensor(torch::empty({0}));
        torch::Tensor empty_retrieved = TestClass::get_tensor();
        if (empty_retrieved.numel() != 0) {
            throw std::runtime_error("Empty tensor not properly handled");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
