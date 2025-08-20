#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Define a simple C++ class to register with torch.classes
class MyStackClass : public torch::CustomClassHolder {
private:
    std::vector<torch::Tensor> stack_;

public:
    MyStackClass() {}

    void push(torch::Tensor x) {
        stack_.push_back(x);
    }

    torch::Tensor pop() {
        if (stack_.empty()) {
            throw std::runtime_error("Empty stack");
        }
        auto ret = stack_.back();
        stack_.pop_back();
        return ret;
    }

    std::vector<torch::Tensor> getStack() {
        return stack_;
    }

    int64_t size() {
        return stack_.size();
    }
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Register the custom C++ class with torch.classes
        static auto registered = torch::class_<MyStackClass>("_TorchScriptTesting", "MyStackClass")
            .def(torch::init<>())
            .def("push", &MyStackClass::push)
            .def("pop", &MyStackClass::pop)
            .def("getStack", &MyStackClass::getStack)
            .def("size", &MyStackClass::size);
        
        // Create an instance of our custom class
        auto stack_instance = c10::make_intrusive<MyStackClass>();
        
        // Ensure we have enough data to create at least one tensor
        if (Size < 4) {
            return 0;
        }
        
        // Determine how many tensors to create and push to the stack
        uint8_t num_tensors = Data[offset++] % 5 + 1; // 1-5 tensors
        
        // Create and push tensors to the stack
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                stack_instance->push(tensor);
            } catch (const std::exception& e) {
                // Continue with the next tensor if one fails
                continue;
            }
        }
        
        // Test various operations on the stack
        if (stack_instance->size() > 0) {
            // Pop a tensor if the stack is not empty
            auto popped = stack_instance->pop();
            
            // Get the current stack
            auto stack = stack_instance->getStack();
            
            // Push a tensor with different properties
            if (offset + 2 < Size) {
                auto another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                stack_instance->push(another_tensor);
            }
            
            // Test edge case: pop until empty and then try to pop again
            while (stack_instance->size() > 0) {
                stack_instance->pop();
            }
            
            // This should throw an exception which we catch in the outer try-catch
            if (Data[0] % 2 == 0) { // Only do this sometimes to avoid always throwing
                try {
                    stack_instance->pop();
                } catch (const std::runtime_error&) {
                    // Expected exception, continue
                }
            }
        }
        
        // Create a new instance and test with potentially problematic tensors
        auto another_instance = c10::make_intrusive<MyStackClass>();
        
        if (offset < Size) {
            try {
                // Try to create a tensor with extreme values
                auto extreme_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                another_instance->push(extreme_tensor);
                
                // Try operations with the extreme tensor
                auto stack = another_instance->getStack();
                if (another_instance->size() > 0) {
                    another_instance->pop();
                }
            } catch (const std::exception&) {
                // Continue execution
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