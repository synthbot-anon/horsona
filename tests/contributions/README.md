# Contributions Test Folder

This folder serves as a sample for contributors, demonstrating how to create test cases for new HorseModules. The example provided here is for the PoseModule, which is a subclass of HorseModule.

## Creating a Test Case for a HorseModule

When creating a test case for a new HorseModule, follow these guidelines:

1. Import necessary modules and classes:
   - Import pytest and mark your test function with `@pytest.mark.asyncio` for asynchronous testing.
   - Import the required classes from horsona, including the module you're testing.

2. Set up the test environment:
   - Create an instance of your module, passing any required dependencies (e.g., an AsyncLLMEngine).
   - Initialize any input variables needed for your module using the `Value` class.

3. Test the main functionality:
   - Call the main methods of your module with the prepared inputs.
   - Assert that the output is of the expected type and contains the expected data.
   - Check if the generated output matches the expected behavior based on the inputs.

4. Test backpropagation:
   - Apply losses to the output using `apply_loss()`.
   - Combine the losses if there are multiple.
   - Call the `step()` function on the combined loss object, passing all variables that should be updated.
   - Assert that the changes have been propagated back to the input variables as expected.

5. Test saving and restoring the module:
   - Create an instance of your module with specific parameters.
   - Save the module's state using `state_dict()`.
   - Create a new instance of your module using `load_state_dict()` with the saved state.
   - Assert that the restored module has the same properties as the original.

## Example: PoseModule Test

The `test_pose_module.py` file demonstrates these principles:

1. `test_pose_module` function:
   - It sets up a PoseModule with a reasoning LLM.
   - It creates input variables for character info and context.
   - It generates a pose using the module and asserts the output structure and content.
   - It applies two losses:
     - One to change the pose from standing to sitting.
     - Another to correct the character's species from Unicorn to Alicorn.
   - It combines the losses and calls `step()` to propagate the changes back to both the context and character info.
   - Finally, it asserts that:
     - The context has been updated to reflect the sitting pose.
     - The character info has been updated to correct the species to Alicorn.

2. `test_pose_module_state_dict` function:
   - It creates an original PoseModule with a specific name.
   - It saves the state dictionary of the original module.
   - It creates a new PoseModule instance by loading the saved state dictionary.
   - It asserts that the restored module has the same name as the original.

When contributing a new module, create a similar test file in the `tests/contributions` folder. Make sure to test the main functionality, backpropagation for all relevant input variables, and saving/restoring the module state. If your module defines any new HorseVariables, include tests for saving and restoring those variables as well.
