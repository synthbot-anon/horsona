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

## Example: PoseModule Test

The `test_pose_module` function in `test_pose_module.py` demonstrates these principles:

1. It sets up a PoseModule with a reasoning LLM.
2. It creates input variables for character info and context.
3. It generates a pose using the module and asserts the output structure and content.
4. It applies two losses:
   - One to change the pose from standing to sitting.
   - Another to correct the character's species from Unicorn to Alicorn.
5. It combines the losses and calls `step()` to propagate the changes back to both the context and character info.
6. Finally, it asserts that:
   - The context has been updated to reflect the sitting pose.
   - The character info has been updated to correct the species to Alicorn.

When contributing a new module, create a similar test file in the `tests/contributions` folder. Make sure to test backpropagation for all relevant input variables.
