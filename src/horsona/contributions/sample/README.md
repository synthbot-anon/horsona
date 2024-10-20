# Sample HorseModule: PoseModule

This folder contains a sample implementation of a HorseModule, which serves as a guide for contributors to understand how to create new HorseModules in the Horsona project.

## Key Components

1. `pose.py`: The main file containing the `PoseModule` implementation.
2. `PoseDescription`: A Pydantic model defining the structure of a pose description.
3. `PoseModule`: A class that extends `HorseModule` and implements the pose generation functionality.

## Requirements for Creating a HorseModule

When creating a new HorseModule, ensure that you follow these requirements:

1. Extend the `HorseModule` class.
2. Make sure every field retained by the HorseModule is a matching constructor argument with the same name.
3. Implement the main functionality as an asynchronous method.
4. Decorate the main functionality method with `@horsefunction`.
5. Set the return type of the main functionality method as `AsyncGenerator` with the return type as the yield result type and `GradContext` as the yield input type.
6. Yield the result instead of returning it.
7. Implement "backpropagation" after the yield statement, which should propagate "errata" back to each variable in the `grad_context`.

## PoseModule Example

The `PoseModule` in `pose.py` demonstrates these requirements:

1. It extends `HorseModule`.
2. The `llm` field is initialized in the constructor with a matching argument name.
3. The `generate_pose` method is asynchronous.
4. The `generate_pose` method is decorated with `@horsefunction`.
5. The return type is `AsyncGenerator[Value[PoseDescription], GradContext]`.
6. The result (`pose_value`) is yielded, not returned.
7. After yielding, it implements backpropagation by checking the `grad_context` and updating both the `context` and `character_info` if necessary.

## Usage

To use the `PoseModule`, you need to initialize it with an `AsyncLLMEngine` instance and then call the `generate_pose` method with the required `character_info` and `context` values. The module will generate a pose description based on the provided information and context.

For more details on implementation and usage, please refer to the `pose.py` file and the corresponding test file in the `tests/contributions` directory.
