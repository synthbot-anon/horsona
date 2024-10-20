# Horsona Contributions

Welcome to the Horsona Contributions folder! This is where you, as a contributor, can add your own custom modules to extend the functionality of the Horsona project.

## Purpose

The purpose of this folder is to encourage community contributions and allow developers to create and share their own HorseModules. These modules can add new features, enhance existing functionality, or provide specialized tools for specific use cases within the Horsona framework.

## How to Contribute

To contribute a new module:

1. Create a new folder for your module within this directory.
2. Implement your module following the HorseModule guidelines (see below).
3. Include a README.md file in your module folder explaining its purpose and usage.
4. Add appropriate tests for your module in the `tests/contributions` directory.
5. Run `format.sh` to auto-format your code.
6. Submit a pull request with your new module.

## Example Module

For a detailed example of how to create a HorseModule, please refer to the `sample` folder in this directory. The `sample` folder contains:

- `pose.py`: An implementation of `PoseModule`, which demonstrates the structure and requirements of a HorseModule.
- `README.md`: A detailed explanation of the `PoseModule` and its components.

The `PoseModule` serves as a starting point to understand how to create your own modules.

## Testing Your Module

When contributing a new module, make sure to include appropriate tests in the `tests/contributions` directory. Your tests should cover:

1. The main functionality of your module
2. Proper input handling
3. Expected output structure and content
4. Backpropagation and gradient handling

For an example of how to structure your tests, refer to the `test_pose_module.py` file in the `tests/contributions` directory.

## Questions and Support

If you have any questions about contributing or need support in developing your module, please don't hesitate to open an issue or reach out to the maintainers.

Happy contributing!
