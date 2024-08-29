# Contributing to Crocodile

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to the Crocodile Python library. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### 1. Reporting Bugs

If you find a bug, please report it by opening a [GitHub issue](https://github.com/your-repo/crocodile/issues). Include as much detail as possible to help us resolve the issue faster:

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as much detail as possible.
- **Provide specific examples** to demonstrate the issue. Include code snippets, screenshots, or links to relevant resources.
- **Describe the behavior you observed after following the steps** and explain what you expected to see instead.

### 2. Suggesting Enhancements

Enhancement suggestions are also welcome! To suggest an enhancement:

- Open a [GitHub issue](https://github.com/your-repo/crocodile/issues) and describe:
  - The feature you would like to see.
  - The problem it would solve.
  - Why you think it would be beneficial for others.
  - Any alternative solutions or features you've considered.
  
### 3. Contributing Code

We appreciate contributions to the codebase. Before you start coding:

- **Check existing issues** to see if someone else is already working on your idea or problem.
- If you plan to work on a big feature, consider opening an issue first to discuss it.
- **Fork the repository** and create your branch:
  
  ```bash
  git checkout -b feature-name
  ```

- Make sure your code adheres to the existing style of the project:
  - Follow PEP 8 guidelines for Python code.
  - Write clear and descriptive commit messages.

- **Write tests** for your changes.
- **Document your changes**. If your code includes new features or changes existing functionality, update the documentation accordingly.
- **Run tests** to ensure your code works as expected:
  
  ```bash
  pytest
  ```

- **Submit a pull request**:
  - Push your branch to your forked repository:
    
    ```bash
    git push origin feature-name
    ```
  
  - Open a [pull request](https://github.com/your-repo/crocodile/pulls) and describe your changes.
  - Ensure that your pull request passes all CI checks.

### 4. Improving Documentation

If you find the documentation lacking or unclear, feel free to improve it. To contribute to the documentation:

- **Fork the repository** and create a branch for your changes.
- Make your changes to the relevant `.md` files or in the `docs` folder.
- **Submit a pull request** following the steps above.

### 5. Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, inclusive, and patient with others in the community.

## Style Guides

### Python Style Guide

- Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
- Use [Black](https://black.readthedocs.io/en/stable/) for code formatting (we recommend setting it up in your editor).
- Ensure that your code is properly linted and formatted before submitting.

### Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line of the commit message to 50 characters or less.
- Include relevant issue numbers in the commit message.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.