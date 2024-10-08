# Pull Request Creation Instructions

Before attempting to merge a branch with main, be sure your updated code runs and passes all unit tests. Run `pytest -m "not model_dependent"`. It is very helpful if you make new unit tests for your code as well. Also make sure you code has been formatted with `black` and `isort`. This should be set up to run automatically on commit by running `pre-commit install`.

When creating your Pull Request, give a detailed description, in the description box, of the changes & additions made before submitting. Once your Pull Request has been submitted, choose an assignee. The assignee selection can be found on the right-hand side of the PR description page (choose someone who is very familiar with the project). 

The assignee will review & run your changes to ensure they are functionable and safe to merge with main. Keep an eye on your pull request until it is merged for any requested changes the assignee needs. These requested changes will need to be made before the code is merged to main. The assignee will review & run the code again once requested changes have been made before merging or requesting more changes, so continue to keep an eye on your pull request. Sending a personal message to the assignee will allow for a swifter response.

# Assignee Instructions

1. Checkout the branch associated with the Pull request you've been assigned on your local device.

2. Visually review the changes made via Pull Request Ticket in GitHub.

3. Run the code on your local device to ensure the code runs and passes all unit tests (`pytest -m "not model_dependent`)

4. If any issues or bugs arise, you will need to request changes via the Pull Request Ticket with a detailed description of what needs to be fixed. These changes will need to be made by the person who created the pull request before you can merge. Sending a personal message to the that person will allow for a swifter response. Once the changes have been made, repeat these instructions from step 1.

5. Once the code associated to the Pull Request is deemed safe, you, the assignee, may merge the Pull Request
