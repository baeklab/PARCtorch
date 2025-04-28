# Development Guideline for PARC

### Workflow
First, an issue is created by team leads, which is then assigned to a developer (either voluntary or assignment). As a developer, the expectation is:

0. Create a blank PR linked to the issue. When you do so don't forget to create a new branch.
0. Work on your implementation
0. Run test
0. Repeat 2-3 until you pass all the tests
0. Run Ruff to format your code. Also make sure that you follow the [Google Python style guideline](https://google.github.io/styleguide/pyguide.html)
0. Push your code and ask for a code review
0. Team lead reviews your code. Iterate.
0. Pull main into your branch to make sure there is no conflict.
0. PR approved and merged to the main branch.
