Working with the Codebase
=========================

Version control, Git, and GitHub
""""""""""""""""""""""""""""""""

To the neophyte, working with Git is one of the more daunting aspects of contributing to open source projects.
It can very quickly become overwhelming, but sticking to the guidelines below will help keep the process straightforward and mostly trouble free. As always, if you are having difficulties please feel free to ask for help.
The code is hosted on `GitHub <https://github.com/fzhu2e/cfr>`_. To contribute you will need to `sign up for a (free) GitHub account <https://github.com/signup/free>`_. `Git <https://git-scm.com/>`_ is the industry standard for version control to allow many people to work together on the project, keep track of issues, manage the project, and much more.

Some great resources for learning Git:
  * the `GitHub help pages <https://help.github.com/>`_
  * the `NumPy documentation <https://numpy.org/doc/stable/dev/index.html>`_
  * Matthew Brett’s `Pydagogue <https://matthew-brett.github.io/pydagogue/>`_

GitHub has `instructions <https://help.github.com/set-up-git-redirect>`_ for installing git, setting up your SSH key, and configuring git. All these steps need to be completed before you can work seamlessly between your local repository and GitHub.

Forking
"""""""
You will need your own fork to work on the code.
Go to the `cfr` repository and hit the Fork button.
You will then want to clone your fork (i.e. download all the code to your local machine so you can edit it locally).
At the command line, this would like something like::
    git clone https://github.com/your-user-name/cfr.git cfr-fork
    cd cfr-fork
    git remote add upstream https://github.com/fzhu2e/cfr.git

This creates the directory `cfr-fork` and connects your repository to the upstream (main project) `cfr` repository. 
However, most Git first-timers may find it easier to do so through the Github web interface or desktop app (where there is a proverbial "button" for that).

Creating a development environment
""""""""""""""""""""""""""""""""""
We recommend developing in the same `conda` environment in which you installed `cfr`.

Creating a branch
"""""""""""""""""
You want your master branch to reflect only production*ready code, so create a feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the `shiny-new-feature` branch.
Keep any changes in this branch specific to one bug or feature, so it is clear what the branch brings to `cfr`.
You can have many `shiny-new-features` and switch in between them using the `git checkout` command.
When creating this branch, make sure your master branch is up-to-date with the latest upstream master version. To update your local master branch, you can do::

    git checkout master
    git pull upstream master --ff-only

When you want to update the feature branch with changes in master after you created the branch, check the section on updating a pull request.


Contributing new or updating existing functionalities
""""""""""""""""""""""""""""""""""""""""""""""""""""""

- Open an issue on GitHub (See above)
- Integrate the new functionality or update the existing functionality. Make sure you:
  - Re-use as many of `cfr`'s existing utilities as you can, introducing new package dependencies only as necessary.
  - Create a docstring for your new function, describing arguments and returned variables, and showing an example of use. (Use an existing docstring for inspiration).
- Test the functionality within a self-contained Jupyter notebook. If the functionality is related to the reconstruction method, **it is required to conduct a pseudoproxy experiment (PPE) using the `pseudoPAGES2k <https://fzhu2e.github.io/cfr/ug-pp2k.html>`_ dataset** so that the result is comparable to existing ones.


Stylistic considerations
""""""""""""""""""""""""
Guido van Rossum's great insight is that code is read far more often than it is written, so it is important for the code to be of a somewhat uniform style, so that people can read and understand it with relative ease. `cfr` strives to use fairly consistent notation, including:

- "CamelCase" or "all-Caps" conventions for classes, e.g., "MyNewClass" or "MYNEWCLASS"
- lowercase letters with underscores for methods and variables, e.g., "my_new_method" and "my_new_variable"

In general, one should follow the existing style of the codebase as much as possible.

Conventions
"""""""""""
- `cfr` functions generally assume that time values are provided in increasing order.

Committing your code
""""""""""""""""""""
Once you've made changes, you can see them by typing::

    git status

If you created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

Typing `git status` again should give something like::

    On branch shiny-new-feature
    modified:   /relative/path/to/file-you-added.py

Finally, commit your changes to your local repository with an explanatory message. The message need not be encyclopedic, but it should say what you did, what GitHub issue it refers to, and what part of the code it is expected to affect.
The preferred style is:

- A subject line with < 80 chars.
- One blank line.
- Optionally, a commit message body.

Now you can commit your changes in your local repository::

    git commit -m 'type your message here'

Pushing your changes
""""""""""""""""""""

When you want your changes to appear publicly on your GitHub page, push your forked feature branch's commits::

    git push origin shiny-new-feature

Here `origin` is the default name given to your remote repository on GitHub. You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something like::

    origin  git@github.com:yourname/cfr.git (fetch)
    origin  git@github.com:yourname/cfr.git (push)
    upstream  git://github.com/fzhu2e/cfr.git (fetch)
    upstream  git://github.com/fzhu2e/cfr.git (push)

Now your code is on GitHub, but it is not yet a part of the `cfr` project. For that to happen, a pull request needs to be submitted on GitHub.

Filing a Pull Request
"""""""""""""""""""""
When you're ready to ask for a code review, file a pull request. But before you do, please double-check that you have followed all the guidelines outlined in this document regarding code style, tests, performance tests, and documentation. You should also double-check your branch changes against the branch it was based on:

- Navigate to your repository on GitHub
- Click on Branches
- Click on the Compare button for your feature branch
- Select the base and compare branches, if necessary. This will be *Development* and *shiny-new-feature*, respectively.

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be reviewed by a project's owners/developers and eventually merged into the master version. This pull request and its associated changes will eventually be committed to the master branch and available in the next release. To submit a pull request:

- Navigate to your repository on GitHub
- Click on the Pull Request button
- You can then click on Commits and Files Changed to make sure everything looks okay one last time
- Write a description of your changes in the Preview Discussion tab
- Click Send Pull Request.

This request then goes to the repository maintainers, and they will review the code.

Updating your pull request
""""""""""""""""""""""""""

Based on the review you get on your pull request, you will probably need to make some changes to the code. In that case, you can make them in your branch, add a new commit to that branch, push it to GitHub, and the pull request will be automatically updated. Pushing them to GitHub again is done by:
git push origin shiny-new-feature
This will automatically update your pull request with the latest code and restart the Continuous Integration tests (which is why it is important to provide a test for your code).
Another reason you might need to update your pull request is to solve conflicts with changes that have been merged into the master branch since you opened your pull request.
To do this, you need to `merge upstream master` in your branch::

    git checkout shiny-new-feature
    git fetch upstream
    git merge upstream/master

If there are no conflicts (or they could be fixed automatically), a file with a default commit message will open, and you can simply save and quit this file.
If there are merge conflicts, you need to solve those conflicts. See `this example <https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_ for an explanation on how to do this. Once the conflicts are merged and the files where the conflicts were solved are added, you can run git commit to save those fixes.
If you have uncommitted changes at the moment you want to update the branch with master, you will need to stash them prior to updating (see the stash docs). This will effectively store your changes and they can be reapplied after updating.
After the feature branch has been updated locally, you can now update your pull request by pushing to the branch on GitHub::

  git push origin shiny-new-feature

Delete your merged branch (optional)
""""""""""""""""""""""""""""""""""""

Once your feature branch is accepted into upstream, you'll probably want to get rid of the branch. First, merge upstream master into your branch so git knows it is safe to delete your branch::

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can do::

    git branch -d shiny-new-feature

Make sure you use a lowercase `-d`, or else git won't warn you if your feature branch has not actually been merged.
The branch will still exist on GitHub, so to delete it there do::

    git push origin --delete shiny-new-feature

Tips for a successful pull request
""""""""""""""""""""""""""""""""""
If you have made it to the "Review your code" phase, one of the core contributors will take a look. Please note however that response time will be variable (e.g. don't try the week before AGU).
To improve the chances of your pull request being reviewed, you should:

- Reference an open issue for non*trivial changes to clarify the PR's purpose
- Ensure you have appropriate tests. These should be the first part of any PR
- Keep your pull requests as simple as possible. Larger PRs take longer to review
- If you need to add on to what you submitted, keep updating your original pull request, either by request or every few days
