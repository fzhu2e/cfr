Overview
========

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome, and take place through `GitHub <https://github.com/fzhu2e/cfr>`_

There are several levels of contributions to an open development software package like `cfr`, including:

- Reporting Bugs
- Updating the documentation
- Updating existing functionalities
- Contributing new functionalities

All of that takes place through GitHub `issues <https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart>`_.

When you start working on an issue, it's a good idea to assign the issue to yourself, again to limit duplication.
If you can't think of an issue of your own, we have you covered: check the list of unassigned issues and assign yourself one you like.
If for whatever reason you are not able to continue working with the issue, please try to unassign it, so other people know it's available again. You can check the list of assigned issues, since people may not be working in them anymore.
If you want to work on one that is assigned, feel free to kindly ask the current assignee (on GitHub) if you can take it (please allow at least a week of inactivity before considering work in the issue discontinued).

Bug reports and enhancement requests
------------------------------------

Bug reports are an important part of improving any software. Having a complete bug report will allow others to reproduce the bug and provide insight into fixing. See this `stackoverflow article <https://stackoverflow.com/help/mcve>`_ and `this blogpost <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_ for tips on writing a good bug report.
Trying the bug-producing code out on the master branch is often a worthwhile exercise to confirm the bug still exists. It is also worth searching existing bug reports and pull requests to see if the issue has already been reported and/or fixed.
Bug reports must:

- Include a minimal working example (a short, self-contained Python snippet reproducing the problem). You can format the code nicely by using GitHub Flavored Markdown.
- Include the full version string of `cfr`, which you can obtain through::

    cfr.__version__

- Explain why the current behavior is wrong/not desired and what you expect or would like to see instead.