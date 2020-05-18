.. _development-workflow:

====================
Development workflow
====================

**Prerequisites**:
You already have your own forked copy of the Simphony_ repository, you have 
configured git, and have linked the upstream repository.

What is described below is a recommended workflow with Git.

.. _Simphony: https://github.com/BYUCamachoLab/simphony


Basic workflow
##############

In short:

1. Start a new *feature branch* for each set of edits that you do.
   See :ref:`below <making-a-new-feature-branch>`.

2. Hack away! See :ref:`below <editing-workflow>`

3. When finished, push your feature branch to your own Github repo, and
   :ref:`create a pull request <asking-for-merging>`.

This way of working helps to keep work well organized and the history
as clear as possible.


For Developers
##############

This package is available on PyPI and updates are regularly pushed as "minor" 
or "micro" (patch) versions. Before submitting any pull requests, however, you should 
ensure that a pip installation of your updated package installs and functions 
properly. To test this, try installing your package locally by removing all 
installed versions of Simphony (by running ```pip3 uninstall simphony``` 
repeatedly until no installations remain) and running the following commands 
(from Simphony's toplevel directory):

```
python3 setup.py sdist bdist_wheel  
pip3 install dist/simphony-[VERSION].tar.gz
```

Also be sure to include tests for all the code you add, making certain the tests are
thorough and always pass.

For Maintainers
###############

Remember that all changes are to be integrated through pull requests. Development
work should be done in branches or forks of the repository. Once implemented 
(and tested on their own), these pull requests should be merged into the 
"master" branch for full testing with the whole program. 

- master (integration and final testing)
- feature-name (feature development and bug fixes)

Even if you are the lone developer, follow the methodology 
[here](https://softwareengineering.stackexchange.com/a/294048).

Be sure to update the version number manually before pushing each new version 
to PyPI. Also be sure to update the changelog.

Eventually, as the project grows, we will work up to using the methods 
detailed below, retained for future reference.


.. _making-a-new-feature-branch:

Making a new feature branch
===========================

First, fetch new commits from the ``upstream`` repository:

::

   git fetch upstream

Then, create a new branch based on the master branch of the upstream
repository::

   git checkout -b my-new-feature upstream/master


.. _editing-workflow:

The editing workflow
====================

Overview
--------

::

   # hack hack
   git status # Optional
   git diff # Optional
   git add modified_file
   git commit
   # push the branch to your own Github repo
   git push origin my-new-feature

In more detail
--------------

#. Make some changes. When you feel that you've made a complete, working set
   of related changes, move on to the next steps.

#. Optional: Check which files have changed with ``git status``.  
   You'll see a listing like this one::

     # On branch my-new-feature
     # Changed but not updated:
     #   (use "git add <file>..." to update what will be committed)
     #   (use "git checkout -- <file>..." to discard changes in working directory)
     #
     #	modified:   README
     #
     # Untracked files:
     #   (use "git add <file>..." to include in what will be committed)
     #
     #	INSTALL
     no changes added to commit (use "git add" and/or "git commit -a")

#. Optional: Compare the changes with the previous version using with ``git
   diff``. This brings up a simple text browser interface that
   highlights the difference between your files and the previous version.

#. Add any relevant modified or new files using  ``git add modified_file``. 
   This puts the files into a staging area, which is a queue
   of files that will be added to your next commit. Only add files that have
   related, complete changes. Leave files with unfinished changes for later
   commits.

#. To commit the staged files into the local copy of your repo, do ``git
   commit``. At this point, a text editor will open up to allow you to write a
   commit message. After saving
   your message and closing the editor, your commit will be saved. For trivial
   commits, a short commit message can be passed in through the command line
   using the ``-m`` flag.

#. Push the changes to your forked repo on GitHub::

      git push origin my-new-feature

.. note::

   Assuming you have followed the instructions in these pages, git will create
   a default link to your GitHub repo called ``origin``.  In git >= 1.7 you
   can ensure that the link to origin is permanently set by using the
   ``--set-upstream`` option::

      git push --set-upstream origin my-new-feature

   From now on git will know that ``my-new-feature`` is related to the
   ``my-new-feature`` branch in your own github repo. Subsequent push calls
   are then simplified to the following::

      git push

   You have to use ``--set-upstream`` for each new branch that you create.


It may be the case that while you were working on your edits, new commits have
been added to ``upstream`` that affect your work. In this case, follow the
:ref:`rebasing-on-master` section of this document to apply those changes to
your branch.


.. _asking-for-merging:

Asking for your changes to be merged with the main repo
=======================================================

When you feel your work is finished, you can create a pull request (PR).

If your changes involve modifications to the API or addition/modification of a
function, you should be sure to emphasize this in the pull request. This may generate
changes and feedback. It might be prudent to start with this step if your
change may be controversial or make existing scripts not backward-compatible.

.. _rebasing-on-master:

Rebasing on master
==================

This updates your feature branch with changes from the upstream `Simphony
github`_ repo. If you do not absolutely need to do this, try to avoid doing
it, except perhaps when you are finished. The first step will be to update
the remote repository with new commits from upstream::

   git fetch upstream

Next, you need to update the feature branch::

   # go to the feature branch
   git checkout my-new-feature
   # make a backup in case you mess up
   git branch tmp my-new-feature
   # rebase on upstream master branch
   git rebase upstream/master

If you have made changes to files that have changed also upstream,
this may generate merge conflicts that you need to resolve. See
:ref:`below<recovering-from-mess-up>` for help in this case.

Finally, remove the backup branch upon a successful rebase::

   git branch -D tmp

.. _Simphony github: https://github.com/BYUCamachoLab/simphony

.. note::

   Rebasing on master is preferred over merging upstream back to your
   branch. Using ``git merge`` and ``git pull`` is discouraged when
   working on feature branches.

.. _recovering-from-mess-up:

Recovering from mess-ups
========================

Sometimes, you mess up merges or rebases. Luckily, in Git it is
relatively straightforward to recover from such mistakes.

If you mess up during a rebase::

   git rebase --abort

If you notice you messed up after the rebase::

   # reset branch back to the saved point
   git reset --hard tmp

If you forgot to make a backup branch::

   # look at the reflog of the branch
   git reflog show my-feature-branch

   8630830 my-feature-branch@{0}: commit: BUG: io: close file handles immediately
   278dd2a my-feature-branch@{1}: rebase finished: refs/heads/my-feature-branch onto 11ee694744f2552d
   26aa21a my-feature-branch@{2}: commit: BUG: lib: make seek_gzip_factory not leak gzip obj
   ...

   # reset the branch to where it was before the botched rebase
   git reset --hard my-feature-branch@{2}

If you didn't actually mess up but there are merge conflicts, you need to
resolve those.  This can be one of the trickier things to get right.


Additional things you might want to do
######################################

Deleting a branch on GitHub
===========================

::

   git checkout master
   # delete branch locally
   git branch -D my-unwanted-branch
   # delete branch on github
   git push origin :my-unwanted-branch

(Note the colon ``:`` before ``test-branch``.  See also:
https://github.com/guides/remove-a-remote-branch
