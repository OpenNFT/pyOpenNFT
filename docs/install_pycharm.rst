.. _install_pycharm:

Installation via PyCharm
========================

The installation process is tested on three x86_64 platforms:

* Linux (Ubuntu 18.04)
* Mac OS (High Sierra 10.13.6)
* Windows 11

Prerequisites
-------------

The following software should be installed:

* `Git <https://git-scm.com/downloads>`_: for installing SPM, Psychtoolbox and OpenNFT
* PyCharm `Professional or Community <https://www.jetbrains.com/pycharm/download/>`_

Install pyOpenNFT
---------------

Create Project by cloning from GitHub repository
++++++++++++++++++++++++++++++++++++++++++++++++++

.. image:: _static/pycharminstall_1.png

Link to the main repository,

.. code-block::

    https://github.com/OpenNFT/pyOpenNFT.git

or, if you plan to contribute to the project, create the fork repository and use your own link:

.. code-block::

    https://github.com/your_github_name/pyOpenNFT.git

.. image:: _static/pycharminstall_2.png

Create and Activate Virtual Environment
++++++++++++++++++++++++++++++++++++++++

To create the virtual environment, go to File -> Settings -> Project Interpreter

.. image:: _static/pycharminstall_3.png

Set the new virtual environment location and choose the interpreter:

.. image:: _static/pycharminstall_4.png

To activate virtual environment, close (click cross near Local) and re-open (click Terminal button) Terminal window.

.. image:: _static/pycharminstall_5.png

Install from Project Directory
++++++++++++++++++++++++++++++


Run Application from PyCharm
----------------------------

Create Run Configuration to run pyOpenNFT:

.. image:: _static/pycharminstall_6.png

.. image:: _static/pycharminstall_7.png

Specify the Module name as "opennft" (NOT Script path) and Project interpreter according to the created Virtual Environment:

.. image:: _static/pycharminstall_8.png

To create the configuration, the checkbox ``Store as project file`` may be required: :ref:`possible_error`.

Press the ``Run`` button,

.. image:: _static/pycharminstall_9.png

