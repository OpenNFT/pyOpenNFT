.. _troubleshooting:

Troubleshooting
===============

Caveats
-------

Python Installations
++++++++++++++++++++

Python installations may require forced installations/upgrades using the following commands

for ``pip``:

.. code-block::

    python -m pip install --upgrade pip
    python -m pip install --upgrade --force-reinstall pip

and for ``wheel``:

.. code-block::

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade --force-reinstall pip setuptools wheel


Paths in the Configuration files
++++++++++++++++++++++++++++++++

All settings and path definitions in the ``*.ini`` files follow conventions of your host operating system, e.g., use '\\' as file separator in Windows and '/' in Unix-based systems.

Runtime errors and troubleshooting
----------------------------------

Real-time exported files not found in watch folder
++++++++++++++++++++++++++++++++++++++++++++++++++

Please check that the real-time data export is properly set up on your scanner and the host computer that is used for OpenNFT. If files are exported correctly, review if the First File Path is set to the correct destination and the MRI Watch Folder is accessible.

- Press 'Review Parameters' and check the status of the First File Path. If you pressed the Setup button and the field is empty indicates that you might have used an invalid syntax to specify the First File Name. Valid formats are:
    - Explicit file names that indicate the first file of the real-time export. Examples:
        - `001_000007_000001.dcm`
        - `001_000007_000001` (file extension will be added based on the MRI Data Type)
    - Template based mechanisms to fill parts of the filename with variables that are defined in the GUI. `{variable name}` defines a variable as specified by the caption in the OpenNFT GUI (e.g., Subject ID), `{#}` refers to the iteration/volume number, and `{…:06}` can be used to set a number format (e.g, 6 digits, zero fill, i.e., 000001). Variable names are case insensitive and spaces are ignored. Examples:
        - `001_{Image Series No:06}_{#:06}.dcm`
        - `{Project Name}/{Subject ID}/{NR Run No:03}_{Image Series No:06}_{#:06}.dcm`

This means users can easily adapt the file locations and file names to their scanner environment.

- Check the status feedback:
    - `MRI Watch folder` exists indicates that the MRI watch folder was found. `MRI Watch folder does not exist` might indicate an error. However, this is not necessarily always the case, given that the folder will be created during image export in certain real-time setups (e.g., Philips DRIN dumper creates a run folder for each new export, e.g., c:\drin\0001, 0002, etc.)

    - `First file does not exist` indicates that OpenNFT has not located the first file of the image series during setup. This is desired in normal online mode operations, as the file export has not yet started. On the other hand, `First file exists` shows that the folder is not empty and might indicate that the wrong folder is used (e.g., the previous run). However, in offline mode, which can be used for offline testing, it is expected that the first file is already available.

Run configuration problem
+++++++++++++++++++++++++

Sometimes it is necessary to select 'Store as project file'. Double-check if PyCharm switches to newly configured venv in the Terminal command line, if not you have to try to close and open the Terminal window.

.. _fork_problem:

Push from local repository to fork problem
++++++++++++++++++++++++++++++++++++++++++

Check following permission and branch settings if you faced problems during push to your fork.

.. image:: _static/troubles_1.png

.. image:: _static/troubles_2.png
