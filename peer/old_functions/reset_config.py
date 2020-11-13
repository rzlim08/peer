#!/usr/bin/env python
"""
Script used on the command line to reset configuration file

Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)  http://jakeson.me

"""

import peer_func as pr
import os
from pprint import pprint


def update_config():

    project_dir, data_dir, stimulus_path = pr.scaffolding()
    os.chdir(project_dir)

    configs = pr.load_config()

    print('\n\nThis is your current config file for reference:')
    print('====================================================\n')
    pprint(configs)
    print('\n')

    print('Update the config file:')
    print('====================================================\n')

    updated_configs = pr.set_parameters(configs, new=True)

    print('\n\nThis is your new config file:')
    print('====================================================\n')
    pprint(updated_configs)
    print('\n')


if __name__ == "__main__":

    update_config()
