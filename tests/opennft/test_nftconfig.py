# -*- coding: utf-8 -*-

from opennft.nftconfig import LegacyNftConfigLoader


def test_load_legacy_config(configs_path):
    filename = configs_path / 'NF_PSC_cont_155.ini'

    loader = LegacyNftConfigLoader()
    loader.load(filename)

    assert loader.config.project_name == 'PSC_CONT'
    assert loader.simulation_protocol['ContrastActivation'] == '+1*NFBREG'
