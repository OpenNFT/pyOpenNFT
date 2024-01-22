# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, Union
from configparser import ConfigParser, ParsingError
from ipaddress import IPv4Address
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .errors import OpenNftError
from .log import logger


class LegacyNftConfig(BaseModel):
    """Legacy NFT INI config file model
    """

    project_name: str = Field(alias='ProjectName')
    subject_id: str = Field(alias='SubjectID')
    watch_dir: Path = Field(alias='WatchFolder')
    nf_run_nr: int = Field(alias='NFRunNr')
    image_ser_nr: int = Field(alias='ImgSerNr')
    dicom_first_image_nr: Optional[int] = Field(alias='DicomFirstImageNr')
    first_file_name_txt: str = Field(alias='FirstFileNameTxt')
    first_file_name: str = Field(alias='FirstFileName')
    volumes_nr: int = Field(alias='NrOfVolumes')
    skip_vol_nr: int = Field(alias='nrSkipVol')
    data_type: str = Field(alias='DataType')
    tr: Optional[int] = Field(alias='TR')
    matrix_size_x: int = Field(alias='MatrixSizeX')
    matrix_size_y: int = Field(alias='MatrixSizeY')
    slices_nr: int = Field(alias='NrOfSlices')
    get_mat: Optional[bool] = Field(alias='GetMAT')
    work_dir: Path = Field(alias='WorkFolder')
    simulation_protocol_file: Optional[Path] = Field(alias='StimulationProtocol')
    roi_files_dir: Path = Field(alias='RoiFilesFolder')
    roi_anat_operation: str = Field(alias='RoiAnatOperation')  # FIXME: do not use computable expr in the config!
    roi_group_dir: Optional[Path] = Field(alias='RoiGroupFolder')
    weights_file_name: Optional[str] = Field(alias='WeightsFileName')
    mc_template_file: Path = Field(alias='MCTempl')
    struct_bg_file: Optional[Path] = Field(alias='StructBgFile')
    task_dir: Optional[Path] = Field(alias='TaskFolder')
    prot: Optional[str] = Field(alias='Prot')
    offline_mode: bool = Field(alias='OfflineMode')
    use_tcp_data: Optional[bool] = Field(alias='UseTCPData')
    tcp_data_address: Optional[IPv4Address] = Field(alias='TCPDataIP')
    tcp_data_port: Optional[int] = Field(alias='TCPDataPort')
    type: str = Field(alias='Type')
    max_feedback_val: Optional[int] = Field(alias='MaxFeedbackVal')
    feedback_val_dec: Optional[int] = Field(alias='FeedbackValDec')
    neg_feedback: Optional[bool] = Field(alias='NegFeedback')
    plot_feedback: Optional[bool] = Field(alias='PlotFeedback')
    use_ptb: bool = Field(False, alias='UsePTB')
    display_feedback_screen_id: int = Field(0, alias='DisplayFeedbackScreenID')
    display_feedback_fullscreen: bool = Field(False, alias='DisplayFeedbackFullscreen')
    targ_ang: Optional[int] = Field(alias='TargANG')
    targ_rad: Optional[int] = Field(alias='TargRAD')
    targ_diam: Optional[int] = Field(alias='TargDIAM')
    use_udp_feedback: bool = Field(False, alias='UseUDPFeedback')
    udp_feedback_address: Optional[IPv4Address] = Field(alias='UDPFeedbackIP')
    udp_feedback_port: Optional[int] = Field(alias='UDPFeedbackPort')
    udp_feedback_control_char: Optional[str] = Field(alias='UDPFeedbackControlChar')
    udp_send_condition: Optional[bool] = Field(alias='UDPSendCondition')
    min_feedback_val: Optional[int] = Field(alias='MinFeedbackVal')
    sham_file: Optional[Path] = Field(alias='ShamFile')


FilePathLike = Union[str, Path]


class NftConfigError(OpenNftError):
    pass


class LegacyNftConfigLoader:
    """Legacy OpenNFT INI and JSON configs loader

    The loader for legacy INI and JSON config files for OpenNFT.

    """

    def __init__(self):
        self._config_parser = ConfigParser()
        self._config_parser.optionxform = str  # case-sensitive option names
        self._config: Optional[LegacyNftConfig] = None
        self._simulation_protocol: Dict[str, Any] = {}
        self._filename: Optional[Path] = None

    @property
    def filename(self) -> Optional[Path]:
        return self._filename

    @property
    def config(self) -> Optional[LegacyNftConfig]:
        return self._config

    @property
    def simulation_protocol(self) -> Dict[str, Any]:
        return self._simulation_protocol

    def load(self, filename: FilePathLike) -> None:
        """Load OpenNFT config files

        Loads legacy OpenNFT INI config file and JSON simulation protocol file if it is required.

        Parameters
        ----------
        filename : Path, str
            Legacy OpenNFT INI config file path

        Raises
        ------
        NftConfigError : Any error when loading NFT or simulation protocol config files

        """

        self._filename = None
        self._config = None
        self._simulation_protocol = {}

        logger.info("Loading OpenNFT INI config file '{}'", filename)

        try:
            self._filename = Path(filename)
            self._config_parser.read(str(filename))
        except ParsingError as err:
            raise NftConfigError(f"Cannot read OpenNFT config file '{filename}':\n{err}") from err

        try:
            config_obj = dict(self._config_parser['General'])
            self._config = LegacyNftConfig.parse_obj(config_obj)
        except (KeyError, ValidationError) as err:
            raise NftConfigError(f"Invalid OpenNFT config file '{filename}':\n{err}") from err

        protocol_file = self.config.simulation_protocol_file

        if protocol_file:
            if not protocol_file.is_absolute():
                protocol_file = self.filename.parent / protocol_file
            logger.info("Loading simulation protocol file '{}'", protocol_file)

            if not protocol_file.is_file():
                raise NftConfigError(f"Simulation protocol file '{protocol_file}' "
                                     "is specified in OpenNFT config but does not exist.")

            try:
                with protocol_file.open('r') as fp:
                    self._simulation_protocol = json.load(fp)
            except Exception as err:
                raise NftConfigError(f"Cannot load simulation protocol file '{protocol_file}': {err}") from err

        logger.info("OpenNFT config files have been loaded successfully.")
