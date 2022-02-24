from loguru import logger

# --------------------------------------------------------------------------
class MrVol():
    # Префикс модальности позмолит нам потом добавить eeg, nirs и тд
    """Contains volume
    """

    # --------------------------------------------------------------------------
    def __init__(self):
        pass

    # --------------------------------------------------------------------------
    def load_vol(self, file_name):
        logger.info(f"Load vol {file_name}")
        pass

    # --------------------------------------------------------------------------
    def coregister(self, iteration):
        # get reference vol from iteration.session.reference_vol
        logger.info(f"coregister to reference vol")

# --------------------------------------------------------------------------
class MrROI():
    """Contains single ROI
    """
    pass


# --------------------------------------------------------------------------
class MrReferenceVol():
    """Contains registration reference volume for motion correction
        aka mc_template
    """
    pass

