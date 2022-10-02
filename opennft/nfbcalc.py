import numpy as np
from scipy.io import savemat


class Nfb():

    def __init__(self, session, iteration):

        self.session = session
        self.iteration = iteration
        self.block_nf = 0
        self.first_nf = 0
        self.disp_value = 0

        # for NFB
        self.condition = 0
        self.end_psc = False
        self.disp_value = 0
        self.disp_values = np.zeros((self.session.nr_vols, 1))
        self.reward = ""
        self.display_data = {"disp_stage": "",
                             "feedback_type": "",
                             "condition": 0,
                             "disp_value": 0,
                             "reward": "",
                             "iteration": 0,
                             "disp_blank_screen": False,
                             "task_seq": False}

        self.norm_perc_values = []
        self.vect_nfbs = np.zeros((self.session.nr_vols, 1))

    def nfb_init(self):

        if self.iteration.iter_number < self.session.config.skip_vol_nr:
            return

        if self.session.config.type == "PSC" or self.session.config.type == "SVM":
            iter_norm_number = self.iteration.iter_norm_number

            condition = self.session.vect_end_cond[iter_norm_number]
            self.condition = condition

            prot = self.session.config.prot
            if prot == "Cont":

                if condition == 1:
                    self.end_psc = False
                    self.disp_value = 0
                    self.reward = ""
                elif condition == 2:
                    self.end_psc = 1
                elif condition == 3:
                    self.end_psc = 0
                    self.reward = ""
                self.display_data["disp_stage"] = "instruction"
                self.display_data["feedback_type"] = "bar_count"

            elif prot == "ContTask":

                if condition == 1:
                    self.end_psc = 0
                    self.disp_value = 0
                    self.reward = ""
                elif condition == 2:
                    self.end_psc = 1
                elif condition == 3:
                    self.end_psc = 0
                    self.reward = ""
                self.display_data["disp_stage"] = "instruction"
                self.display_data["feedback_type"] = "bar_count_task"

            elif prot == "Inter":

                if condition in [1, 2, 3, 4, 5]:
                    self.end_psc = 0
                    self.disp_value = 0
                    self.reward = ""
                    self.display_data["disp_stage"] = "instruction"
                elif condition == 6:
                    self.end_psc = 1
                    self.display_data["disp_stage"] = "feedback"
                self.display_data["feedback_type"] = "value_fixation"

            self.display_data["condition"] = condition
            self.display_data["disp_value"] = self.disp_value
            self.display_data["reward"] = self.reward

        self.display_data["iteration"] = self.iteration.iter_number
        self.display_data["disp_blank_screen"] = 0
        self.display_data["task_seq"] = 0

    def nfb_calc(self):

        type = self.session.config.type
        condition = self.condition
        ind_vol_norm = self.iteration.iter_norm_number
        nr_rois = self.session.nr_rois
        scale_time_series = self.iteration.mr_time_series.scale_time_series

        max_fb_value = self.session.config.max_feedback_val
        min_fb_value = self.session.config.min_feedback_val
        fb_val_dec = self.session.config.feedback_val_dec

        if type == "PSC" and (self.session.config.prot == "Cont" or self.session.config.prot == "ContTask"):

            block_nf = self.block_nf
            first_nf = self.first_nf

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds[0]:
                    block_nf = np.where(self.session.first_nf_inds[0] == ind_vol_norm)[0][0]
                    first_nf = ind_vol_norm

                i_block_bas = np.array([], dtype=np.int32)
                for i_bas in range(0, block_nf + 1):
                    i_block_bas = np.append(i_block_bas, self.session.prot_cond[0][i_bas][2:])

                norm_perc_values = np.zeros((nr_rois,1))
                for i_roi in range(0, nr_rois):
                    m_bas = np.median(scale_time_series[i_roi][i_block_bas])
                    m_cond = scale_time_series[i_roi][ind_vol_norm]
                    norm_perc_values[i_roi] = m_cond - m_bas

                tmp_fb_val = np.median(norm_perc_values)
                self.disp_value = np.round(max_fb_value * tmp_fb_val, decimals=fb_val_dec)

                if not self.session.config.neg_feedback and self.disp_value < 0:
                    self.disp_value = 0
                elif not self.session.config.neg_feedback and self.disp_value < min_fb_value:
                    self.disp_value = min_fb_value
                if self.disp_value > max_fb_value:
                    self.disp_value = max_fb_value

                self.norm_perc_values.append(norm_perc_values)
                self.disp_values[ind_vol_norm] = self.disp_value
            else:

                tmp_fb_val = 0
                self.disp_value = 0

            self.vect_nfbs[ind_vol_norm] = tmp_fb_val
            self.block_nf = block_nf
            self.first_nf = first_nf

            self.display_data["reward"] = ''
            self.display_data["disp_value"] = self.disp_value

        if type == "SVM":

            block_nf = self.block_nf
            first_nf = self.first_nf

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds[0]:
                    block_nf = np.where(self.session.first_nf_inds[0] == ind_vol_norm)[0][0]
                    first_nf = ind_vol_norm

                norm_perc_values = np.zeros((nr_rois, 1))
                for i_roi in range(0, nr_rois):
                    norm_perc_values[i_roi] = scale_time_series[i_roi][ind_vol_norm]

                tmp_fb_val = norm_perc_values.mean()
                self.disp_value = np.round(max_fb_value * tmp_fb_val, decimals=fb_val_dec)

                self.norm_perc_values.append(norm_perc_values)
                self.disp_values[ind_vol_norm] = self.disp_value

                self.vect_nfbs[ind_vol_norm] = tmp_fb_val
                self.block_nf = block_nf
                self.first_nf = first_nf

                self.display_data["reward"] = ''
                self.display_data["disp_value"] = self.disp_value

    def nfb_save(self, save_path):

        path = save_path / "py_disp_values.mat"

        savemat(str(path), {"disp_values": self.disp_values,
                            "vect_end_cond": self.session.vect_end_cond,
                            "norm_perc_values": self.norm_perc_values
                            })
