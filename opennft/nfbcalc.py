import numpy as np


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
        self.reward = ""
        self.display_data = {"disp_stage": "",
                             "feedback_type": "",
                             "condition": 0,
                             "disp_value": 0,
                             "reward": "",
                             "iteration": 0,
                             "disp_blank_screen": False,
                             "task_seq": False}

        pass

    def main_loop_entry(self):

        ind_vol = self.iteration.iter_number
        skip_number = self.session.config.skip_vol_nr

        if ind_vol < skip_number:
            return

        if self.session.config.type == "PSC":
            iter_norm_number = ind_vol - skip_number
            self.iteration.iter_norm_number = iter_norm_number

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

    def nfb_calc(self, iter_norm_number, condition):

        type = self.session.config.type
        condition = self.condition
        ind_vol_norm = self.iteration.iter_norm_number

        if type == "SVM":

            block_nf = self.block_nf
            first_nf = self.first_nf
            disp_value = self.disp_value

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds:
                    self.block_nf = self.session.first_nf_inds.index(ind_vol_norm)+1
                    self.first_nf = ind_vol_norm

                pass

        pass
