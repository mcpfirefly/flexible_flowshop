import numpy as np
import collections, re
import itertools, os
from itertools import chain, repeat
import datetime
import pandas as pd


class StudyCase:
    def __init__(self, config):
        self.config = config
        self.experiment_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.experiment_time = datetime.datetime.now().strftime("%H-%M-%S")
        self.experiment_folder = f"{self.experiment_date}/{self.experiment_time}"
        self.filename = self.StudyCaseFilename(self.config.workbook)
        self._initialize_data(self.config)

        self.log_path = f"outputs/{self.experiment_folder}/{self.test}/Logs"
        self.best_model_save_path = (
            f"outputs/{self.experiment_folder}/{self.test}/Saved_Models"
        )

    def _initialize_data(self, cfg):
        self.seed = cfg.seed
        self.N_TIMESTEPS = cfg.N_TIMESTEPS
        self.masking = cfg.masking
        self.reward = cfg.reward
        self.buffer_usage = cfg.buffer_usage
        self.time_in_buffer_custom = cfg.time_in_buffer_custom
        self.obs_size = cfg.obs_size
        self.action_space = cfg.action_space
        self.solution_hints = cfg.solution_hints
        self.products_number = cfg.products_number
        self.generate_heuristic_schedules = cfg.generate_heuristic_schedules
        self.heuristics_policy_rl = cfg.heuristics_policy_rl
        self.recreate_solution = cfg.recreate_solution
        self.test = cfg.test
        self.workbook = cfg.workbook
        self.optuna_trials = cfg.optuna_trials
        np.random.seed(self.seed)

        if self.workbook == "kopanos" and self.recreate_solution is None:
            self.GENERAL_DATA = self.KopanosStudyCase()
            self.SOLUTION = None
        elif self.workbook == "general" and self.recreate_solution is None:
            self.GENERAL_DATA = self.GeneralStudyCase()
            self.SOLUTION = None
        elif self.recreate_solution is not None or (
            self.workbook == "kopanos"
            and self.action_space == "discrete_probs"
            and self.products_number == 30
            and self.solution_hints is not None
        ):
            self.FIFO = False
            self.buffer_usage = "no_buffers"
            self.workbook = "kopanos"
            self.filename = self.StudyCaseFilename()
            self.products_number = 30
            self.GENERAL_DATA = pd.read_excel(
                self.filename, sheet_name="general_data_30"
            )
            self.SOLUTION = self.KopanosStudyCase()

        self.MAKESPAN = list()
        self.TASK_LIST = list()
        self.FINISHED_ORDERS = list()
        self.PROCESSING_TIMES = pd.read_excel(
            self.filename, sheet_name="processing_times"
        )
        self.MACHINES_PER_STAGE = pd.read_excel(self.filename, sheet_name="machines")
        self.NUM_MACHINES_PER_STAGE = list(
            collections.Counter(self.MACHINES_PER_STAGE["stage"].tolist()).values()
        )
        self.IMPACT_FACTORS = pd.read_excel(self.filename, sheet_name="impact_factors")

        if self.buffer_usage != "no_buffers":
            self.BUFFERS_PER_STAGE = pd.read_excel(
                self.filename, sheet_name="buffer_tanks"
            )
            self.NUM_BUFFERS_PER_STAGE = list(
                collections.Counter(self.BUFFERS_PER_STAGE["stages"].tolist()).values()
            )
            self.BUFFERS = list(self.BUFFERS_PER_STAGE.buffer)
        else:
            self.BUFFERS_PER_STAGE = None
            self.NUM_BUFFERS_PER_STAGE = None
            self.BUFFERS = None

        self.MACHINES = list(self.MACHINES_PER_STAGE.machine)
        if self.BUFFERS:
            self.RESOURCES = self.MACHINES + self.BUFFERS
        else:
            self.RESOURCES = self.MACHINES

        self.STAGES = range(len(self.NUM_MACHINES_PER_STAGE))

        self.ORDERS = self.create_orders_from_general_data_workbook()

        if self.recreate_solution != None or (
            self.workbook == "kopanos"
            and self.action_space == "discrete_probs"
            and self.solution_hints != None
        ):
            self.ORDERS = self.SOLUTION
        else:
            self.ORDERS = self.create_orders_from_general_data_workbook()
        self.ORDERS = self.ORDERS.merge(
            self.MACHINES_PER_STAGE, on=["machine"], how="left"
        )
        self.ORDERS = self.set_invalid_actions_workbook()
        self.JOBS = list(set(list(self.ORDERS.order_id)))

        self.INDEX_MACHINES = self.get_index_machines_workbook()
        self.TOTAL_VISITED_STAGES = self.get_total_visited_stages_workbook()
        (
            self.TOTAL_JOBS_PER_STAGE,
            self.N_OPERATIONS,
        ) = self.get_jobs_per_stage_workbook()
        self.CHANGEOVER = self.read_changeover_times()
        self.ID_MACHINES_PER_STAGE = self.obtain_machine_requested_per_stage()

    def create_orders_from_general_data_workbook(self):
        flatten = lambda *n: (
            e
            for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,))
        )
        a = self.GENERAL_DATA.quantity.tolist()
        b = self.GENERAL_DATA.product_code.tolist()
        c = list(chain.from_iterable([x] * y for x, y in zip(b, a)))
        d = list(range(len(c)))  # order_id
        e = len(self.MACHINES_PER_STAGE.machine)
        f = [d, c]
        g = list(map(list, zip(*f)))
        h = [x for item in g for x in repeat(item, e)]
        i = list(self.MACHINES_PER_STAGE.machine)
        j = [list(x) for x in zip(h, itertools.cycle(i))]
        k = list(range(len(h)))
        l = [k, j]
        m = list(map(list, zip(*l)))
        n = []
        for index in k:
            sublist = list(flatten(m[index]))
            n.append(sublist)  # operation_id
        df = pd.DataFrame.from_records(n)
        df.columns = ["operation_id", "order_id", "product_code", "machine"]
        return df

    def set_invalid_actions_workbook(self):
        df = self.ORDERS[["product_code", "machine"]]
        result = [tuple(r) for r in df.to_numpy()]
        list_of_valid_actions = []
        for item in result:
            product = item[0]
            product_id = int(re.search("\d+", product)[0]) - 1
            machine = item[1]

            value = self.PROCESSING_TIMES.loc[product_id, machine]
            if value == "-":
                list_of_valid_actions.append(0)
            else:
                list_of_valid_actions.append(1)

        valid_actions = pd.DataFrame({"valid": list_of_valid_actions})
        orders_with_invalid = self.ORDERS.join(valid_actions)
        return orders_with_invalid

    def get_total_visited_stages_workbook(self):
        df = self.ORDERS[["order_id", "stage", "valid"]]
        df = df[df["valid"] == 1]
        result = [tuple(r) for r in df.to_numpy()]
        set_ = list(set(result))
        self.TOTAL_VISITED_STAGES = [0 for _ in range(len(self.JOBS))]
        for item in set_:
            self.TOTAL_VISITED_STAGES[item[0]] += 1
        return self.TOTAL_VISITED_STAGES

    def get_stage_of_machine_workbook(self):
        df = self.ORDERS[["operation_id", "machine"]]
        result = [tuple(r) for r in df.to_numpy()]
        set_ = list(set(result))
        self.N_OPERATIONS = len(set_)
        self.TOTAL_JOBS_PER_STAGE = [0 for _ in range(len(self.STAGES))]
        for item in set_:
            self.TOTAL_JOBS_PER_STAGE[item[1]] += 1
        return self.TOTAL_JOBS_PER_STAGE, self.N_OPERATIONS

    def get_jobs_per_stage_workbook(self):
        # Get the number of orders per each stage and store in variable TOTAL_JOBS_PER_STAGE
        df = self.ORDERS[["order_id", "stage", "valid"]]
        df = df[df["valid"] == 1]
        result = [tuple(r) for r in df.to_numpy()]
        set_ = list(set(result))
        self.N_OPERATIONS = len(set_)
        self.TOTAL_JOBS_PER_STAGE = [0 for _ in range(len(self.STAGES))]
        for item in set_:
            self.TOTAL_JOBS_PER_STAGE[item[1]] += 1
        return self.TOTAL_JOBS_PER_STAGE, self.N_OPERATIONS

    def get_index_machines_workbook(self):
        # INDEX MACHINES will be used to generate the variable machines_per_stage
        df = self.MACHINES_PER_STAGE[["stage", "machine_id"]]
        result = [tuple(r) for r in df.to_numpy()]
        set_ = list(set(result))
        self.INDEX_MACHINES = [[] for _ in range(len(self.STAGES))]
        for item in set_:
            self.INDEX_MACHINES[item[0] - 1].append(item[1])
        self.INDEX_MACHINES = sorted([sorted(item) for item in self.INDEX_MACHINES])
        return self.INDEX_MACHINES

    def read_changeover_times(self):
        """Function that stores ALL the changeover times excel sheets into
        a list. This was done so that all the changeovers are accessed in memory
        and not accessed at each timestep in the workbook"""
        self.CHANGEOVER = list()
        for i in self.STAGES:
            stage = i + 1
            self.CHANGEOVER.append(
                pd.read_excel(self.filename, sheet_name=f"CH_S{stage}")
            )
        for i in range(len(self.CHANGEOVER)):
            self.CHANGEOVER[i].set_index("Current order", inplace=True)

        return self.CHANGEOVER

    def obtain_machine_requested_per_stage(self):
        """Function that helps to obtain all the IDs of the machines divided
        in sublists that represent each stage"""
        self.ID_MACHINES_PER_STAGE = dict()
        for i in self.STAGES:
            self.ID_MACHINES_PER_STAGE[i] = list(
                self.MACHINES_PER_STAGE.set_index("stage").loc[[i]].loc[:, "machine"]
            )
        return self.ID_MACHINES_PER_STAGE

    def StudyCaseFilename(self, workbook):
        if workbook == "kopanos":
            self.filename = "C:/Users/{}/PycharmProjects/flexible_flowshop/src/flexible_flow_shop/resources/workbooks/kopanos_scheduling_data.xlsx".format(
                os.getlogin()
            )
        elif workbook == "general":
            self.filename = "C:/Users/{}/PycharmProjects/flexible_flowshop/src/flexible_flow_shop/resources/workbooks/general_scheduling_data.xlsx".format(
                os.getlogin()
            )
        return self.filename

    def GeneralStudyCase(self):
        self.GENERAL_DATA = pd.read_excel(self.filename, sheet_name="general_data")
        return self.GENERAL_DATA

    def KopanosStudyCase(self):
        if self.products_number == 30:
            if self.recreate_solution == None or self.solution_hints == None:
                self.GENERAL_DATA = pd.read_excel(
                    self.filename, sheet_name="general_data_30"
                )

                if self.recreate_solution == "kopanos" or (
                    self.action_space == "discrete_probs"
                    and self.solution_hints == "kopanos"
                ):
                    self.GENERAL_DATA = pd.read_excel(
                        self.filename, sheet_name="kopanos_solution_30"
                    )
                elif self.recreate_solution == "SPT" or (
                    self.action_space == "discrete_probs"
                    and self.solution_hints == "SPT"
                ):
                    self.GENERAL_DATA = pd.read_excel(
                        self.filename, sheet_name="spt_solution_30"
                    )
                elif self.recreate_solution == "SCT" or (
                    self.action_space == "discrete_probs"
                    and self.solution_hints == "SCT"
                ):
                    self.GENERAL_DATA = pd.read_excel(
                        self.filename, sheet_name="sct_solution_30"
                    )
                elif self.recreate_solution == "EDD" or (
                    self.action_space == "discrete_probs"
                    and self.solution_hints == "EDD"
                ):
                    self.GENERAL_DATA = pd.read_excel(
                        self.filename, sheet_name="edd_solution_30"
                    )
                elif self.recreate_solution == "FIFO" or (
                    self.action_space == "discrete_probs"
                    and self.solution_hints == "FIFO"
                ):
                    self.GENERAL_DATA = pd.read_excel(
                        self.filename, sheet_name="fifo_solution_30"
                    )

        elif self.products_number == 60:
            self.GENERAL_DATA = pd.read_excel(
                self.filename, sheet_name="general_data_60"
            )

        return self.GENERAL_DATA
