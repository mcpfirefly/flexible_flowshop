import gym, itertools
import numpy as np
import simpy

from src.custom_plotters.gantt_plotter.GanttPlotter import GanttJob, GanttPlotter, JobTypes
from src.flexible_flow_shop.resources.functions.class_objects import Factory, Order
from src.flexible_flow_shop.resources.functions.scheduling_functions import generate_results
from src.flexible_flow_shop.resources.functions.scheduling_functions import processing_times
from src.flexible_flow_shop.resources.functions.scheduling_functions import changeover_times
from src.flexible_flow_shop.resources.functions.scheduling_functions import get_timestep
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    check_if_can_be_legal_again,
    get_impact_factor,
)

from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    get_action_heuristics,
)

from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    prepare_historical_and_flags,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    update_time_arrays,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    current_variables,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import past_variables
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    flag_as_legals_in_next_stage,
    get_orders_next_stage,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    assign_position_in_schedule,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    disable_operations_of_same_job_in_same_stage,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    set_time_machines_idle,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    legalize_with_release_dates,
    GeneratePreliminaryHeuristicResultsFiles,
)
from gym.spaces import Discrete


class flexible_flow_shop(gym.Env):
    def __init__(self, study):
        super().__init__()
        self.study = study
        self.observation_space = Discrete(1)
        self.action_space = Discrete(1)
        self.reset()
        self.num_envs = 1

    def reset(self):
        """Function that initializes the environment state and returns the initial self.observation"""

        self.env = simpy.Environment()
        self.factory = Factory(self.study, self.env)
        self.terminated = False
        self.truncated = False
        self.done = False
        self.total_reward = 0
        self.schedule = []
        self.orders = [
            Order(
                self.study,
                self.env,
                row.operation_id,
                row.order_id,
                row.machine,
                row.machine_id,
                row.product_code,
                row.stage,
                row.valid,
            )
            for _, row in self.study.ORDERS.iterrows()
        ]
        self.num_steps = 0
        self.all_changeover_costs = np.array(0, dtype=np.float)
        self.weighted_total_lateness = 0
        self.legal_operations = [
            True
            if (order.stage == 0 and order.release_date == 0 and order.valid == 1)
            else False
            for order in self.orders
        ]
        self.legal_operations.append(False)
        self.legal_operations = np.array(self.legal_operations)
        self.history_default_changeover = np.zeros(self.study.N_OPERATIONS)
        self.legal_jobs = np.array([True for product in self.study.JOBS], dtype=np.int)
        self.time_until_job_done = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.time_until_operation_done = np.zeros(
            (self.study.N_OPERATIONS), dtype=np.float
        )
        self.jobs_completion = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.operations_completion = np.zeros((self.study.N_OPERATIONS), dtype=np.int)
        self.time_until_machine_free = np.zeros(
            len(self.study.MACHINES), dtype=np.float
        )  # I consider the remaining changeovers
        self.operation_waiting_time = np.zeros(
            (self.study.N_OPERATIONS), dtype=np.float
        )
        self.time_left_previous_stage = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.total_current_idle_time = 0
        self.total_past_idle_time = 0
        # To store preceeding orders for changeover calculations
        self.historical_selection = {}
        for i, machine_id in enumerate(self.study.MACHINES):
            self.historical_selection[machine_id] = []
        self.timestep = 0
        self.counter_historical_selection = {key: 0 for key in self.study.MACHINES}
        self.TASK_LIST = list()
        self.no_operation_active = 0
        self.current_operation_in_machine = np.zeros(
            len(self.study.MACHINES), dtype=np.int
        )  # no_operation
        self.next_operation_in_machine = np.zeros(
            len(self.study.MACHINES), dtype=np.int
        )  # no_operation
        self.stage_operations = np.zeros(self.study.N_OPERATIONS)
        self.stage_jobs = np.zeros(len(self.study.JOBS))
        self.job_processing_time = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.operation_processing_time = np.zeros(
            (self.study.N_OPERATIONS), dtype=np.float
        )
        self.job_changeover_time = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.operation_changeover_time = np.zeros(
            (self.study.N_OPERATIONS), dtype=np.float
        )
        self.job_waiting_time = np.zeros(len(self.study.JOBS), dtype=np.float)
        self.legal_machines = np.ones(len(self.study.MACHINES), dtype=np.int)
        self.counter = 0
        self.scheduled_operations = np.zeros((self.study.N_OPERATIONS), dtype=np.int)
        self.scheduled_jobs = np.zeros(self.study.N_OPERATIONS, dtype=np.int)
        self.scheduled_machines = np.zeros((self.study.N_OPERATIONS), dtype=np.int)
        self.stages_queue_size = np.zeros(len(self.study.STAGES), dtype=np.int)
        self.stage_progression = np.zeros(len(self.study.STAGES), dtype=np.float)
        self.machines_idle = [
            [0, 0] for i in range(len(self.study.MACHINES))
        ]  # (active machine boolean,idle_time)
        self.time_machines_idle = np.zeros(len(self.study.MACHINES), dtype=np.float)
        self.completion_score = np.array(0, dtype=np.float)
        self.sim_duration = (
            0  # small positive constant to avoid issues when dividing over makespan
        )
        self.oc_costs = 0
        self.machine_queue = {key: 0 for key in self.study.MACHINES}
        self.reward = 0
        machines_to_select = []
        self.cumulative_changeover = 0
        self.total_waiting = 0
        self.difference_idle_times = 0
        self.oc_costs_current = 0
        self.oc_costs_past = 0
        self.wl_past = 0
        self.wl_current = 0
        self.sim_duration_past = 0
        self.sim_duration_current = 0
        self.time_inactive = 0
        self.episode_length = 0
        self.episode_length_current = 0
        self.episode_length_past = 0
        self.job_completion_current = 0
        self.job_completion_past = 0
        self.legal_machines_per_stage = [
            self.legal_machines[i] for i in self.study.INDEX_MACHINES
        ]
        self.history_buffer = [[] for i in self.study.STAGES]
        self.past_product_in_buffer = [[] for i in self.study.STAGES]
        self.current_product_in_buffer = [[] for i in self.study.STAGES]
        self.original_history = [[] for i in self.study.STAGES]
        self.list_of_reamining_times = []
        self.remaining_time = 0
        self.current_product_in_machine = [[] for i in self.study.MACHINES]
        self.heuristics_products_per_stage = [[] for i in self.study.STAGES]
        self.counter_positions_heuristics = [0 for i in self.study.STAGES]
        self.heuristics_solution = []
        self.counter_heuristics = 0
        return self._get_observation()

    def valid_action_mask(self):
        """Only valid for discrete environment. Used with MaskablePPO"""
        if self.legal_operations[:len(self.legal_operations)-1].any():
            self.legal_operations[-1] = False
        else:
            self.legal_operations[-1] = True

        if self.study.heuristics_policy_rl != None:
            alpha = 0.2
            mask_ppo = self.legal_operations
            legal_operations_ppo = list(np.argwhere(np.array(mask_ppo, dtype=int)).flatten())
            legal_operations_heuristics, self.counter_heuristics = get_action_heuristics(self, legal_operations_ppo,
                                                                    self.counter_heuristics, None,
                                                                    self.study.heuristics_policy_rl,
                                                                    self.study.CHANGEOVER)

            mask_heuristics = np.zeros(np.array(len(self.orders)+1))

            if isinstance(legal_operations_heuristics, list):
                for index in legal_operations_heuristics:
                    mask_heuristics[index] = 1
            else:
                if legal_operations_heuristics == len(self.orders):
                    mask_heuristics[-1] = 1
                else:
                    mask_heuristics[legal_operations_heuristics] = 1


            random_variable = np.random.random()
            if random_variable < (1 - alpha):
                mask = mask_heuristics
            else:
                mask = mask_ppo
        else:
            mask = []
            mask.append(self.legal_operations)

        if self.study.buffer_usage == "agent_decides":
            mask_use_buffer = np.ones(2)  # [0, 1]
            mask_time_in_buffer = [np.zeros(1), np.ones(5)]  # [0, 1, 2, 3, 4, 5]
            mask.append(mask_use_buffer)
            mask.append(mask_time_in_buffer[0])
            mask.append(mask_time_in_buffer[1])
            mask = list(itertools.chain.from_iterable(mask))

        mask = np.array(mask, dtype=bool)
        mask = mask.flatten()
        return mask

    def _scheduling(self, env, action):
        """This function defines the main scheduling process of the environment.
        It uses an order and machine as inputs, and each order is allocated
        in the corresponding stage in sequential order taking into account the machine it
        is instructed to allocate into. Once the order finishes its processing in
        a stage, the variable order.stage is increased by += 1 and order it's
        inserted into the orders list on the first position."""

        self.sim_duration = np.round(self.sim_duration, 4)
        self.oc_costs = np.round(self.oc_costs, 4)
        self.weighted_total_lateness = np.round(self.weighted_total_lateness, 4)

        if not self.finished_orders.all():
            self.sim_duration = env.now

        update_time_arrays(self)
        self.completion_score = (sum(self.jobs_completion)) / len(
            self.study.JOBS
        )  # ratio between jobs progression and number of jobs
        set_time_machines_idle(self)
        legalize_with_release_dates(self)

        if self.legal_operations[action] and action != len(self.study.ORDERS):
            order = self.orders[action]

            if self.study.action_space == "continuous" or (self.study.masking == False and self.study.action_space == "discrete"):
                self.reward = 1

            if self.study.buffer_usage != "no_buffers" and order.stage > 0:
                buffer_stage = order.stage - 1
                if self.study.buffer_usage == "random":
                    go_to_buffer = np.random.randint(2)
                    order.time_in_buffer = self.study.time_in_buffer_custom
                elif self.study.buffer_usage == "agent_decides":
                    go_to_buffer = order.go_to_buffer
                    order.time_in_buffer = order.time_in_buffer
                elif self.study.buffer_usage == "always_buffers":
                    go_to_buffer = 1
                    order.time_in_buffer = self.study.time_in_buffer_custom

                if go_to_buffer and order.visited_buffer == False:
                    if len(self.factory.buffer[buffer_stage].users) == 0 or (
                        order.product_code
                        == self.current_product_in_buffer[buffer_stage]
                        and len(self.factory.buffer[buffer_stage].users) < 10
                    ):
                        order.visited_buffer = True
                        self.history_buffer[buffer_stage].append(order.operation_id)
                        self.current_product_in_buffer[
                            buffer_stage
                        ] = order.product_code
                        disable_operations_of_same_job_in_same_stage(self, order)
                        self.legal_operations[order.operation_id] = False
                        with self.factory.buffer[buffer_stage].request() as request:
                            yield request

                            #####CHANGEOVER BUFFER STARTS
                            if len(self.history_buffer[buffer_stage]) > 1:
                                past_order_id = self.history_buffer[buffer_stage].pop(0)
                                past_order = self.orders[past_order_id]

                                if self.remaining_time > 0:
                                    past_order.time_end_changeover_buffer = (
                                        self.env.now + self.remaining_time
                                    )
                                    past_order.changeover_in_progress = True

                                if order.product_code != past_order.product_code:
                                    order.buffer_changeover = 1
                                    order.changeover_in_progress = True
                                    self.time_until_operation_done[
                                        order.position_in_schedule
                                    ] = order.buffer_changeover
                                    self.time_until_job_done[
                                        order.order_id
                                    ] = order.buffer_changeover
                                    order.time_end_changeover_buffer = (
                                        self.env.now + order.buffer_changeover
                                    )
                                    yield env.timeout(order.buffer_changeover)
                                    order.changeover_in_progress = False
                                    buffer_changeover_object = GanttJob(
                                        past_order.time_end_block,
                                        order.buffer_changeover,
                                        self.study.BUFFERS[buffer_stage],
                                        "CHANGEOVER",
                                        job_type=JobTypes.CHANGEOVER,
                                    )
                                    self.TASK_LIST.append(buffer_changeover_object)

                                else:
                                    while past_order.changeover_in_progress:
                                        self.remaining_time = (
                                            past_order.time_end_changeover_buffer
                                            - self.env.now
                                        )
                                        self.time_until_operation_done[
                                            order.position_in_schedule
                                        ] = self.remaining_time
                                        self.time_until_job_done[
                                            order.position_in_schedule
                                        ] = self.remaining_time

                                        if self.remaining_time > 0:
                                            yield env.timeout(self.remaining_time)
                                        else:
                                            past_order.changeover_in_progress = False
                            # PROCESS IN BUFFER
                            self.time_until_operation_done[
                                order.position_in_schedule
                            ] = order.time_in_buffer
                            self.time_until_job_done[
                                order.order_id
                            ] = order.time_in_buffer

                            order.time_start_buffer = self.env.now
                            order.time_end_buffer = (
                                order.time_start_buffer + order.time_in_buffer
                            )
                            order.time_start_changeover_buffer = order.time_end_buffer

                            yield env.timeout(order.time_in_buffer)

                            buffer_task_object = GanttJob(
                                order.time_start_buffer,
                                order.time_in_buffer,
                                self.study.BUFFERS[buffer_stage],
                                order.product_code,
                                job_type=JobTypes.PROCESS,
                            )
                            self.TASK_LIST.append(buffer_task_object)
                            order.time_start_block = self.env.now

                            prepare_historical_and_flags(
                                self, order, order.machine, action
                            )
                            initial_number_of_users = len(
                                self.factory.buffer[buffer_stage].users
                            )
                            while self.legal_machines[order.machine_id] == False:
                                if (
                                    len(self.factory.buffer[buffer_stage].users)
                                    > initial_number_of_users
                                    and order.time_end_block == None
                                ):
                                    order.time_end_block = self.env.now
                                yield env.timeout(0.01)
                            intermediate_number_of_users = len(
                                self.factory.buffer[buffer_stage].users
                            )
                            if order.time_end_block == None:
                                order.time_end_block = self.env.now
                            block_duration = (
                                order.time_end_block - order.time_start_block
                            )

                            # If the current order is the only one remaining in the buffer, plot the blocking
                            if order.time_end_block != order.time_start_block:
                                if (
                                    len(self.factory.buffer[buffer_stage].users) == 1
                                    or initial_number_of_users
                                    <= intermediate_number_of_users
                                ):
                                    buffer_block_task_object = GanttJob(
                                        order.time_start_block,
                                        block_duration,
                                        self.study.BUFFERS[buffer_stage],
                                        "blocking",
                                        job_type=JobTypes.BLOCKING,
                                    )
                                    self.TASK_LIST.append(buffer_block_task_object)

                            yield self.factory.buffer[buffer_stage].release(request)

            if self.study.generate_heuristic_schedules != None or self.study.heuristics_policy_rl != None:
                MAX_QUEUE_SIZE = np.inf
            else:
                MAX_QUEUE_SIZE = 1

            if self.machine_queue[order.machine] < MAX_QUEUE_SIZE:
                assign_position_in_schedule(
                    self, order, self.study.N_OPERATIONS, self.study.MACHINES
                )
                disable_operations_of_same_job_in_same_stage(self, order)
                self.no_operation_active = 0
                order.scheduled_by_agent = 1
                self.orders[action].scheduled = 1
                prepare_historical_and_flags(self, order, order.machine, action)

                ########## START #############
                i = order.stage
                if self.study.generate_heuristic_schedules != None:
                    order.position_heuristics = self.counter_positions_heuristics[i]
                    self.counter_positions_heuristics[i] += 1

                    self.heuristics_products_per_stage[i].append(order.order_id)
                    self.heuristics_solution.append(order.operation_id)

                order.changeover_value = 0
                order.time_changeover_starts = 0
                order.time_changeover_ends = 0
                stage = i + 1  # for #print statements
                # Arrival of a product to a stage
                order.time_arriving_stage = self.env.now
                #print("{} arrives at stage {} at {} and waits for machine allocation.".format(order.product_code, stage, order.time_arriving_stage))

                self.machine_queue[order.machine] += 1

                with self.factory.stage[i].get(
                    lambda machine: machine == order.machine
                ) as request:
                    self.time_until_job_done[
                        order.order_id
                    ] = self.time_until_machine_free[order.machine_id]
                    self.time_until_operation_done[
                        order.position_in_schedule
                    ] = self.time_until_machine_free[order.machine_id]
                    self.current_product_in_machine[order.machine_id].append(
                        order.product_code
                    )
                    yield request
                    self.legal_machines[order.machine_id] = 0
                    self.machine_queue[order.machine] -= 1
                    self.stage_progression[i] += 1 / (
                        self.study.TOTAL_JOBS_PER_STAGE[i]
                    )
                    self.stages_queue_size = [
                        len(self.factory.stage[i].get_queue)
                        for i in range(len(self.study.STAGES))
                    ]

                    order.machine = request.value
                    self.machines_idle[order.machine_id] = [1, 0]

                    order.time_start_process = self.env.now
                    self.schedule.append(order.operation_id)
                    self.current_operation_in_machine[
                        order.machine_id
                    ] = order.operation_id
                    relative_stage = np.around(
                        np.divide(order.stage, order.total_stages), 4
                    )
                    self.stage_operations[order.position_in_schedule] = relative_stage
                    self.stage_jobs[order.order_id] = relative_stage

                    # waiting time between the allocation and the start of the process
                    order.waiting_value = np.around(
                        (order.time_start_process - order.time_arriving_stage), 4
                    )
                    self.operation_waiting_time[
                        order.position_in_schedule
                    ] = order.waiting_value
                    self.job_waiting_time[order.order_id] += order.waiting_value

                    self.total_waiting += order.waiting_value
                    # else:
                    #    # waiting time between last order finishing in previous stage and start of the order's processing in the current stage
                    #    order.waiting_value = np.around((order.time_start_process - self.time_left_previous_stage[order.order_id]),4)
                    #    self.operation_waiting_time[order.position_in_schedule] = order.waiting_value
                    #    self.job_waiting_time[order.order_id] += self.operation_waiting_time[order.position_in_schedule]
                    #print("{} waited {} in queue to enter stage {}.".format(order.product_code, order.waiting_value, stage))
                    #print("{} starts processing in machine {} at {}.".format(order.product_code,order.machine,order.time_start_process))
                    process_time = processing_times(
                        self, env, self.study.PROCESSING_TIMES, order, order.machine_id
                    )
                    yield env.timeout(process_time)

                    self.operations_completion[order.position_in_schedule] = 1
                    self.jobs_completion[order.order_id] += np.divide(
                        1, order.total_stages
                    )
                    order.time_leaving_stage = self.env.now
                    order.processing_value = (
                        order.time_leaving_stage - order.time_start_process
                    )
                    #print("{} left stage {} at {}.".format(order.product_code, stage, order.time_leaving_stage))
                    self.sim_duration = env.now
                    self.time_left_previous_stage[
                        order.order_id
                    ] = order.time_leaving_stage

                    processing_task_object = GanttJob(
                        order.time_start_process,
                        order.processing_value,
                        order.machine,
                        order.product_code,
                        job_type=JobTypes.PROCESS,
                    )
                    self.TASK_LIST.append(processing_task_object)

                    # When the order finishes processing in the machine, flag the next order's operation
                    # in the following stage as legal, and also the "global" flag of the order as legal.
                    order.next_stage = get_orders_next_stage(
                        self, order, self.study.STAGES
                    )

                    if (
                        self.study.generate_heuristic_schedules == "FIFO"
                        and order.stage != len(self.study.STAGES) - 1
                    ):
                        if (
                            order.position_heuristics
                            > self.counter_positions_heuristics[order.next_stage]
                            and order.next_stage != len(self.study.STAGES) - 1
                            and self.study.TOTAL_JOBS_PER_STAGE[order.next_stage]
                            != len(self.study.JOBS)
                        ):
                            while (
                                order.position_heuristics
                                > self.counter_positions_heuristics[
                                    order.next_stage + 1
                                ]
                            ):
                                yield env.timeout(0.001)
                            self.counter_positions_heuristics[
                                order.next_stage
                            ] = order.position_heuristics
                        while (
                            order.position_heuristics
                            != self.counter_positions_heuristics[order.next_stage]
                        ):
                            yield env.timeout(0.001)

                    # arrays of legal operations in stage 0
                    flag_as_legals_in_next_stage(self, order, self.study.STAGES)
                    # if order finished its processing in the last stage
                    if (
                        np.around(
                            self.jobs_completion[order.order_id] * order.total_stages, 2
                        )
                        == order.total_stages
                    ):
                        self.factory.finished_orders.append(order.order_id)
                        order.earliness = order.due_date - order.time_leaving_stage
                        order.tardiness = order.time_leaving_stage - order.due_date
                        if order.tardiness < 0:
                            order.tardiness = 0
                        if order.earliness < 0:
                            order.earliness = 0
                        order.weighted_lateness = (
                            0.9 * order.earliness + 4.5 * order.tardiness
                        )
                        self.weighted_total_lateness += order.weighted_lateness

                    self.machines_idle[order.machine_id] = [0, 0]

                    #####Changeover######

                    # STAGE 1 has no changeovers.
                    if self.study.workbook == "kopanos":
                        changeover_condition = (
                            order.stage != 0
                            and np.around(self.stage_progression[i], 3) != 1
                        )
                    else:
                        changeover_condition = (
                            np.around(self.stage_progression[i], 3) != 1
                        )
                    if changeover_condition:
                        order.time_changeover_starts = self.env.now
                        #print("Changeover in machine {} starts at {}.".format(order.machine, order.time_changeover_starts))
                        # While no other orders have been allocated do:
                        while len(self.historical_selection[order.machine]) == 0:
                            # While there is still no next order, keep the machine "busy" and perform a "dummy"
                            # changeover. When the next product is actually known, break the loop and compare
                            # if the actual changeover is greater than the default changeover applied during the loop.
                            # If yes, apply the remaining changeover. If not, just allocate the next product directly.
                            default_changeover_time = 0.01
                            self.history_default_changeover[
                                order.position_in_schedule
                            ] += default_changeover_time
                            yield env.timeout(default_changeover_time)

                        # ALLOCATION FINALLY OCCURS!
                        # Identify the next order and its parameters
                        # Pop the index 1 as index 0 is the initial order for every machine (ignore index 0)
                        next_order_id = self.historical_selection[order.machine].pop(0)
                        next_order = self.orders[next_order_id]
                        self.next_operation_in_machine[
                            order.machine_id
                        ] = next_order.operation_id
                        order.next_order = next_order.product_code
                        order.impact_factor_value = get_impact_factor(
                            self.study.IMPACT_FACTORS, order, next_order
                        )
                        order.changeover_value = changeover_times(
                            self, env, self.study.CHANGEOVER[i], order, next_order
                        )

                        # In case there is still a remaining changeover time, apply:
                        if (
                            order.changeover_value
                            > self.history_default_changeover[
                                order.position_in_schedule
                            ]
                        ):
                            remaining_changeover_time = (
                                order.changeover_value
                                - self.history_default_changeover[
                                    order.position_in_schedule
                                ]
                            )
                            self.time_until_machine_free[
                                order.machine_id
                            ] = remaining_changeover_time
                            self.time_until_operation_done[
                                order.position_in_schedule
                            ] = remaining_changeover_time
                            self.time_until_job_done[
                                order.order_id
                            ] = remaining_changeover_time
                            yield env.timeout(remaining_changeover_time)

                        order.time_changeover_ends = self.env.now
                        order.changeover_cost = (
                            order.changeover_value * order.impact_factor_value
                        )
                        self.cumulative_changeover += order.changeover_value
                        self.all_changeover_costs += order.changeover_cost
                        #print("Changeover to receive {}  in machine {} finished at {}.".format(next_order.product_code,order.machine,order.time_changeover_starts + order.changeover_value))

                    if order.changeover_value != 0:
                        changeover_task_object = GanttJob(
                            order.time_leaving_stage,
                            order.changeover_value,
                            order.machine,
                            "CHANGEOVER",
                            job_type=JobTypes.CHANGEOVER,
                        )
                        self.TASK_LIST.append(changeover_task_object)

                    yield self.factory.stage[i].put(
                        order.machine
                    )  # Put until next order is ready to start processing

                    self.legal_machines[order.machine_id] = 1
                    self.current_product_in_machine[order.machine_id].pop(0)

            else:
                if order.visited_buffer == True:
                    self.legal_operations[order.operation_id] = True

                while self.machine_queue[order.machine] >= MAX_QUEUE_SIZE:
                    self.legal_operations[order.operation_id] = False
                    yield env.timeout(self.timestep)

                can_be_legal = check_if_can_be_legal_again(self, order)
                if can_be_legal:
                    self.legal_operations[order.operation_id] = True

    def step(self, action):
        """Function that sends an action to the scheduling function described above
        and receives the self.observation, self.reward, terminated and truncated signals.
        """
        #print("Legal actions: {}".format(np.array(np.argwhere(self.legal_operations),dtype=int).flatten().tolist()))
        #print("Action selected from legal actions: {}".format(action))
        if not self.legal_operations[action]:
            self.episode_length += 1

            if self.finished_orders.all() or np.round(self.completion_score, 5) == 1.0:
                print("------------------------------------------------------")
                print(
                    "Episode terminated, good job! Completion score: {}%".format(
                        self.completion_score * 100
                    )
                )
                print("Episode length: {}".format(self.episode_length))
                print("Makespan value: {}".format(self.sim_duration))
                print("OCC value: {}".format(self.oc_costs))
                print("WL value: {}".format(self.weighted_total_lateness))
                print("Reward: {}".format(self.total_reward))
                if (
                    self.sim_duration < 30
                    or self.oc_costs < 70
                    or self.weighted_total_lateness < 250
                ):
                    self.render()
                self.done = True

            if self.sim_duration >= 60 or self.episode_length >= 2500:
                print("------------------------------------------------------")
                print(
                    "Episode truncated! Completion score: {}%".format(
                        self.completion_score * 100
                    )
                )
                print("Episode length: {}".format(self.episode_length))
                self.render()
                print("Makespan value: {}".format(self.sim_duration))
                print("OCC value: {}".format(self.oc_costs))
                print("WL value: {}".format(self.weighted_total_lateness))
                print("Reward: {}".format(self.total_reward))
                self.done = True

            info = {
                "sim_duration": self.sim_duration,
                "oc_costs": self.oc_costs,
                "weighted_lateness": self.weighted_total_lateness,
                "total_waiting": self.total_waiting,
                "task_list": self.TASK_LIST,
                "completion_score": self.completion_score,
                "schedule": self.schedule,
                "total_reward": self.total_reward,
            }

            self.reward = -10

            return self._get_observation(), self.reward, self.done, info

        else:
            self.timestep = get_timestep(self, action)
            self.episode_length += 1
            past_variables(self)
            self.env.process(self._scheduling(self.env, action))
            self.env.run(until=self.env.now + self.timestep)
            current_variables(self)

            self.makespan_reward = -(
                np.around(self.sim_duration_current, 4)
                - np.around(self.sim_duration_past, 4)
            )
            self.occ_reward = -(
                np.around(self.oc_costs_current, 4) - np.around(self.oc_costs_past, 4)
            )
            self.wl_reward = -(
                np.around(self.wl_current, 4) - np.around(self.wl_past, 4)
            )
            self.length_reward = -(
                np.around(self.episode_length_current, 4)
                - np.around(self.episode_length_past, 4)
            )

            if self.study.reward == "MAKESPAN":
                self.total_reward += self.makespan_reward + self.reward
            elif self.study.reward == "OCC":
                self.total_reward += self.occ_reward + self.reward
            elif self.study.reward == "WL":
                self.total_reward += self.wl_reward + self.reward
            elif self.study.reward == "LENGTH":
                self.total_reward += self.length_reward + self.reward

            info = {
                "sim_duration": self.sim_duration,
                "oc_costs": self.oc_costs,
                "weighted_lateness": self.weighted_total_lateness,
                "total_waiting": self.total_waiting,
                "task_list": self.TASK_LIST,
                "completion_score": self.completion_score,
                "schedule": self.schedule,
                "total_reward": self.total_reward,
            }

            if self.finished_orders.all() or np.round(self.completion_score, 5) == 1.0:
                print("------------------------------------------------------")
                print(
                    "Episode terminated, good job! Completion score: {}%".format(
                        self.completion_score * 100
                    )
                )
                print("Episode length: {}".format(self.episode_length))
                print("Makespan value: {}".format(self.sim_duration))
                print("OCC value: {}".format(self.oc_costs))
                print("WL value: {}".format(self.weighted_total_lateness))
                print("Reward: {}".format(self.total_reward))
                if self.study.solution_hints == "kopanos":
                    if self.sim_duration < 28 or self.oc_costs < 65 or self.weighted_total_lateness < 200:
                        self.render()
                else:
                    if self.sim_duration < 30 or self.oc_costs < 70 or self.weighted_total_lateness < 250:
                        self.render()
                self.done = True

            if self.sim_duration >= 100 or self.episode_length > 2500:
                print("------------------------------------------------------")
                print(
                    "Episode truncated! Completion score: {}%".format(
                        self.completion_score * 100
                    )
                )
                print("Episode length: {}".format(self.episode_length))
                self.render()
                print("Makespan value: {}".format(self.sim_duration))
                print("OCC value: {}".format(self.oc_costs))
                print("WL value: {}".format(self.weighted_total_lateness))

                print("Reward: {}".format(self.total_reward))
                self.done = True

            return self._get_observation(), self.reward, self.done, info

    def _get_observation(self):
        self.legal_machines_per_stage = [
            self.legal_machines[i] for i in self.study.INDEX_MACHINES
        ]
        self.finished_orders_index = [
            i
            for i, e in enumerate(self.study.JOBS)
            if e in set(self.factory.finished_orders)
        ]
        self.finished_orders = np.in1d(
            range(len(self.study.JOBS)), self.finished_orders_index
        )
        self.oc_costs = self.sim_duration * 0.9 + self.all_changeover_costs
        self.machine_queue_list = list(self.machine_queue.values())
        self.observation = 0
        return self.observation

    def render(self):
        """Function that generates the Gantt Chart which is automatically saved in disk
        and generates the results in an excel workbook"""
        my_plotter = GanttPlotter(
            resources=self.study.RESOURCES,
            jobs=self.TASK_LIST,
            xticks_step_size=8,
            xticks_max_value=float(np.around(self.sim_duration, 3)),
        )

        episode_makespan = np.around(self.sim_duration, 3)
        episode_oc_costs = np.around(self.oc_costs, 3)
        episode_wl = np.around(self.weighted_total_lateness, 3)

        results_path = "outputs/{}/{}/render".format(
            self.study.experiment_folder, self.study.test
        )
        if self.study.generate_heuristic_schedules != None:
            filename_gantt = "{}/{}_Mks_{}_OCC_{}_WL_{}_GanttPlot.png".format(
                results_path,
                self.study.generate_heuristic_schedules,
                episode_makespan,
                episode_oc_costs,
                episode_wl,
            )
        else:
            filename_gantt = "{}/Mks_{}_OCC_{}_WL_{}_GanttPlot.png".format(
                results_path, episode_makespan, episode_oc_costs, episode_wl
            )

        my_plotter.generate_gantt(
            title="Kopanos Scheduling Problem",
            ylabel="",
            description="Makespan = {}, OCC = {}, WL = {}".format(
                episode_makespan, episode_oc_costs, episode_wl
            ),
            label_processes=True,
            label_changeovers=False,
            color_mode=1,
            save_to_disk=True,
            filename=filename_gantt,
        )

        # my_plotter.show_gantt()
        generate_results(
            self.orders,
            self.study.ORDERS,
            episode_makespan,
            self.factory.finished_orders,
            self.study.generate_heuristic_schedules,
            self.study.test,
            self.study.experiment_folder,
        )

        if self.study.generate_heuristic_schedules != None:
            GeneratePreliminaryHeuristicResultsFiles(
                self,
                results_path,
                self.study.generate_heuristic_schedules,
                episode_makespan,
                episode_oc_costs,
                episode_wl,
            )

    def _get_product_number(self, probability):
        item_number = int(
            probability * len(self.study.JOBS)
        )  # Multiply by the total number of items
        return min(
            item_number, (len(self.study.JOBS) - 1)
        )  # Cap the item number at 29 (total items - 1)

    def _get_machine_number(self, probability):
        item_number = int(
            probability * len(self.study.MACHINES)
        )  # Multiply by the total number of items
        return min(
            item_number, (len(self.study.MACHINES) - 1)
        )  # Cap the item number at 16 (total items - 1)

    def _observations_big(self):
        # finished orders list
        self.legal_machines_per_stage = [
            self.legal_machines[i] for i in self.study.INDEX_MACHINES
        ]
        self.finished_orders_index = [
            i
            for i, e in enumerate(self.study.JOBS)
            if e in set(self.factory.finished_orders)
        ]
        self.finished_orders = np.in1d(
            range(len(self.study.JOBS)), self.finished_orders_index
        )
        self.machine_queue_list = list(self.machine_queue.values())
        observation = []
        # global_schedule_observations
        observation.append(self.scheduled_operations)
        # observation.append(self.scheduled_jobs)
        observation.append(self.scheduled_machines)
        # legal_moves_observations
        observation.append(self.legal_operations)
        observation.append(self.legal_jobs)
        observation.append(self.legal_machines)
        # progress_observations
        observation.append(self.operations_completion)
        observation.append(self.jobs_completion)
        # production_time_observations
        observation.append(self.operation_processing_time)
        observation.append(self.operation_changeover_time)
        observation.append(self.operation_waiting_time)
        observation.append(self.job_processing_time)
        observation.append(self.job_changeover_time)
        observation.append(self.job_waiting_time)
        # time_until_observations
        observation.append(self.time_until_operation_done)
        observation.append(self.time_until_job_done)
        observation.append(self.time_until_machine_free)
        # machines_observations
        observation.append(self.current_operation_in_machine)
        observation.append(self.next_operation_in_machine)
        observation.append(self.time_machines_idle)
        observation.append(self.machine_queue_list)
        # observation = np.ndarray.flatten(np.array(observation))
        observation = list(itertools.chain.from_iterable(observation))

        # objective_variables_observations
        observation.insert(0, self.sim_duration)
        observation.insert(1, self.oc_costs)
        observation.insert(2, self.weighted_total_lateness)

        observation = np.array(observation)

        return observation

    def _observations_medium(self):
        # finished orders list
        self.legal_machines_per_stage = [
            self.legal_machines[i] for i in self.study.INDEX_MACHINES
        ]
        self.finished_orders_index = [
            i
            for i, e in enumerate(self.study.JOBS)
            if e in set(self.factory.finished_orders)
        ]
        self.finished_orders = np.in1d(
            range(len(self.study.JOBS)), self.finished_orders_index
        )
        self.machine_queue_list = list(self.machine_queue.values())

        observation = []
        # global_schedule_observations
        observation.append(self.scheduled_jobs)
        observation.append(self.scheduled_machines)
        # legal_moves_observations
        observation.append(self.legal_operations)
        observation.append(self.legal_jobs)
        observation.append(self.legal_machines)
        # progress_observations
        observation.append(self.jobs_completion)
        # production_time_observations
        observation.append(self.job_processing_time)
        observation.append(self.job_changeover_time)
        observation.append(self.job_waiting_time)
        # time_until_observations
        observation.append(self.time_until_job_done)
        observation.append(self.time_until_machine_free)
        # machines_observations
        observation.append(self.time_machines_idle)
        observation.append(self.machine_queue_list)
        # observation = np.ndarray.flatten(np.array(observation))
        observation = list(itertools.chain.from_iterable(observation))

        # objective_variables_observations
        observation.insert(0, self.sim_duration)
        observation.insert(1, self.oc_costs)
        observation.insert(2, self.weighted_total_lateness)

        observation = np.array(observation)

        return observation

    def _observations_small(self):
        # finished orders list
        self.legal_machines_per_stage = [
            self.legal_machines[i] for i in self.study.INDEX_MACHINES
        ]
        self.finished_orders_index = [
            i
            for i, e in enumerate(self.study.JOBS)
            if e in set(self.factory.finished_orders)
        ]
        self.finished_orders = np.in1d(
            range(len(self.study.JOBS)), self.finished_orders_index
        )
        self.machine_queue_list = list(self.machine_queue.values())

        observation = []
        # global_schedule_observations
        observation.append(self.scheduled_jobs)
        observation.append(self.scheduled_machines)
        # legal_moves_observations
        observation.append(self.legal_jobs)
        observation.append(self.legal_machines)
        # progress_observations
        observation.append(self.jobs_completion)
        # production_time_observations
        observation.append(self.job_processing_time)
        observation.append(self.job_changeover_time)
        observation.append(self.job_waiting_time)
        # time_until_observations
        observation.append(self.time_until_job_done)
        observation.append(self.time_until_machine_free)
        # machines_observations
        observation.append(self.time_machines_idle)
        observation.append(self.machine_queue_list)
        # observation = np.ndarray.flatten(np.array(observation))

        observation = list(itertools.chain.from_iterable(observation))

        # objective_variables_observations
        observation.insert(0, self.sim_duration)
        observation.insert(1, self.oc_costs)
        observation.insert(2, self.weighted_total_lateness)

        observation = np.array(observation)

        return observation
