import numpy as np
import pandas as pd
import os, re


def assign_position_in_schedule(self, order, N_OPERATIONS, MACHINES):
    order.position_in_schedule = self.counter
    self.scheduled_operations[order.position_in_schedule] = order.operation_id
    self.scheduled_machines[order.position_in_schedule] = MACHINES.index(order.machine)
    self.scheduled_jobs[order.position_in_schedule] = order.order_id
    if self.counter < (N_OPERATIONS - 2):
        self.counter += 1


def prepare_historical_and_flags(self, order, machine, action):
    if self.counter_historical_selection[machine] != 0:
        self.historical_selection[machine].append(action)
    self.counter_historical_selection[machine] += 1
    # Flag the order as illegal before sending to production
    self.legal_operations[
        action
    ] = False  # this will be kept as False once is sent to schedule
    self.legal_jobs[
        order.order_id
    ] = False  # this will alternate between T/F as the scheduling flag for the agent


def legalize_with_release_dates(self):
    for i, order in enumerate(self.orders):
        if (
            order.release_date >= self.sim_duration
            and order.stage == 0
            and order.scheduled_by_agent == 0
            and order.valid == 1
        ):
            self.legal_operations[order.operation_id] = True


def get_orders_next_stage(self, order, STAGES):
    if order.stage != len(STAGES) - 1:
        for i in range(len(self.orders)):
            if (
                self.orders[i].order_id == order.order_id
                and self.orders[i].scheduled == 0
                and self.orders[i].stage > order.stage
                and self.orders[i].valid == 1
            ):
                next_stage = self.orders[i].stage
                break

        next_stage = self.orders[i].stage
        return next_stage


def flag_as_legals_in_next_stage(self, order, STAGES):
    if order.stage != len(STAGES) - 1:
        for j in range(len(self.orders)):
            order_next_stage = self.orders[j]
            if (
                order_next_stage.order_id == order.order_id
                and self.orders[j].scheduled == 0
                and order_next_stage.stage == order.next_stage
                and order_next_stage.valid == 1
            ):
                self.legal_operations[order_next_stage.operation_id] = True
                self.legal_jobs[order_next_stage.order_id] = True


def disable_operations_of_same_job_in_same_stage(self, order):
    for j in range(len(self.orders)):
        order_same_stage = self.orders[j]
        if (
            order_same_stage.order_id == order.order_id
            and order_same_stage.stage == order.stage
            and order_same_stage.machine != order.machine
        ):
            self.legal_operations[order_same_stage.operation_id] = False

def check_if_can_be_legal_again(self, order):

    order_list = []
    for j in range(len(self.orders)):
        order_same_stage = self.orders[j]
        if (
            order_same_stage.order_id == order.order_id
            and order_same_stage.stage == order.stage
            and order_same_stage.machine != order.machine
            and order_same_stage.scheduled_by_agent == True
        ):
            order_list.append(order_same_stage.operation_id)
    if len(order_list) > 0:
        return False

    else:
        return True
def update_time_arrays(self):
    # Substract the current simulation time to the leftover time jobs array
    self.time_until_job_done = np.array(
        [x - self.timestep for x in self.time_until_job_done]
    )  # substract env.now
    self.time_until_operation_done = np.array(
        [x - self.timestep for x in self.time_until_operation_done]
    )  # substract env.now
    self.time_until_machine_free = np.array(
        [x - self.timestep for x in self.time_until_machine_free]
    )  # substract env.now

def get_tassel_reward(self,action):

    if action == len(self.study.ORDERS) or not(self.legal_operations[action]):
        # If action is NOOP, consider holes in all machines.
        tassel_processing_time = 0
        tassel_machines_holes = sum(np.array(self.time_machines_idle_per_stage).sum())
        tassel_reward = tassel_processing_time - tassel_machines_holes

    elif self.machine_queue[self.orders[action].machine] >= self.MAX_QUEUE_SIZE:
        # If action is legal, but machine is busy and queue is unavailable, consider holes from the corresponding stage
        # plus remaining time until operation can enter the queue. This operation won't enter the schedule
        stage = self.orders[action].stage
        machine = self.orders[action].machine_id
        tassel_processing_time = 0
        tassel_machines_holes = sum(self.time_machines_idle_per_stage[stage]) + self.time_until_machine_free[machine]
        tassel_reward = tassel_processing_time - tassel_machines_holes
    else:
        # if action is legal and machine or queue are directly available
        stage = self.orders[action].stage
        machine = self.orders[action].machine_id
        tassel_processing_time = self.orders[action].processing_time
        tassel_machines_holes = sum(self.time_machines_idle_per_stage[stage]) - self.time_machines_idle[machine]
        tassel_reward = tassel_processing_time - tassel_machines_holes

    return tassel_reward

def set_time_machines_idle(self):

    for stage, machines in enumerate(self.study.INDEX_MACHINES):
        for machine, i in enumerate(machines):
            if self.stage_progression[stage] < 0.999:
                if self.legal_machines[i] and self.stage_visited_indicator[stage]:  # if the i machine is not active and the stage has started producing, then add up idle time
                    self.time_machines_idle_per_stage[stage][machine] += self.timestep
                else:
                    self.time_machines_idle_per_stage[stage][machine] = 0
            else:
                self.time_machines_idle_per_stage[stage][machine] = 0
def get_timestep(self, action):
    round_value = 3
    self.time_until_job_done = np.around(
        np.array(self.time_until_job_done.clip(min=0)), round_value
    )  # make negative values equal to zero
    self.time_until_operation_done = np.around(
        np.array(self.time_until_operation_done.clip(min=0)), round_value
    )  # make negative values equal to zero
    self.time_until_machine_free = np.around(
        np.array(self.time_until_machine_free.clip(min=0)), round_value
    )  # make negative values equal to zero

    if self.time_until_operation_done.any() == False:  # needed in first timestep
        timestep = 0.01

    elif action == len(self.orders):
        min_non_zero = np.min(
            self.time_until_operation_done[np.nonzero(self.time_until_operation_done)]
        )
        timestep = min_non_zero
        # print("NO OP: Go to the next time step until we have a legal action! Jump: {}".format(timestep))

    elif self.orders[action].stage == 0:
        timestep = 0.01

    elif self.legal_machines_per_stage[self.orders[action].stage].any() == True:
        timestep = 0.01

    else:
        min_non_zero = np.min(
            self.time_until_operation_done[np.nonzero(self.time_until_operation_done)]
        )
        timestep = min_non_zero
        # print("ELSE: Go to the next time step! Jump: {}".format(timestep))

    return timestep


def generate_results(
    orders,
    ORDERS,
    episode_makespan,
    FINISHED_ORDERS,
    generate_heuristic_schedules,
    test,
    experiment_folder,
):
    """Function that generates the results of the experiment and stores them
    into an Excel Workbook, whose name changes dynamically based on the
    experiment date."""
    values_of_interest = [None for i in range(len(ORDERS))]
    df_list = []
    for i in range(len(ORDERS)):
        values_of_interest[i] = [
            orders[i].operation_id,
            orders[i].order_id,
            orders[i].stage,  # stages
            orders[i].product_code,  # scalar product_code
            orders[i].time_arriving_stage,  # time arriving
            orders[i].machine,  # allocated in machine
            orders[i].time_changeover_starts,  # time changeover in
            orders[i].time_changeover_ends,  # time changeouver out
            orders[i].time_start_process,  # time process in
            orders[i].time_leaving_stage,  # time process out
            orders[i].processing_value,  # process value
            orders[i].waiting_value,  # waiting value
            orders[i].changeover_value,  # changeover value
            orders[i].visited_stage,  # scalar done value
            orders[i].impact_factor_value,
            orders[i].changeover_cost,
            orders[i].earliness,
            orders[i].tardiness,
            orders[i].weighted_lateness,
        ]

    for i in range(len(values_of_interest)):
        df_values_of_interest = pd.DataFrame(values_of_interest[i])
        df_list.append(df_values_of_interest.T)

    df_values_of_interest = pd.concat(df_list)
    df_values_of_interest.columns = [
        "operation_id",
        "order_id",
        "stage",
        "product",
        "time_arriving_stage",
        "machine_requested",
        "time_changeover_starts",
        "time_changeover_ends",
        "time_start_process",
        "time_leaving_stage",
        "processing_value",
        "waiting_value",
        "changeover_value",
        "visited_stages",
        "impact_factor_value",
        "changeover_cost",
        "earliness",
        "tardiness",
        "weighted_lateness",
    ]
    df_values_of_interest.sort_values(by=["time_arriving_stage"], inplace=True)
    if generate_heuristic_schedules != None:
        filename_values_of_interest = (
            "outputs/{}/{}/render/{}_Mks_{}_scheduling_results.xlsx".format(
                experiment_folder, test, generate_heuristic_schedules, episode_makespan
            )
        )
    else:
        filename_values_of_interest = (
            "outputs/{}/{}/render/Mks_{}_scheduling_results.xlsx".format(
                experiment_folder, test, episode_makespan
            )
        )

    os.makedirs(os.path.dirname(filename_values_of_interest), exist_ok=True)
    df_values_of_interest.to_excel(
        filename_values_of_interest, sheet_name="scheduling_results", index=False
    )

    # df_finished_orders = pd.DataFrame(FINISHED_ORDERS)
    # df_finished_orders.columns = ["finished_orders"]
    # filename_finished_orders = "outputs/{}/render/Rew_{}_Mks_{}_finished_orders.xlsx".format(test,episode_reward,episode_length)
    # df_finished_orders.to_excel(filename_finished_orders, sheet_name="finished_orders", index=False)


def processing_times(self, env, PROCESSING_TIMES, order, machine_idx):
    """Function that extracts the processing times from the in-memory saved Excel Worksheet"""
    current_machine = order.machine
    order.visited_stage += 1
    product_id = int(re.search("\d+", order.product_code)[0]) - 1
    process_time = PROCESSING_TIMES.loc[product_id, current_machine]
    self.time_until_operation_done[order.position_in_schedule] = process_time
    self.time_until_job_done[order.order_id] = process_time
    self.time_until_machine_free[machine_idx] = process_time
    self.job_processing_time[order.order_id] += process_time
    self.operation_processing_time[order.position_in_schedule] = process_time
    return process_time


def changeover_times(self, env, changeover_datasheet, order, next_order):
    """Function that extracts the CHANGEOVER times from the in-memory saved Excel Worksheet"""
    changeover_time = changeover_datasheet.loc[
        order.product_code, next_order.product_code
    ]  # look up for the CHANGEOVER time of the given product and the next product
    self.job_changeover_time[order.order_id] += changeover_time
    self.operation_changeover_time[order.position_in_schedule] = changeover_time
    return changeover_time


def get_impact_factor(IMPACT_FACTORS, order, next_order):
    product_id = int(re.search("\d+", order.product_code)[0]) - 1
    impact_factor = IMPACT_FACTORS.loc[product_id, next_order.product_code]
    return impact_factor


def get_machine_idx(machine, MACHINES):
    index = MACHINES.index(machine)
    return index


def current_variables(self):
    self.job_completion_current = np.sum(self.jobs_completion)
    self.oc_costs_current = self.oc_costs
    self.wl_current = self.weighted_total_lateness
    self.sim_duration_current = self.sim_duration
    self.episode_length_current = self.episode_length


def past_variables(self):
    # variables to store objective values before applying action
    self.oc_costs_past = self.oc_costs_current
    self.wl_past = self.wl_current
    self.sim_duration_past = self.sim_duration_current
    self.episode_length_past = self.episode_length_current
    self.job_completion_past = self.job_completion_current


def generate_schedule_first_stage(
    env, current_legal_operations, counter, generate_heuristic_schedules, heuristics_policy_rl, CHANGEOVER
):

    a = list(
        np.arange(0, 495, 17)
    )  # ACTION INDICES FOR FIRST STAGE OPERATIONS IN MACHINE 1
    b = list(
        np.arange(1, 496, 17)
    )  # ACTION INDICES FOR FIRST STAGE OPERATIONS IN MACHINE 2

    if not (counter % 2):
        first_stage_operations = sorted(list(set(a) & set(current_legal_operations)))
    else:
        first_stage_operations = sorted(list(set(b) & set(current_legal_operations)))

    if first_stage_operations:
        first_stage_in_progress = 1
        if generate_heuristic_schedules == "FIFO" or heuristics_policy_rl == "FIFO":
            action = np.random.choice(current_legal_operations)
            action = search_actions_FIFO(env,action, first_stage_operations,heuristics_policy_rl)
        elif generate_heuristic_schedules == "EDD" or heuristics_policy_rl == "EDD":
            action = search_actions_EDD(env, first_stage_operations,heuristics_policy_rl)
        elif generate_heuristic_schedules == "SPT" or heuristics_policy_rl == "SPT":
            action = search_actions_SPT(env, first_stage_operations,heuristics_policy_rl)
        elif generate_heuristic_schedules == "SCT" or heuristics_policy_rl == "SCT":
            action = search_actions_SCT(env, first_stage_operations, CHANGEOVER,heuristics_policy_rl)
    else:
        first_stage_in_progress = 0
        action = None

    return action, first_stage_in_progress, counter


def search_actions_FIFO(env, action, operations,heuristics_policy_rl):
    new_actions_machine_free = []
    new_actions_machine_busy = []
    order = env.orders[action]

    for _, new_action in enumerate(operations):
        new_order = env.orders[new_action]
        if (
            new_order.order_id == order.order_id
            and new_order.stage == order.stage
            and new_order.valid == 1
        ):
            new_possible_action = new_order.operation_id
            if env.legal_machines[env.orders[new_possible_action].machine_id] == 1:
                new_actions_machine_free.append(new_possible_action)
            else:
                new_actions_machine_busy.append(new_possible_action)

    if heuristics_policy_rl == None:
        if new_actions_machine_free != []:
            action = np.random.choice(new_actions_machine_free)
        else:
            action = np.random.choice(new_actions_machine_busy)
    else:
        if new_actions_machine_free != []:
            action = new_actions_machine_free
        else:
            action = new_actions_machine_busy
    return action


def search_actions_EDD(env, operations,heuristics_policy_rl):
    new_order_dd = []
    new_order_id = []
    new_order_ids = []
    for _, new_action in enumerate(operations):
        new_order = env.orders[new_action]
        new_order_dd.append(new_order.due_date)
        new_order_id.append(new_order.operation_id)

    min_due_date_idx = [
        i for i, x in enumerate(new_order_dd) if x == np.min(new_order_dd)
    ]
    for i, value in enumerate(min_due_date_idx):
        new_order_ids.append(new_order_id[value])

    if heuristics_policy_rl == None:
        action = np.random.choice(new_order_ids)
    else:
        action = new_order_ids
    return action


def search_actions_SPT(env, operations,heuristics_policy_rl):
    new_order_ids = []

    # actions with free machines
    new_orders_processing_times = []
    new_order_id = []

    # actions with busy machines
    new_order_id_busy = []

    for _, new_action in enumerate(operations):
        new_order = env.orders[new_action]
        if env.legal_machines[new_order.machine_id] == 1:
            new_orders_processing_times.append(new_order.processing_time)
            new_order_id.append(new_order.operation_id)
        else:
            new_order_id_busy.append(new_order.operation_id)

    if new_orders_processing_times != []:
        min_processing_times_idx = [
            i
            for i, x in enumerate(new_orders_processing_times)
            if x == np.min(new_orders_processing_times)
        ]
        for i, value in enumerate(min_processing_times_idx):
            new_order_ids.append(new_order_id[value])

        if heuristics_policy_rl == None:
            action = np.random.choice(new_order_ids)
        else:
            action = new_order_ids


    else:
        if heuristics_policy_rl == None:
            action = np.random.choice(new_order_id_busy)
        else:
            action = new_order_id_busy
    return action


def search_actions_SCT(env, operations, CHANGEOVER,heuristics_policy_rl):
    new_order_ids = []

    # actions with free machines
    new_orders_changeover_times = []
    new_order_id = []

    # actions with busy machines
    new_orders_changeover_times_busy = []
    new_order_id_busy = []

    for _, new_action in enumerate(operations):
        new_order = env.orders[new_action]
        if env.current_product_in_machine[new_order.machine_id] != []:
            current_product_in_machine = env.current_product_in_machine[
                new_order.machine_id
            ][-1]
            new_order.changeover = CHANGEOVER[new_order.stage].loc[
                current_product_in_machine, new_order.product_code
            ]
        else:
            new_order.changeover = 0

        if env.legal_machines[new_order.machine_id] == 1:
            new_orders_changeover_times.append(new_order.changeover)
            new_order_id.append(new_order.operation_id)
        else:
            new_orders_changeover_times_busy.append(new_order.changeover)
            new_order_id_busy.append(new_order.operation_id)

    if new_orders_changeover_times != []:
        min_changeover_times_idx = [
            i
            for i, x in enumerate(new_orders_changeover_times)
            if x == np.min(new_orders_changeover_times)
        ]
        for i, value in enumerate(min_changeover_times_idx):
            new_order_ids.append(new_order_id[value])

    else:
        min_changeover_times_idx = [
            i
            for i, x in enumerate(new_orders_changeover_times_busy)
            if x == np.min(new_orders_changeover_times_busy)
        ]
        for i, value in enumerate(min_changeover_times_idx):
            new_order_ids.append(new_order_id_busy[value])

    # print("CHOOSE FROM: {} or {}".format(new_orders_changeover_times,new_orders_changeover_times_busy))
    if heuristics_policy_rl == None:
        action = np.random.choice(new_order_ids)
    else:
        action = new_order_ids
    # print("CHANGEOVER SELECTED: {}".format(env.orders[action].changeover))
    return action


def search_options_with_available_machines(
    env, action, current_legal_operations, generate_heuristic_schedules, heuristics_policy_rl, CHANGEOVER
):
    if generate_heuristic_schedules == "FIFO":
        action = search_actions_FIFO(env, action, current_legal_operations,heuristics_policy_rl)

    elif generate_heuristic_schedules == "EDD":
        action = search_actions_EDD(env, current_legal_operations,heuristics_policy_rl)

    elif generate_heuristic_schedules == "SPT":
        action = search_actions_SPT(env, current_legal_operations,heuristics_policy_rl)

    elif generate_heuristic_schedules == "SCT":
        action = search_actions_SCT(env, current_legal_operations, CHANGEOVER,heuristics_policy_rl)

    return action


def get_action_heuristics(
    env, current_legal_operations, counter, generate_heuristic_schedules, heuristics_policy_rl, CHANGEOVER
):
    action, first_stage_in_progress, counter = generate_schedule_first_stage(
        env, current_legal_operations, counter, generate_heuristic_schedules, heuristics_policy_rl, CHANGEOVER
    )

    if first_stage_in_progress:
        counter += 1
    else:
        if len(current_legal_operations) != 0:
            action = np.random.choice(current_legal_operations)
        else:
            action = 510
        action = search_options_with_available_machines(
            env,
            action,
            current_legal_operations,
            generate_heuristic_schedules,
            heuristics_policy_rl,
            CHANGEOVER,
        )
    return action, counter


def GeneratePreliminaryHeuristicResultsFiles(
    self,
    filename,
    generate_heuristic_schedules,
    episode_makespan,
    episode_oc_costs,
    episode_wl,
):
    with open(
        "{}/{}_Mks_{}_OCC_{}_WL_{}_Sequences.txt".format(
            filename,
            generate_heuristic_schedules,
            episode_makespan,
            episode_oc_costs,
            episode_wl,
        ),
        "w",
    ) as f:
        for i in range(len(self.heuristics_products_per_stage)):
            print(
                "Stage {}: {}".format(i, self.heuristics_products_per_stage[i]), file=f
            )
    with open(
        "{}/{}_Mks_{}_OCC_{}_WL_{}_Solution.txt".format(
            filename,
            generate_heuristic_schedules,
            episode_makespan,
            episode_oc_costs,
            episode_wl,
        ),
        "w",
    ) as f:
        print("Solution in this episode: {}".format(self.heuristics_solution), file=f)


def GenerateHeuristicResultsFiles(
    data, filename, generate_heuristic_schedules, info_heuristics
):
    with open(
        "{}/{}_Summary.txt".format(filename, generate_heuristic_schedules), "w"
    ) as f:
        for key, value in info_heuristics.items():
            print("{}: {}".format(key, value), file=f)
