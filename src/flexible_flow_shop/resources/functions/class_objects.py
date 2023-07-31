import simpy
import numpy as np
from flexible_flow_shop.resources.functions.global_variables import NUM_MACHINES_PER_STAGE
from flexible_flow_shop.resources.functions.global_variables import NUM_BUFFERS_PER_STAGE
from flexible_flow_shop.resources.functions.global_variables import GENERAL_DATA
from flexible_flow_shop.resources.functions.global_variables import PROCESSING_TIMES
from flexible_flow_shop.resources.functions.global_variables import ID_MACHINES_PER_STAGE
from flexible_flow_shop.resources.functions.global_variables import STAGES
from flexible_flow_shop.resources.functions.global_variables import TOTAL_VISITED_STAGES
from main import buffer_usage

class Order:

    """A class that helps me to define and create an Order object.
    The class variables will be defined by the Master Excel Workbook that
    is contained in the root directory of this repository. We have the product_code
    and the due_date as class variables. On the other hand, the remaining instance variables
    are auxiliary in the monitoring of the scheduling process. These instance variables are later
    used in the code to create an Excel Workbook of results."""

    def __init__(self,env,operation_id,order_id, machine,machine_id, product_code,stage,valid):
        self.env = env
        self.order_id = order_id
        self.valid = valid
        self.machine = machine
        self.machine_id = machine_id
        self.product_code = product_code
        self.operation_id = operation_id
        self.stage = stage
        self.total_stages = TOTAL_VISITED_STAGES[self.order_id]
        product_index_gd = GENERAL_DATA[GENERAL_DATA['product_code'] == self.product_code].index.values.astype(int)[0]
        self.due_date = GENERAL_DATA.due_date[product_index_gd]
        self.release_date = GENERAL_DATA.release_date[product_index_gd]
        self.processing_time = PROCESSING_TIMES[machine][product_index_gd]
        self.changeover = None
        self.visited_stage = 0
        self.impact_factor_value = 0
        self.changeover_cost = 0
        self.earliness = 0
        self.tardiness = 0
        self.weighted_lateness = 0
        self.visited_buffer = False
        self.buffer = None
        self.go_to_buffer = None
        self.time_in_buffer = None
        self.position_in_schedule = None
        self.scheduled_by_agent = 0
        #logging variables
        self.changeover_time = 0
        self.buffer_changeover = 0
        self.time_arriving_stage = None
        self.time_start_changeover_buffer = None
        self.time_start_buffer = None
        self.time_end_buffer = None
        self.time_changeover_starts = None
        self.time_changeover_ends = None
        self.time_start_process = None
        self.time_leaving_stage = None
        self.waiting_value = None
        self.changeover_value = None
        self.processing_value = None
        self.scheduled = False
        self.next_order = None
        self.changeover_in_progress = False
        self.time_end_changeover_buffer = None
        self.time_start_block = None
        self.time_end_block = None
        self.position_heuristics = None
        self.next_stage = None
class Factory(object):

    """A class that helps me to define and create a SimPy Factory Object.
    Here the stages, as well as the machines contained in each stage
    are being defined. Stage here is a FilterStore, whose items are
    the machines."""

    def __init__(self, env):
        self.env = env
        self.stage = [simpy.FilterStore(env, NUM_MACHINES_PER_STAGE[i]) for i in STAGES]
        for i in STAGES:
            self.stage[i].items = ID_MACHINES_PER_STAGE[i].copy()

        if buffer_usage != "no_buffers":
            self.buffer = [simpy.Resource(env, capacity=np.inf) for i in range(len(NUM_BUFFERS_PER_STAGE))]
            #up to 10 orders of the same product can be stored in the buffer
        self.finished_orders = []