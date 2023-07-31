import collections,re
import pandas as pd

from main import GENERAL_DATA
from flexible_flow_shop.resources.functions.scheduling_functions import obtain_machine_requested_per_stage
from main import filename, buffer_usage, recreate_solution,action_space, workbook,solution_hints
import itertools
from itertools import chain, repeat
if recreate_solution != None or (workbook == "kopanos" and action_space == "discrete_probs" and solution_hints  != None ):
    from main import SOLUTION

MAKESPAN = list()
TASK_LIST = list()
FINISHED_ORDERS = list()
PROCESSING_TIMES = pd.read_excel(filename, sheet_name="processing_times")
MACHINES_PER_STAGE = pd.read_excel(filename, sheet_name="machines")
NUM_MACHINES_PER_STAGE = list(collections.Counter(MACHINES_PER_STAGE["stage"].tolist()).values())
IMPACT_FACTORS = pd.read_excel(filename, sheet_name="impact_factors")

if buffer_usage != "no_buffers":
    BUFFERS_PER_STAGE = pd.read_excel(filename, sheet_name="buffer_tanks")
    NUM_BUFFERS_PER_STAGE = list(collections.Counter(BUFFERS_PER_STAGE["stages"].tolist()).values())
    BUFFERS = list(BUFFERS_PER_STAGE.buffer)
else:
    BUFFERS_PER_STAGE = None
    NUM_BUFFERS_PER_STAGE = None
    BUFFERS = None

MACHINES = list(MACHINES_PER_STAGE.machine)
if BUFFERS:
    RESOURCES = (MACHINES+BUFFERS)
else:
    RESOURCES = MACHINES

STAGES = range(len(NUM_MACHINES_PER_STAGE))

def create_orders_from_general_data_workbook():
    flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    a = GENERAL_DATA.quantity.tolist()
    b = GENERAL_DATA.product_code.tolist()
    c = list(chain.from_iterable([x] * y for x, y in zip(b, a)))
    d = list(range(len(c))) #order_id
    e = len(MACHINES_PER_STAGE.machine)
    f = [d, c]
    g = list(map(list, zip(*f)))
    h = [x for item in g for x in repeat(item,e)]
    i = list(MACHINES_PER_STAGE.machine)
    j = [list(x) for x in zip(h,itertools.cycle(i))]
    k = list(range(len(h)))
    l = [k,j]
    m = list(map(list, zip(*l)))
    n = []
    for index in k:
        sublist=list(flatten(m[index]))
        n.append(sublist) #operation_id
    df = pd.DataFrame.from_records(n)
    df.columns = ['operation_id', 'order_id', 'product_code',"machine"]
    return df

def set_invalid_actions_workbook():
    df = ORDERS[['product_code', 'machine']]
    result = [tuple(r) for r in df.to_numpy()]
    list_of_valid_actions = []
    for item in result:
        product = item[0]
        product_id = int(re.search("\d+",product)[0])-1
        machine = item[1]

        value = PROCESSING_TIMES.loc[product_id,machine]
        if value == "-":
            list_of_valid_actions.append(0)
        else:
            list_of_valid_actions.append(1)

    valid_actions = pd.DataFrame({'valid': list_of_valid_actions})
    orders_with_invalid = ORDERS.join(valid_actions)
    return orders_with_invalid

if recreate_solution != None or (workbook == "kopanos" and action_space == "discrete_probs" and solution_hints != None):
    ORDERS = SOLUTION
else:
    ORDERS = create_orders_from_general_data_workbook()
ORDERS = ORDERS.merge(MACHINES_PER_STAGE, on=['machine'], how="left")
ORDERS = set_invalid_actions_workbook()
JOBS = list(set(list(ORDERS.order_id)))


def get_total_visited_stages_workbook():
    df = ORDERS[['order_id', 'stage','valid']]
    df = df[df["valid"] == 1]
    result = [tuple(r) for r in df.to_numpy()]
    set_ = list(set(result))
    TOTAL_VISITED_STAGES =  [ 0 for _ in range(len(JOBS)) ]
    for item in set_:
        TOTAL_VISITED_STAGES[item[0]] += 1
    return TOTAL_VISITED_STAGES

def get_stage_of_machine_workbook():
    df = ORDERS[['operation_id', 'machine']]
    result = [tuple(r) for r in df.to_numpy()]
    set_ = list(set(result))
    N_OPERATIONS = len(set_)
    TOTAL_JOBS_PER_STAGE = [0 for _ in range(len(STAGES))]
    for item in set_:
        TOTAL_JOBS_PER_STAGE[item[1]] += 1
    return TOTAL_JOBS_PER_STAGE, N_OPERATIONS


def get_jobs_per_stage_workbook():
    # Get the number of orders per each stage and store in variable TOTAL_JOBS_PER_STAGE
    df = ORDERS[['order_id', 'stage','valid']]
    df = df[df["valid"] == 1]
    result = [tuple(r) for r in df.to_numpy()]
    set_ = list(set(result))
    N_OPERATIONS = len(set_)
    TOTAL_JOBS_PER_STAGE =  [ 0 for _ in range(len(STAGES)) ]
    for item in set_:
        TOTAL_JOBS_PER_STAGE[item[1]] += 1
    return TOTAL_JOBS_PER_STAGE, N_OPERATIONS

def get_index_machines_workbook():
    #INDEX MACHINES will be used to generate the variable machines_per_stage
    df = MACHINES_PER_STAGE[['stage', 'machine_id']]
    result = [tuple(r) for r in df.to_numpy()]
    set_ = list(set(result))
    INDEX_MACHINES = [ [] for _ in range(len(STAGES)) ]
    for item in set_:
        INDEX_MACHINES[item[0]-1].append(item[1])
    INDEX_MACHINES =  sorted([sorted(item) for item in INDEX_MACHINES])
    return INDEX_MACHINES

INDEX_MACHINES = get_index_machines_workbook()
TOTAL_VISITED_STAGES = get_total_visited_stages_workbook()
TOTAL_JOBS_PER_STAGE, N_OPERATIONS = get_jobs_per_stage_workbook()

def read_changeover_times(STAGES):
    """Function that stores ALL the changeover times excel sheets into
    a list. This was done so that all the changeovers are accessed in memory
    and not accessed at each timestep in the workbook"""
    CHANGEOVER = list()
    for i in STAGES:
        stage = i + 1
        CHANGEOVER.append(
            pd.read_excel(filename, sheet_name=f"CH_S{stage}")
        )
    for i in range(len(CHANGEOVER)):
        CHANGEOVER[i].set_index("Current order", inplace=True)

    return CHANGEOVER

CHANGEOVER = read_changeover_times(STAGES)
ID_MACHINES_PER_STAGE = obtain_machine_requested_per_stage(STAGES, MACHINES_PER_STAGE)
