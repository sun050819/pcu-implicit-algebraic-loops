"""(4) Engineering acceleration layer: thread pool/process pool + priority control."""
from .pool import map_parallel, PriorityPool, n_cpu, make_node_fn_spec, build_node_fn_from_spec
