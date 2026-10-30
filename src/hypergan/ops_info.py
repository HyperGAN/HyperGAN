"""HNDL operator introspection and catalog."""
from typing import Any, Dict, List, Optional


def get_operators(query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return catalog of all 85 HNDL operators, optionally filtered by search query.
    
    Args:
        query: Optional case-insensitive search string matching name, category, or summary.
        
    Returns:
        List of operator documentation records.
    """
    from hndl import Registry
    from .hndl_augmentation import register_augmentation

    reg = register_augmentation(Registry.builtins())
    ops: List[Dict[str, Any]] = []

    for op in reg.operators:
        name = getattr(op, "alias", None) or getattr(op, "identity", "")
        category = getattr(op, "category", "") or "general"
        summary = getattr(op, "summary", "") or ""
        shape_text = getattr(op, "shape_text", "") or ""
        doc = getattr(op, "doc", "") or ""

        inputs = [str(p) for p in (op.inputs or ())]
        outputs = [str(p) for p in (op.outputs or ())]

        args_dict: Dict[str, Any] = {}
        if op.args:
            for arg_name, arg in op.args.items():
                arg_type = getattr(arg, "type", None)
                type_str = arg_type.__name__ if hasattr(arg_type, "__name__") else str(arg_type)
                default_val = getattr(arg, "default", None)
                if default_val is not None and type(default_val) is object:
                    default_val = None
                elif default_val is not None and not isinstance(default_val, (str, int, float, bool, list, dict)):
                    default_val = str(default_val)
                args_dict[arg_name] = {
                    "type": type_str,
                    "help": getattr(arg, "help", "") or "",
                    "positional": getattr(arg, "positional", False),
                    "default": default_val,
                }

        record = {
            "name": name,
            "category": category,
            "summary": summary,
            "shape": shape_text,
            "inputs": inputs,
            "outputs": outputs,
            "args": args_dict,
            "doc": doc,
        }

        if query:
            q = query.strip().lower()
            if q not in name.lower() and q not in category.lower() and q not in summary.lower():
                continue

        ops.append(record)

    ops.sort(key=lambda item: (item["category"], item["name"]))
    return ops
