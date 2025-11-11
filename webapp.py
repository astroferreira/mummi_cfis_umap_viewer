from flask import Flask, render_template, request, jsonify, abort
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd

app = Flask(__name__)

# Constants
DEFAULT_COLOR_COLUMN = "nFit2D"
DEFAULT_FILTER = 1
DEFAULT_LOG_SCALE = True
VALID_FILTERS = {1, 2, 3, 4, 5}
TRUE_VALUES = {'1', 'true', 'yes', 'on'}
CFIS_NEIGHBORS = 100
MOCK_NEIGHBORS = 100


# Pre-load data needed by the UMAP view and nearest-neighbour lookups
FEATURES_CFIS = np.load('data/UMAPS/feature_explorer/cfis_reduce.npy')
FEATURES_MOCK = np.load('data/UMAPS/feature_explorer/mock_survey_reduced.npy')
METADATA = pd.read_pickle('data/UMAPS/mock_survey_meta.pk')
DR7_TABLE = pd.read_pickle('data/UMAPS/DR7_with_UMAP.pk').reset_index(drop=True)

if len(DR7_TABLE) != len(FEATURES_CFIS):
    raise ValueError("DR7 table length does not match CFIS feature matrix.")

if 'cluster_id' not in DR7_TABLE.columns:
    DR7_TABLE['cluster_id'] = 0

# Ensure cluster ids are integers and compute global cluster count
DR7_TABLE['cluster_id'] = DR7_TABLE['cluster_id'].astype(int)
CLUSTER_COUNT = max(1, int(DR7_TABLE['cluster_id'].nunique()))
if DEFAULT_COLOR_COLUMN in DR7_TABLE.columns:
    DEFAULT_COL_INDEX = int(DR7_TABLE.columns.get_loc(DEFAULT_COLOR_COLUMN))
else:
    DEFAULT_COL_INDEX = 0

FEATURES_TREE_CFIS = KDTree(FEATURES_CFIS)
FEATURES_TREE_MOCK = KDTree(FEATURES_MOCK)


def _apply_filter(frame: pd.DataFrame, filter_value: int) -> pd.DataFrame:
    """Return a filtered copy of the DR7 table for the requested subset."""
    if filter_value == 4:
        return frame.query('STAGE1_votes_12RP==20 & STAGE2_prob_12RP > 0.5').copy()
    if filter_value == 5:
        return frame.query('STAGE1_votes_12RP>10 & STAGE2_prob_12RP > 0.5').copy()
    return frame.copy()


def _get_int_arg(name: str, default: int, *, valid_values=None) -> int:
    """Parse an integer query parameter, falling back to a default value."""
    value = request.args.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if valid_values is not None and parsed not in valid_values:
        return default
    return parsed


def _get_bool_arg(name: str, default: bool = False) -> bool:
    value = request.args.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in TRUE_VALUES


def _transform_color(values: np.ndarray, column_name: str, log_scale: bool):
    """Optionally apply log10 scaling to the supplied column values."""
    column_values = values.astype(float)
    colorbar_label = column_name
    if log_scale:
        positive_mask = column_values > 0
        if positive_mask.any():
            min_positive = column_values[positive_mask].min()
            column_values = column_values.copy()
            column_values[~positive_mask] = min_positive
            column_values[column_values <= 0] = min_positive
            column_values = np.log10(column_values)
            colorbar_label = f"log10({column_name})"
        else:
            log_scale = False
    return column_values, colorbar_label, log_scale


def _resolve_axis_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise RuntimeError(f"None of the candidate columns {candidates} exist in the dataframe.")


@app.route("/umap")
def UMAP():
    col_index = _get_int_arg('col_index', DEFAULT_COL_INDEX)
    filter_value = _get_int_arg('filter', DEFAULT_FILTER, valid_values=VALID_FILTERS)
    log_scale = _get_bool_arg('log_scale', DEFAULT_LOG_SCALE)

    dr7_frame = _apply_filter(DR7_TABLE, filter_value)
    if dr7_frame.empty:
        abort(404, description="No data available for the requested filter.")

    total_columns = len(dr7_frame.columns)
    if not 0 <= col_index < total_columns:
        abort(400, description=f"Column index {col_index} is out of bounds (0-{total_columns-1}).")

    column = dr7_frame.columns[col_index]
    column_values, colorbar_label, log_scale = _transform_color(
        dr7_frame[column].values.astype(float),
        column,
        log_scale,
    )

    values = column_values.tolist()
    color_min = float(np.nanmin(column_values))
    color_max = float(np.nanmax(column_values))
    opacity = ['0.1'] * dr7_frame.shape[0]

    x_column = _resolve_axis_column(dr7_frame, ["UMAP_dim1", "UMAP_x", "UMAP1"])
    y_column = _resolve_axis_column(dr7_frame, ["UMAP_dim2", "UMAP_y", "UMAP2", "UMAP"])

    return render_template(
        "umap.html",
        columns=dr7_frame.columns.tolist(),
        ids=dr7_frame['objID'].values.astype(str).tolist(),
        zip=zip,
        x=dr7_frame[x_column].values.astype(float).tolist(),
        y=dr7_frame[y_column].values.astype(float).tolist(),
        s=values,
        opacity=opacity,
        filter=filter_value,
        col_index=col_index,
        col_name=column,
        log_scale=log_scale,
        colorbar_label=colorbar_label,
        color_min=color_min,
        color_max=color_max,
        cluster_labels=dr7_frame['cluster_id'].astype(int).tolist(),
        cluster_count=CLUSTER_COUNT,
    )


@app.route("/stream")
def stream():
    object_id = request.args.get('id')
    if object_id is None:
        abort(400, description="Missing required parameter 'id'.")
    try:
        object_id = int(object_id)
    except (TypeError, ValueError):
        abort(400, description="Parameter 'id' must be an integer.")

    filter_value = _get_int_arg('filter', DEFAULT_FILTER, valid_values=VALID_FILTERS)

    match = DR7_TABLE.loc[DR7_TABLE.objID == object_id]
    if match.empty:
        abort(404, description=f"Object ID {object_id} was not found.")

    vec = FEATURES_CFIS[match.index]

    _, idx = FEATURES_TREE_CFIS.query(vec.reshape(1, FEATURES_CFIS.shape[1]), k=CFIS_NEIGHBORS)

    if filter_value == 1:
        ids = DR7_TABLE.objID.values.astype(str)[idx][0].tolist()
    elif filter_value == 3:
        ids = DR7_TABLE.iloc[idx[0]].query('STAGE1_votes_12RP == 20 and STAGE2_prob_12RP < 0.5').objID.values.astype(str).tolist()
    elif filter_value == 4:
        ids = DR7_TABLE.iloc[idx[0]].query('STAGE1_votes_12RP == 20 and STAGE2_prob_12RP > 0.5').objID.values.astype(str).tolist()
    elif filter_value == 5:
        ids = DR7_TABLE.iloc[idx[0]].query('STAGE1_votes_12RP > 10 and STAGE2_prob_12RP > 0.5').objID.values.astype(str).tolist()
    else:
        ids = DR7_TABLE.iloc[idx[0]].objID.values.astype(str).tolist()

    _, idx_mock = FEATURES_TREE_MOCK.query(vec.reshape(1, FEATURES_MOCK.shape[1]), k=MOCK_NEIGHBORS)
    ids_mock = METADATA.iloc[idx_mock[0]].DB_ID.values.astype(str).tolist()

    return jsonify(ids=ids, ids_mock=ids_mock)
