import pyam
import scmdata
import silicone.time_projectors as timeproj
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_crossing_timings(df_input, variable, level, meta_column):
    """
    Calculate the timing when a variable crosses a specified threshold level.

    This function filters a pyam IamDataFrame for a specific variable, determines
    when each scenario's timeseries crosses a given threshold level, and stores
    the crossing timing as metadata in the original dataframe.

    Args:
        df_input (pyam.IamDataFrame): Input dataframe containing scenario timeseries data
        variable (str): Name of the variable to analyze for threshold crossing
        level (float): Threshold value to detect crossings
        meta_column (str): Name of the metadata column where crossing timings will be stored

    Returns:
        pandas.Series: Series containing the crossing timings for each scenario,
                      indexed by the multi-index from the input dataframe

    Raises:
        TypeError: If df_input is not a pyam.IamDataFrame, or if variable/meta_column
                  are not strings, or if level is not numeric
        ValueError: If variable doesn't exist in the dataframe or if the filtered
                   dataframe is empty

    Notes:
        - Uses pyam.timeseries.cross_threshold() which returns the first year when
          the timeseries crosses the specified threshold
        - The crossing timing is added as metadata to the input dataframe in-place
        - If a scenario never crosses the threshold, the result may be NaN or None
    """
    # Validate df_input type
    if not isinstance(df_input, pyam.IamDataFrame):
        raise TypeError(
            f"df_input must be a pyam.IamDataFrame, got {type(df_input).__name__}"
        )

    # Validate variable type
    if not isinstance(variable, str):
        raise TypeError(
            f"variable must be a string, got {type(variable).__name__}"
        )

    # Validate meta_column type
    if not isinstance(meta_column, str):
        raise TypeError(
            f"meta_column must be a string, got {type(meta_column).__name__}"
        )

    # Validate level type (accept int or float)
    if not isinstance(level, (int, float)):
        raise TypeError(
            f"level must be numeric (int or float), got {type(level).__name__}"
        )

    # Check if variable exists in the dataframe
    if variable not in df_input.variable:
        available_vars = df_input.variable
        raise ValueError(
            f"Variable '{variable}' not found in dataframe. "
            f"Available variables: {list(available_vars)[:10]}"
            + ("..." if len(available_vars) > 10 else "")
        )

    # Filter and get timeseries
    df_timeseries = (
        df_input
        .filter(
            variable=variable
        )
        .timeseries()
    )

    # Calculate crossing timings
    def get_crossing_year(timeseries):
        """Extract first crossing year, return None if no crossing occurs."""
        crossings = pyam.timeseries.cross_threshold(timeseries, threshold=level)
        return crossings[0] if len(crossings) > 0 else None

    crossing_timings = df_timeseries.apply(get_crossing_year, axis=1)

    # Set metadata
    df_input.set_meta(
        crossing_timings,
        name=meta_column
    )

    return crossing_timings

# Adapted from climate_assessment: https://github.com/iiasa/climate-assessment/blob/main/src/climate_assessment/utils.py
def _remove_equiv_and_hyphens_from_unit(inscmrun):
    # equiv and hyphens in units don't place nice with pint
    out = inscmrun.copy()
    out["unit"] = out["unit"].str.replace("-", "").str.replace("equiv", "")

    return out


def convert_units_to_co2_equiv(df, metric):
    """
    Converts the units of gases reported in kt into Mt CO2 equivalent per year

    Uses GWP100 values from either (by default) AR5 or AR4 IPCC reports.

    Parameters
    ----------
    df : :class:`pyam.IamDataFrame`
        The input dataframe whose units need to be converted.

    metric : str
        The name of the conversion metric to use. This will usually be AR<4/5/6>GWP100.

    Return
    ------
    :class:`pyam.IamDataFrame`
        The input data with units converted.
    """
    # probably possible to do this directly with pyam too somehow, but
    # here's the scmdata version
    scmrun = scmdata.ScmRun(df.timeseries())

    # strip hyphens and equiv from inputs
    scmrun = _remove_equiv_and_hyphens_from_unit(scmrun)

    res = scmrun.convert_unit("Mt CO2/yr", context=metric).drop_meta("unit_context")

    # put back the equiv (even though it breaks pint)
    res["unit"] = "Mt CO2-equiv/yr"
    res = pyam.IamDataFrame(res.timeseries(time_axis="year"))

    return res

def add_gwp100_kyoto(
    df,
    kyoto_gases,
    gwp_instance="AR6GWP100",
    prefix="",
):
    total_co2_var = f"{prefix}Emissions|CO2"

    tmp = df.copy()
    tmp.require_data(variable=total_co2_var, exclude_on_fail=True)
    calc_df = tmp.filter(exclude=False)

    # aggregate CO2 before moving on
    tmp_no_total_co2 = tmp.filter(exclude=True)
    if not tmp_no_total_co2.empty:
        tmp_no_total_co2.aggregate(total_co2_var, append=True)
        calc_df = calc_df.append(tmp_no_total_co2)

    dfp = calc_df.filter(variable=[prefix + s for s in kyoto_gases])


    diff = list(set(df.data["variable"]) - set(dfp.data["variable"]))

    kyoto = (
        convert_units_to_co2_equiv(
            dfp,
            gwp_instance,
        )
        .timeseries()
        .groupby(["model", "scenario", "region", "unit"])
        .sum(min_count=1)
    )

    if gwp_instance == "AR5GWP100":
        kyoto["variable"] = prefix + "Emissions|Kyoto Gases (AR5-GWP100)"
    elif gwp_instance == "AR6GWP100":
        kyoto["variable"] = prefix + "Emissions|Kyoto Gases (AR6-GWP100)"
    else:
        raise NotImplementedError(gwp_instance)

    return pyam.IamDataFrame(pyam.concat([df, kyoto]))


def _process_single_scenario(
    model,
    scenario,
    constituent_emissions,
    nz_year,
    start_year,
    end_year,
):
    """
    Helper function to process a single scenario for parallel execution.

    Parameters
    ----------
    model : str
        Model name
    scenario : str
        Scenario name
    constituent_emissions : pyam.IamDataFrame
        IamDataFrame of constituent emissions (all scenarios)
    nz_year : int
        Net-zero year for this scenario
    start_year : int
        Starting year for output
    end_year : int
        Ending year for output

    Returns
    -------
    pyam.IamDataFrame
        Extended timeseries for this scenario
    """
    # Filter for this model-scenario combination
    scenario_data = constituent_emissions.filter(model=model, scenario=scenario)

    # Filter data from start_year to net-zero year
    scenario_data_to_nz = scenario_data.filter(year=range(start_year, nz_year + 1))

    if nz_year == end_year:
        # No extension needed
        return scenario_data_to_nz
    else:
        # Flat-line extension from net-zero to end_year
        # Pre-create the projector to avoid recreation overhead
        projected_emissions = timeproj.LinearExtender().derive_relationship(
            variable=scenario_data_to_nz.variable,
            gradient=0,
            times=range(nz_year + 1, end_year + 1)
        )(scenario_data_to_nz)

        return pyam.concat([scenario_data_to_nz, projected_emissions])


def extend_flatline_after_netzero(
    df,
    constituent_variables,
    aggregate_variable,
    netzero_meta_column,
    start_year=2015,
    end_year=2100,
    scenario_suffix=None,
    default_netzero_year=2100,
    show_progress=True,
    n_workers=None,
):
    """
    Flat-line constituent emissions after the aggregate variable reaches net-zero.

    This function extends constituent variable timeseries with constant values
    (flat line) after the aggregate variable crosses zero. This is useful for
    creating scenarios where individual gas emissions remain constant after
    achieving net-zero for the aggregate.

    Parameters
    ----------
    df : pyam.IamDataFrame
        Input dataframe containing emissions timeseries and net-zero metadata
    constituent_variables : list of str
        List of constituent variable names to flat-line (e.g., individual gas emissions)
    aggregate_variable : str
        Name of the aggregate variable that defines net-zero timing
        (e.g., "Emissions|CO2", "Emissions|Kyoto Gases (AR6-GWP100)")
    netzero_meta_column : str
        Name of the metadata column containing net-zero crossing years
    start_year : int, optional
        Starting year for the output timeseries (default: 2015)
    end_year : int, optional
        Ending year for the output timeseries (default: 2100)
    scenario_suffix : str, optional
        Suffix to append to scenario names (e.g., "_NZCO2"). If None, original names are kept
    default_netzero_year : int, optional
        Default year to use if net-zero metadata is NaN (default: 2100)
    show_progress : bool, optional
        Whether to show progress bar (default: True)
    n_workers : int, optional
        Number of parallel workers to use. If None, uses all available CPU cores (default: None)

    Returns
    -------
    pyam.IamDataFrame
        DataFrame containing:
        - Flat-lined constituent variables (extended to end_year)
        - All other variables from the input (unchanged)
        - Optionally renamed scenarios if scenario_suffix is provided

    Notes
    -----
    - Scenarios that never reach net-zero (NaN in metadata) use default_netzero_year
    - Variables not in constituent_variables list are passed through unchanged
    - The function uses silicone.LinearExtender with gradient=0 for flat-line projection
    - Uses parallel processing to speed up computation across multiple scenarios

    Examples
    --------
    >>> # Flat-line CO2 components after total CO2 reaches net-zero
    >>> df_extended = extend_flatline_after_netzero(
    ...     df,
    ...     constituent_variables=["Emissions|CO2|Fossil", "Emissions|CO2|Biosphere"],
    ...     aggregate_variable="Emissions|CO2",
    ...     netzero_meta_column="year_netzero_co2",
    ...     scenario_suffix="_NZCO2",
    ...     n_workers=4
    ... )
    """
    # Validate inputs
    if not isinstance(df, pyam.IamDataFrame):
        raise TypeError(f"df must be a pyam.IamDataFrame, got {type(df).__name__}")

    if netzero_meta_column not in df.meta.columns:
        raise ValueError(
            f"Metadata column '{netzero_meta_column}' not found. "
            f"Available columns: {list(df.meta.columns)}"
        )

    # Check if time domain is datetime and swap if needed
    if df.time_col == "time":
        df_working = df.swap_time_for_year()
    else:
        df_working = df

    # Split constituent variables from other variables
    constituent_emissions = df_working.filter(variable=constituent_variables)
    other_emissions = df_working.filter(variable=constituent_variables, keep=False)

    # Fill NaN net-zero years with default
    netzero_years = df_working.meta[netzero_meta_column].fillna(default_netzero_year)

    # Prepare list of scenarios to process
    scenarios_to_process = [
        (index[0], index[1], int(netzero_years.loc[index]))
        for index, _ in constituent_emissions.meta.iterrows()
    ]

    # Process scenarios in parallel using threads (avoids pickling overhead)
    extended_constituents = []

    # Use ThreadPoolExecutor for parallel processing (better for I/O bound operations)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                _process_single_scenario,
                model,
                scenario,
                constituent_emissions,
                nz_year,
                start_year,
                end_year,
            )
            for model, scenario, nz_year in scenarios_to_process
        ]

        # Collect results with progress bar
        if show_progress:
            with tqdm(total=len(futures), desc="Processing scenarios") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    extended_constituents.append(result)
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                result = future.result()
                extended_constituents.append(result)

    # Combine all extended constituent emissions
    flat_lined_constituents = pyam.concat(extended_constituents)

    # Combine with other emissions
    compiled_emissions = pyam.concat([flat_lined_constituents, other_emissions])

    # Reaggregate the aggregate variable to reflect the flat-lining of constituents
    # First, remove the old aggregate variable if it exists
    compiled_emissions_no_agg = compiled_emissions.filter(
        variable=aggregate_variable,
        keep=False
    )

    # Reaggregate from the constituent variables
    try:
        compiled_emissions_no_agg.aggregate(
            variable=aggregate_variable,
            components=constituent_variables,
            append=True
        )
        compiled_emissions = compiled_emissions_no_agg
    except Exception:
        # If aggregation fails, just keep the data without the aggregate
        # This can happen if constituents don't sum to the aggregate properly
        compiled_emissions = compiled_emissions_no_agg

    # Rename scenarios if suffix provided
    if scenario_suffix:
        # Build rename mapping all at once instead of one at a time
        scenario_mapping = {scen: f"{scen}{scenario_suffix}" for scen in compiled_emissions.scenario}
        compiled_emissions.rename(scenario=scenario_mapping, inplace=True)

    return compiled_emissions


def sanitize_label(label):
    """
    Sanitize a label by replacing whitespace and backslashes with double underscores.

    This function is used to create filesystem-safe filenames from scenario/model names
    that may contain spaces or backslashes.

    Parameters
    ----------
    label : str
        The label to sanitize (e.g., model name, scenario name)

    Returns
    -------
    str
        Sanitized label with whitespace and backslashes replaced by '__'

    Raises
    ------
    TypeError
        If label is not a string

    Examples
    --------
    >>> sanitize_label("SSP2 4.5")
    'SSP2__4.5'
    >>> sanitize_label("REMIND\\IMAGE")
    'REMIND__IMAGE'
    >>> sanitize_label("AIM/CGE V2.2")
    'AIM/CGE__V2.2'
    """
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}")

    # Replace all whitespace characters with '__'
    sanitized = label.replace(" ", "__").replace("\t", "__").replace("\n", "__").replace("\r", "__")
    # Replace backslashes with '__'
    sanitized = sanitized.replace("/", "__")

    return sanitized
