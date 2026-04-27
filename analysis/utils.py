import pyam

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
    crossing_timings=(
        df_timeseries
        .apply(
            lambda x: pyam.timeseries.cross_threshold(
                x,
                threshold=level
            )[0],
            axis=1
        )
    )

    # Set metadata
    df_input.set_meta(
        crossing_timings,
        name=meta_column
    )

    return crossing_timings