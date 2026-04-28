import unittest
import numpy as np
import pandas as pd
import pyam
from analysis.utils import calculate_crossing_timings


class TestCalculateCrossingTimings(unittest.TestCase):
    """Test suite for calculate_crossing_timings function."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        # Create sample data for pyam IamDataFrame
        # Using descending values to cross threshold "from above"
        # All scenarios cross the threshold to ensure no IndexError
        self.sample_data = pd.DataFrame([
            # Scenario 1: crosses threshold of 50 between 2025 and 2030
            ["Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2020, 100],
            ["Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2025, 60],
            ["Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2030, 40],
            ["Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2035, 30],
            ["Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2040, 20],
            # Scenario 2: crosses threshold of 50 between 2030 and 2035
            ["Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr", 2020, 120],
            ["Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr", 2025, 90],
            ["Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr", 2030, 60],
            ["Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr", 2035, 40],
            ["Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr", 2040, 30],
            # Scenario 3: crosses threshold of 50 between 2040 and 2045
            ["Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr", 2020, 150],
            ["Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr", 2025, 140],
            ["Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr", 2030, 80],
            ["Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr", 2035, 60],
            ["Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr", 2040, 40],
        ], columns=["model", "scenario", "region", "variable", "unit", "year", "value"])

        self.df = pyam.IamDataFrame(self.sample_data)

    def test_valid_crossing_detection(self):
        """Test that function correctly detects threshold crossings."""
        result = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50,
            meta_column="net_zero_year"
        )

        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)

        # Check that metadata was set on the dataframe
        self.assertIn("net_zero_year", self.df.meta.columns)

        # The cross_threshold function returns interpolated years
        # Index includes: (model, scenario, region, variable, unit)
        # Scenario1: crosses between 2025 (60) and 2030 (40), interpolated year ~ 2028
        # Scenario2: crosses between 2030 (60) and 2035 (40), interpolated year ~ 2032
        # Scenario3: crosses between 2035 (60) and 2040 (40), interpolated year ~ 2038
        idx_s1 = ("Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr")
        idx_s2 = ("Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr")
        idx_s3 = ("Model1", "Scenario3", "Region1", "Emissions|CO2", "Mt CO2/yr")

        crossing_s1 = result[idx_s1]
        crossing_s2 = result[idx_s2]
        crossing_s3 = result[idx_s3]

        # Check that crossings occur in the expected ranges
        self.assertGreater(crossing_s1, 2025)
        self.assertLess(crossing_s1, 2030)
        self.assertGreater(crossing_s2, 2030)
        self.assertLess(crossing_s2, 2035)
        self.assertGreater(crossing_s3, 2035)
        self.assertLess(crossing_s3, 2040)

    def test_metadata_set_correctly(self):
        """Test that metadata is properly set on the input dataframe."""
        result = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50,
            meta_column="crossing_year"
        )

        # Verify metadata column exists
        self.assertIn("crossing_year", self.df.meta.columns)

        # Verify metadata values match the crossing timings returned
        # Note: metadata uses (model, scenario) index while
        # result uses (model, scenario, region, variable, unit)
        meta_values = self.df.meta["crossing_year"]
        idx_full = ("Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr")
        idx_meta = ("Model1", "Scenario1")

        self.assertEqual(
            meta_values.loc[idx_meta],
            result[idx_full]
        )

    def test_invalid_df_type(self):
        """Test that TypeError is raised for invalid df_input type."""
        invalid_df = pd.DataFrame({"a": [1, 2, 3]})

        with self.assertRaises(TypeError) as context:
            calculate_crossing_timings(
                invalid_df,
                variable="Emissions|CO2",
                level=50,
                meta_column="net_zero_year"
            )

        self.assertIn("must be a pyam.IamDataFrame", str(context.exception))

    def test_invalid_variable_type(self):
        """Test that TypeError is raised for invalid variable type."""
        with self.assertRaises(TypeError) as context:
            calculate_crossing_timings(
                self.df,
                variable=123,  # Should be string
                level=50,
                meta_column="net_zero_year"
            )

        self.assertIn("variable must be a string", str(context.exception))

    def test_invalid_meta_column_type(self):
        """Test that TypeError is raised for invalid meta_column type."""
        with self.assertRaises(TypeError) as context:
            calculate_crossing_timings(
                self.df,
                variable="Emissions|CO2",
                level=50,
                meta_column=123  # Should be string
            )

        self.assertIn("meta_column must be a string", str(context.exception))

    def test_invalid_level_type(self):
        """Test that TypeError is raised for invalid level type."""
        with self.assertRaises(TypeError) as context:
            calculate_crossing_timings(
                self.df,
                variable="Emissions|CO2",
                level="fifty",  # Should be numeric
                meta_column="net_zero_year"
            )

        self.assertIn("level must be numeric", str(context.exception))

    def test_level_accepts_int_and_float(self):
        """Test that level parameter accepts both int and float."""
        # Test with int
        result_int = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50,
            meta_column="crossing_int"
        )
        self.assertIsInstance(result_int, pd.Series)

        # Test with float
        result_float = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50.0,
            meta_column="crossing_float"
        )
        self.assertIsInstance(result_float, pd.Series)

    def test_variable_not_found(self):
        """Test that ValueError is raised when variable doesn't exist."""
        with self.assertRaises(ValueError) as context:
            calculate_crossing_timings(
                self.df,
                variable="NonExistent|Variable",
                level=50,
                meta_column="net_zero_year"
            )

        self.assertIn("Variable 'NonExistent|Variable' not found", str(context.exception))
        self.assertIn("Available variables", str(context.exception))

    def test_with_different_threshold(self):
        """Test crossing detection with different threshold values."""
        # Test with threshold of 70
        # Scenario1: crosses between 2020 (100) and 2025 (60)
        # Scenario2: crosses between 2025 (90) and 2030 (60)
        result = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=70,
            meta_column="threshold_70"
        )

        # Both scenarios should have crossing timings
        idx_s1 = ("Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr")
        idx_s2 = ("Model1", "Scenario2", "Region1", "Emissions|CO2", "Mt CO2/yr")

        crossing_s1 = result[idx_s1]
        crossing_s2 = result[idx_s2]

        # Verify crossings are in expected year ranges
        self.assertGreater(crossing_s1, 2020)
        self.assertLess(crossing_s1, 2025)
        self.assertGreater(crossing_s2, 2025)
        self.assertLess(crossing_s2, 2030)

    def test_return_type_and_structure(self):
        """Test that the return value has correct type and index structure."""
        result = calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50,
            meta_column="net_zero_year"
        )

        # Check return type
        self.assertIsInstance(result, pd.Series)

        # Check that index is a MultiIndex with correct levels
        # The index includes all scenario dimensions plus variable and unit
        self.assertTrue(isinstance(result.index, pd.MultiIndex))
        self.assertEqual(
            result.index.names,
            ["model", "scenario", "region", "variable", "unit"]
        )

    def test_multiple_variables_in_dataframe(self):
        """Test that function works correctly when dataframe contains multiple variables."""
        # Add another variable to the dataframe
        additional_data = pd.DataFrame([
            ["Model1", "Scenario1", "Region1", "Temperature|Global", "°C", 2020, 1.0],
            ["Model1", "Scenario1", "Region1", "Temperature|Global", "°C", 2025, 1.2],
            ["Model1", "Scenario1", "Region1", "Temperature|Global", "°C", 2030, 1.5],
        ], columns=["model", "scenario", "region", "variable", "unit", "year", "value"])

        df_multi = pyam.IamDataFrame(
            pd.concat([self.sample_data, additional_data], ignore_index=True)
        )

        # Should still work correctly filtering for specific variable
        result = calculate_crossing_timings(
            df_multi,
            variable="Emissions|CO2",
            level=50,
            meta_column="net_zero_year"
        )

        self.assertIsInstance(result, pd.Series)
        idx = ("Model1", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr")
        # Should cross between 2025 (60) and 2030 (40)
        self.assertGreater(result[idx], 2025)
        self.assertLess(result[idx], 2030)

    def test_metadata_overwrite(self):
        """Test that calling the function twice with same meta_column overwrites metadata."""
        idx_meta = ("Model1", "Scenario1")

        # First call with threshold of 50
        calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=50,
            meta_column="crossing_year"
        )
        first_value = self.df.meta["crossing_year"].loc[idx_meta]

        # Second call with different threshold of 70
        calculate_crossing_timings(
            self.df,
            variable="Emissions|CO2",
            level=70,
            meta_column="crossing_year"
        )
        second_value = self.df.meta["crossing_year"].loc[idx_meta]

        # Values should be different (70 threshold crossed earlier than 50)
        self.assertNotEqual(first_value, second_value)
        self.assertLess(second_value, first_value)

    def test_edge_case_exact_threshold_value(self):
        """Test behavior when timeseries values exactly match the threshold."""
        edge_data = pd.DataFrame([
            ["Model2", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2020, 100],
            ["Model2", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2025, 50],
            ["Model2", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr", 2030, 25],
        ], columns=["model", "scenario", "region", "variable", "unit", "year", "value"])

        df_edge = pyam.IamDataFrame(edge_data)

        result = calculate_crossing_timings(
            df_edge,
            variable="Emissions|CO2",
            level=50,
            meta_column="crossing_year"
        )

        # cross_threshold uses linear interpolation
        # Between 2020 (100) and 2025 (50), threshold 50 is crossed at 2025
        # But interpolation happens, so the year might be slightly after 2025
        idx = ("Model2", "Scenario1", "Region1", "Emissions|CO2", "Mt CO2/yr")
        crossing_year = result[idx]

        # Verify crossing is detected and is numeric
        self.assertIsInstance(crossing_year, (int, float, np.integer, np.floating))
        # Should be around 2025 (within a reasonable range)
        self.assertGreater(crossing_year, 2020)
        self.assertLessEqual(crossing_year, 2030)


if __name__ == "__main__":
    unittest.main()
