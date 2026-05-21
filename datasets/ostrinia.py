from tsl.datasets.prototypes import DatetimeDataset
import pandas as pd
import os
import numpy as np
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


class Ostrinia(DatetimeDataset):

    similarity_options = {'distance'}

    def __init__(self, 
                 root: str = "datasets",
                   input_zeros: bool = True,
                     freq=None, target="nb_ostrinia",
                       smooth: bool = False,
                         full_normalization: bool = False,
                          drop_nodes: bool = True,
                            add_second_target: bool = True,
                             delay: int = 14, 
                              spatial_information: bool = True,):
                
        self.root = root
        self.extra_data = None
        self.target = target
        self.smooth = smooth
        self.full_normalization = full_normalization
        self.drop_nodes = drop_nodes
        self.add_second_target = add_second_target
        self.delay = delay
        self.spatial_information = spatial_information

        if add_increment_flag:
            self.flags = {'increment_flag': None}

        df, dist, mask = self.load(input_zeros)

        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="Ostrinia")

        if self.spatial_information:
            self.add_covariate('dist', dist, pattern='n n')


    def load_raw(self):

        self.maybe_build()
        # load insect data
        insect_path = os.path.join(self.root_dir, 'thesis_dataset_after_transformation.csv')
        df = pd.read_csv(insect_path)

        # add missing values (index is sorted)
        # date_range = pd.date_range(df.index[0], df.index[-1], freq='5T')
        # df = df.reindex(index=date_range)

        # load distance matrix
        path = os.path.join(self.root_dir, 'ostrinia_distance_matrix.npy')
        dist = np.load(path)

        # convert distance matrix to edge list and edge weights

        # load coordinates
        coords_path = os.path.join(self.root_dir, 'ostrinia_coordinate_pairs.npy')
        coords = np.load(coords_path)

        return df, dist, coords

    def load(self, impute_zeros=True):

        df, dist, coords = self.load_raw()

        # change Latitude and Longitude vars to its closes cluster (which are coords)
        if self.spatial_information:
            df = map_to_nearest_coords(df, coords)
        else:
            df = collapse_to_single_node(df)

        if self.add_second_target:
            df = add_increment_flag(df)

        # reshape the df to be date as index and node_index as columns
        df = df.set_index('Date')
        df_clean = df.pivot(columns='node_index', values=self.target)

        # if smooth is True, apply a rolling mean with a window of 7
        if self.smooth:
            df_clean = df_clean.rolling(window=7, min_periods=7, center=True).mean()

        if self.drop_nodes:
            df_clean = mask_top2_node_year_max(df_clean)

        # compute mask for nan values
        mask = ((~np.isnan(df_clean.values)).astype('uint8'))
        if impute_zeros:
            # replace NaN with 0
            df_clean = df_clean.fillna(0)
        else:
            print("Imputing zeros is disabled, NaN values will remain in the dataset.")

        # Check for duplicate combinations of Date and node_index
        duplicates = df.reset_index().duplicated(subset=['Date', 'node_index'], keep=False)
        if duplicates.any():
            print("Duplicate combinations of Date and node_index found:")
            print(df.reset_index()[duplicates][['node_index']])
        # do the same with columns trap_elevation, corn_size, incrementing_ostrinia, nb_ostrinia_new, Weather Station Changins, TempAv, TempMin, TempMax, TempGrasMin, TempSol-5, TempSoil-10, RHmoy, Prec, PrecMax, Insol, Ray, EvapTranspi, WindSpeedAb, WindSpeedmax, WaterDef, Somt, Soms
        # put them in self.extra_data which will be a dict of DataFrames
        list_of_extra_columns = [
            'nb_ostrinia',
            'trap_elevation',
            'corn_size',
            'incrementing_ostrinia',
            'nb_ostrinia_new',
            'Weather Station Changins',
            'TempAv',
            'TempMin',
            'TempMax',
            'TempGrasMin',
            'TempSol-5',
            'TempSoil-10',
            'RHmoy',
            'Prec',
            'PrecMax',
            'Insol',
            'Ray',
            'EvapTranspi',
            'WindSpeedAb',
            'WindSpeedmax',
            'WaterDef',
            'Somt',
            'Soms',
            # spatio temporal features
            'Latitude_x',
            'Longitude_x',
            'year',
            'month',
        ]

        if self.add_second_target:
            list_of_extra_columns.append('increment_flag')

        extra_data = {}
        for element in list_of_extra_columns:
            if element != self.target:
                extra_data[element] = df.pivot(columns='node_index', values=element)
                extra_data[element] = extra_data[element].fillna(0)  # Fill NaN with 0 for extra data
                if element in ['nb_ostrinia', 'incrementing_ostrinia', 'nb_ostrinia_new']:
                    # if the column is one of these, we want to impute zeros
                    extra_data[element] = extra_data[element].rolling(window=7, min_periods=7).mean()
                    extra_data[element] = extra_data[element].fillna(0)


        # if self.full_normalization is True, normalize self.target and extra_data in [nb_ostrinia, incrementing_ostrinia, nb_ostrinia_new] 
        # by the max of each node each year
        if self.full_normalization:
            for col in ['nb_ostrinia', 'incrementing_ostrinia', 'nb_ostrinia_new']:
                if col in extra_data:

                    extra_data[col].index = pd.to_datetime(extra_data[col].index, format='%Y-%m-%d')
                    yearly_max = extra_data[col].groupby(extra_data[col].index.year).transform('max')
                    # if yearly_max is 0, we set it to 1 to avoid division by zero
                    yearly_max[yearly_max == 0] = 1
                    # normalize by the yearly max
                    extra_data[col] = extra_data[col].div(yearly_max, axis=0)
            
            df_clean.index = pd.to_datetime(df_clean.index, format='%Y-%m-%d')
            yearly_max = df_clean.groupby(df_clean.index.year).transform('max')
            # if yearly_max is 0, we set it to 1 to avoid division by zero
            yearly_max[yearly_max == 0] = 1
            # normalize by the yearly max
            df_clean = df_clean.div(yearly_max, axis=0)

        if self.add_second_target:

            # df_increment = df.pivot(columns='node_index', values='increment_flag')
            # df_increment.index = pd.to_datetime(df_increment.index, format='%Y-%m-%d')

            # # turn nan into 0
            # df_increment = df_increment.fillna(0)

            # # --- inputs ----------------------------------------------------------
            # channels = {"clean": df_clean, "increment": df_increment}

            # # 1 — sanity check: identical date index on every channel
            # base_idx = next(iter(channels.values())).index
            # assert all(df.index.equals(base_idx) for df in channels.values())

            # # 2 — build the MultiIndex for columns
            # nodes      = df_clean.columns            # ['node-1', 'node-2', …]
            # first_lvl  = list(channels)              # ['clean', 'increment']
            # multi_cols = pd.MultiIndex.from_product(
            #                 [first_lvl, nodes], names=["channel", "node"]
            #             )

            # # 3 — horizontally stack the underlying NumPy blocks
            # data_block = np.hstack([channels[k].to_numpy() for k in first_lvl])

            # # 4 — assemble the final DataFrame
            # df_clean = pd.DataFrame(data_block, index=base_idx, columns=multi_cols)

            # # optional: keep original ordering of nodes
            # df_clean = (df_clean                       # your original channel→node frame
            #             .swaplevel('channel', 'node', axis=1)
            #             .sort_index(axis=1, level=['node', 'channel']))

            # add to extra elements
            self.flags['increment_flag'] = df.pivot(columns='node_index', values='increment_flag')
            self.flags['increment_flag'] = self.flags['increment_flag'].fillna(0)  # Fill NaN with 0 for increment_flag

        # add as covariate the day of the year. index is a string with format YYYY-MM-DD. extract the day of the year and create an enconding 
        index = df.index
        # encode it as a number from 0 to 365
        df['day_of_year'] = pd.to_datetime(index, format='%Y-%m-%d').dayofyear
        extra_data['day_of_year'] = df.pivot(columns='node_index', values="day_of_year")

        # drop Weather Station Changins
        if 'Weather Station Changins' in extra_data:
            extra_data.pop('Weather Station Changins')
            
        self.extra_data = extra_data

        return df_clean, dist, mask


    def get_connectivity(self, layout, **kwargs):
        """
        Get the connectivity matrix for the dataset.
        
        Returns:
            np.ndarray: The connectivity matrix.
        """
        if layout == 'edge_index':
            from tsl.ops.connectivity import adj_to_edge_index
            return adj_to_edge_index(self._covariates['dist']['value'])
        elif layout == 'distance':
            dist = self._covariates['dist']['value']
        return dist

    # def set_connectivity(self, connectivity):
    #     """
    #     Set the connectivity matrix for the dataset.
        
    #     Parameters:
    #         connectivity (np.ndarray): The new connectivity matrix.
    #     """
    #     self.edge_index, self.edge_weight = self._parse_connectivity(
    #         connectivity, 'edge_index')
        

def map_to_nearest_coords(df, coord_list):
    """
    Map DataFrame coordinates to nearest coordinates in coord_list
    
    Parameters:
    df: DataFrame with 'Latitude_x' and 'Longitude_x' columns
    coord_list: List of coordinate pairs [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
    DataFrame with added columns for nearest coordinates and their indices
    """
    
    # Convert coord_list to numpy array
    coord_array = np.array(coord_list)
    
    # Extract df coordinates
    df_coords = df[['Latitude_x', 'Longitude_x']].values
    
    # Convert to radians for haversine distance
    df_coords_rad = np.radians(df_coords)
    coord_array_rad = np.radians(coord_array)
    
    # Use BallTree for efficient nearest neighbor search with haversine metric
    tree = BallTree(coord_array_rad, metric='haversine')
    
    # Find nearest neighbor for each point in df
    distances, indices = tree.query(df_coords_rad, k=1)
    
    # Convert distances from radians to kilometers (Earth's radius ≈ 6371 km)
    distances_km = distances.flatten() * 6371
    
    # Add results to dataframe
    df['node_index'] = indices.flatten()
    df['error_distance'] = distances_km

    return df

def collapse_to_single_node(df):
    """
    Collapse all data points to a single node (node_index = 0)
    
    Parameters:
    df: DataFrame with 'Latitude_x' and 'Longitude_x' columns
    
    Returns:
    DataFrame with added 'node_index' column set to 0
    """
    
    df = df.copy()
    df['node_index'] = 0
    df['error_distance'] = 0.0  # No error since all points map to the same node

    # add all rows with the same date into a single row by summing them
    df = df.groupby('Date').sum().reset_index()

    return df

def mask_top2_node_year_max(df, return_masked=False):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Extract year
    df_years = df.copy()
    df_years['year'] = df_years.index.year

    # Compute max(abs) per (node, year)
    max_by_node_year = (
        df_years
        .groupby('year')
        .agg(lambda x: np.abs(x).max())
        .reset_index()
        .melt(id_vars='year', var_name='node', value_name='abs_max')
    )

    # Get top 2 node-year pairs with highest max values
    top2 = max_by_node_year.nlargest(2, 'abs_max')[['node', 'year']]

    # Apply mask
    masked_pairs = []
    for _, row in top2.iterrows():
        node, year = row['node'], row['year']
        df.loc[df.index.year == year, node] = np.nan
        masked_pairs.append((node, year))

    return (df, masked_pairs) if return_masked else df


def _parse_pair(cell) -> tuple[float, float]:
    """
    Convert "2'506'873.8, 1'139'573.6" → (2506873.8, 1139573.6).
    For any cell that
        • is NaN / None,
        • is a lone int/float,
        • lacks a comma,
    return (nan, nan) so it will be skipped.
    """
    # 1. missing?
    if pd.isna(cell):
        return (np.nan, np.nan)

    # 2. numeric scalar (e.g., 0, 0.0) -> skip
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return (np.nan, np.nan)

    # 3. treat as string
    s = str(cell).strip()
    if ',' not in s:                # not a coordinate pair
        return (np.nan, np.nan)

    x_str, y_str = map(str.strip, s.split(',', 1))  # split only once

    try:
        x = float(x_str.replace("'", ""))           # remove apostrophes
        y = float(y_str.replace("'", ""))
    except ValueError:                              # parsing failed → skip
        return (np.nan, np.nan)

    return (x, y)

def scale_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse every cell, split into X and Y, and scale each axis to [-1, 1].
    Cells that do not contain a valid coordinate pair remain NaN.
    """
    parsed   = df.map(_parse_pair)
    x_raw    = parsed.map(lambda xy: xy[0])
    y_raw    = parsed.map(lambda xy: xy[1])

    # axis-wise extrema (NaNs ignored)
    x_min, x_max = x_raw.min().min(), x_raw.max().max()
    y_min, y_max = y_raw.min().min(), y_raw.max().max()

    # avoid division by zero if all valid values are identical
    if x_max > x_min:
        x_scaled = 2 * (x_raw - x_min) / (x_max - x_min) - 1
    else:
        x_scaled = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    if y_max > y_min:
        y_scaled = 2 * (y_raw - y_min) / (y_max - y_min) - 1
    else:
        y_scaled = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    return x_scaled, y_scaled


def add_increment_flag(df: pd.DataFrame,
                       cum_col: str = "incrementing_ostrinia",
                       flag_col: str = "increment_flag") -> pd.DataFrame:
    """
    Append a binary flag indicating where the cumulative count increases.

    Parameters
    ----------
    df : pd.DataFrame
        Input data‐frame that already contains the cumulative‐sum column.
    cum_col : str, default "incrementing_ostrinia"
        Name of the cumulative‐sum column.
    flag_col : str, default "increment_flag"
        Name of the flag column to be created.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the additional flag column.
    """
    out = df.copy()

    # Detect changes, then shift one row upward so the flag appears earlier
    changes = out[cum_col].diff().fillna(0).ne(0)
    out[flag_col] = changes.shift(-1, fill_value=False).astype(int)

    return out

def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Ensure datetime64[ns] dtype,
    2. Remove any time-zone offset,
    3. Floor to the day,
    4. Drop or aggregate duplicates.
    """
    idx = pd.to_datetime(df.index, errors="raise")        # 1
    idx = idx.tz_localize(None, ambiguous="raise")        # 2
    idx = idx.floor("D")                                  # 3
    df = df.copy()
    df.index = idx

    # 4 — pick ONE of the following strategies
    # (a) keep the last occurrence
    df = df[~df.index.duplicated(keep="last")]
    # (b) or aggregate explicitly:
    # df = df.groupby(level=0).mean()

    return df