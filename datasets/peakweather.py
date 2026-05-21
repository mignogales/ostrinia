from tsl.datasets.prototypes import DatetimeDataset
import pandas as pd
import os
import numpy as np
import pandas as pd
import numpy as np
from peakweather import PeakWeatherDataset
from numpy import concatenate, isnan, nan_to_num
from dataset_generator.functions import degree_days_from_multiindex_wide, simulate_hatching_from_dd, simulate_hatching_from_dd_v2, propagate_event, simulate_hatching_from_dd_v3, generate_soil_quality

class PeakWeatherTSL(DatetimeDataset):

    similarity_options = {'distance'}

    def __init__(self, 
                 root: str = "datasets",
                   input_zeros: bool = True,
                    station_type = "meteo_station",
                     freq: str = 'h', 
                      target="precipitation",
                       delay: int = 0,
                        synth_data: bool = False, 
                            add_second_target: bool = False):
        
        self.root = root
        self.extra_data = None
        self.target = target
        self.delay = delay
        self.freq = freq
        self.station_type = station_type
        self.add_second_target = add_second_target
        self.flags = {}

        self.dataset = PeakWeatherDataset(
                                        root="data",  # Path to the dataset
                                        pad_missing_variables=True,  # Pad missing variables with NaN
                                        years=None,  # Years to include in the dataset (None for all)
                                        parameters=None,  # Parameters to include in the dataset (None for all)
                                        extended_topo_vars="all",  # Optional extended topographic variables
                                        imputation_method="zero" if input_zeros else None,  # Method for imputing missing values
                                        freq="h",  # Frequency of the data (e.g., "h" for hourly)
                                        compute_uv=True,  # Compute u and v components of wind
                                        station_type=self.station_type # Which station type to load (None for all)
                          )
            
        df, mask = self.dataset.get_observations(return_mask=True)

        # drop 2025 data
        df = df[df.index.year < 2025]

        target_df, covariates = prepare_data(df, self.target, self.dataset.parameters)

        #get synth data
        if synth_data:
            
            if synth_data == "v1":

                adj = self.get_connectivity(threshold=20, include_self=True, layout='dist')
                adj = 1 - adj
                adj[adj < 0.9] = 0.0  # cap max weight to 1.0


                res = degree_days_from_multiindex_wide(target_df, base=10.0, upper=30.0)

                quantity = target_df.copy()# get station heights
                stations_h = self.dataset.stations_table['station_height']
                n_eggs = lambda h: 1000 * np.exp(-h / 2000)  # decay with 1km scale

                # get station number of eggs per year using elevation and as n_eggs_year_i = f(station_height_i) + 0.1*n_eggs_year_i-1
                n_eggs_year = np.zeros((stations_h.shape[0], 10))
                for year in range(10):
                    if year == 0:
                        n_eggs_year[:, year] = n_eggs(stations_h)
                    else:
                        n_eggs_year[:, year] = n_eggs(stations_h) + n_eggs_year[:, year - 1] @ (adj * 0.035)

                n_eggs_year = n_eggs_year[:, 1:]

                # convert to dataframe, index is year, columns are station names
                n_eggs_year_df = pd.DataFrame(n_eggs_year.T, columns=res.columns.get_level_values('nodes')[:len(res.columns)//2], index=range(2017, 2026))


                out = simulate_hatching_from_dd(res['dd'],
                                    eggs_per_year=n_eggs_year_df,
                                    mode="cum",
                                    dd50=700.0,
                                    k=0.20,
                                    p_max=0.1,
                                    dd50_daily=70.0,
                                    k_daily=0.07,
                                    p_max_daily=0.15,
                                    dd50_cum=700.0,
                                    k_cum=0.25,
                                    p_max_cum=0.05,
                                    lag_days=30,
                                    random_state=42,
                                    return_probs=True,
                                    mix_alpha=0.2
                                )
                # agregate data into daily data only for temperature in target df (resample to daily data)
                target_df = target_df.resample('D').mean()

                # target_df = target_df.resample('D').mean()
                self.extra_data = {'temperature': target_df.copy()}
                target_df = out['hatched_cum'].astype(float)
                target = 'hatched'

            elif synth_data == "v2":
                 
                adj = self.get_connectivity(threshold=20, include_self=True, layout='dist')
                adj = 1 - adj
                adj[adj < 0.9] = 0.0  # cap max weight to 1.0

                res = degree_days_from_multiindex_wide(target_df, base=10.0, upper=30.0)

                # gaussian distribution of eggs per station, mean 500, std 200, min 100
                # n_eggs = 500 + 200 * np.random.randn(len(res.columns)//2)
                # n_eggs[n_eggs < 100] = 100
                # n_eggs = pd.DataFrame(n_eggs.astype(int), index=res.columns.get_level_values('nodes')[:len(res.columns)//2])

                n_eggs = 500

                # get station number of eggs per year using elevation and as n_eggs_year_i = f(station_height_i) + 0.1*n_eggs_year_i-1
                out = simulate_hatching_from_dd_v2(res['dd'],
                                    adjacency_matrix=adj,
                                    eggs_first_year=n_eggs,
                                    mode="cum",
                                    dd50=500.0,
                                    k=0.20,
                                    p_max=0.1,
                                    dd50_daily=70.0,
                                    k_daily=0.07,
                                    p_max_daily=0.15,
                                    dd50_cum=500.0,
                                    k_cum=0.25,
                                    p_max_cum=0.05,
                                    lag_days=30,
                                    random_state=42,
                                    return_probs=True,
                                    mix_alpha=0.2,
                                    propagation_rate=1,
                                    dispersal_efficiency=0.5
                                )

                
                # agregate data into daily data only for temperature in target df (resample to daily data)
                target_df = target_df.resample('D').mean()

                # target_df = target_df.resample('D').mean()
                self.extra_data = {'temperature': target_df.copy()}
                target_df = out['hatched_cum'].astype(float)
                target = 'hatched'

            elif synth_data == "v3":
                 
                adj_orig = self.get_connectivity(threshold=20, include_self=True, layout='dist')
                adj = 1 - adj_orig
                adj[adj < 0.95] = 0.0  # cap max weight to 1.0

                res = degree_days_from_multiindex_wide(target_df, base=10.0, upper=30.0)

                # get soil quality vars
                positions = [[46.2, 9.016667], [47.36667, 8.55], [46.94809, 7.44744]] # bellinzona, zurich, bern
                spreads = [40, 60, 50]  # in meters, roughly 50km
                weights = [1.0] * len(positions)
                soil_quality = generate_soil_quality(self.dataset.stations_table[['latitude','longitude']], positions, spreads, weights)

                # gaussian distribution of eggs per station, mean 500, std 200, min 100
                # n_eggs = 500 + 200 * np.random.randn(len(res.columns)//2)
                # n_eggs = 500 * np.ones(len(res.columns)//2)
                # n_eggs[n_eggs < 100] = 100
                # n_eggs = pd.DataFrame(n_eggs.astype(int), index=res.columns.get_level_values('nodes')[:len(res.columns)//2])
                # # turn n_eggs into series
                # n_eggs = n_eggs[0]

                # # multiply the n_eggs by the matrix adj to have a different number of eggs per station
                # n_eggs = adj @ (adj @ n_eggs.values)
                # n_eggs = pd.DataFrame(n_eggs.astype(int), index=res.columns.get_level_values('nodes')[:len(res.columns)//2])
                # n_eggs = n_eggs[0]

                # generate n_eggs as a function of soil quality
                # n_eggs = 300 + 700 * np.ones_like(soil_quality)
                n_eggs = 300 + 700 * soil_quality
                n_eggs = pd.DataFrame(n_eggs.astype(int), index=res.columns.get_level_values('nodes')[:len(res.columns)//2])
                n_eggs = n_eggs[0]

                # get station number of eggs per year using elevation and as n_eggs_year_i = f(station_height_i) + 0.1*n_eggs_year_i-1
                out = simulate_hatching_from_dd_v3(res['dd'],
                                    adjacency_matrix=adj,
                                    eggs_first_year=n_eggs,
                                    mode="cum",
                                    dd50=500.0,
                                    k=0.20,
                                    p_max=0.1,
                                    dd50_daily=70.0,
                                    k_daily=0.07,
                                    p_max_daily=0.15,
                                    dd50_cum=500.0,
                                    k_cum=0.25,
                                    p_max_cum=0.05,
                                    lag_days=30,
                                    random_state=42,
                                    return_probs=True,
                                    mix_alpha=0.2,
                                    dispersal_rate=0.25,
                                    propagation_rate=1,
                                    dispersal_efficiency=0.5,
                                    soil_quality=None
                                )

                
                # agregate data into daily data only for temperature in target df (resample to daily data)
                target_df = target_df.resample('D').mean()

                # target_df = target_df.resample('D').mean()

                # some nodes wont have temperature data, fill with the closest station, which can be seen in adj matrix. 
                target_df, mask_temp = drop_and_fill_vectorized(target_df, adj_orig, drop_prob=0.7)

                self.extra_data = {'temperature': target_df.copy()}

                # self.extra_data = {}
                target_df = out['hatched'].astype(float)
                target = 'hatched'

        # mix all covariates into a single covariate called u
        u = []
        for key in covariates.keys():
            if key != target:
                u.append(covariates[key].values)
        u = np.stack(u, axis=-1)
        self.u = {'u': u }

        if synth_data:
            # Mask out year-station combinations where the entire year is zero
            mask = target_df.notna().astype(float).resample('YE').sum()
            
            # Create mask with proper handling of leap years
            mask_list = []
            years = target_df.index.year.unique()
            
            for year in years:
                year_mask = mask.loc[str(year)].values
                days_in_year = 366 if pd.Timestamp(year, 1, 1).is_leap_year else 365
                year_mask_expanded = np.repeat(year_mask[np.newaxis, :], days_in_year, axis=0)
                
                # Mask from day 280 onwards
                year_mask_expanded[280:, :] = 0.0
                mask_list.append(year_mask_expanded)
            
            mask = np.vstack(mask_list)[:target_df.shape[0], :]
            mask = (mask > 0).astype(float)
            
            print(f"Using synthetic data, {np.sum(mask==0)} out of {mask.size} values masked ({100*np.sum(mask==0)/mask.size:.2f}%)")
        else:
            mask = mask.xs(self.target, level="name", axis=1).values

        mask = mask[:,0,:]
        # self.covariates = covariates

        super().__init__(target=target_df,
                        #  covariates=covariates,
                            mask=mask,
                            freq=freq if not synth_data else 'D',
                            similarity_score="distance",
                            temporal_aggregation="nearest",
                            name="PeakWeather")
        
        if freq == 'D':
            
            # add also the date as covariate (cos and sin)
            date = target_df.index
            covariates['day_sin'] = np.sin(2 * np.pi * date.dayofyear / 365.25)
            covariates['day_cos'] = np.cos(2 * np.pi * date.dayofyear / 365.25)
            covariates['month_sin'] = np.sin(2 * np.pi * date.month / 12)
            covariates['month_cos'] = np.cos(2 * np.pi * date.month / 12)
            covariates['year_sin'] = np.sin(2 * np.pi * (date.year - date.year.min()) / (date.year.max() - date.year.min()))
            covariates['year_cos'] = np.cos(2 * np.pi * (date.year - date.year.min()) / (date.year.max() - date.year.min()))
            # also the year as a covariate
            covariates['year'] = date.year - date.year.min()

        # self.add_covariate('dist', dist, pattern='n n')

    def get_connectivity(self, method: str = "dist",
                         threshold: float = 0.0,
                         include_self: bool = False,
                         layout: str = "edge_index", 
                         train_slice: slice = None):
        # build dist matrix
        dist = self.dataset.stations_table

        # get only the data for latitude and longitude
        dist = dist[['latitude', 'longitude']]

        # compute distances from each node to each other node
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(dist.values, dist.values)

        if layout == 'dist':
            return dist_matrix/np.max(dist_matrix)
        
        adj = 1 - dist_matrix
        adj[adj < threshold] = 0.0  # cap max weight to 1.0

        if not include_self:
            np.fill_diagonal(adj, 0)

        if layout == "edge_index":
            from tsl.ops.connectivity import adj_to_edge_index
            return adj_to_edge_index(adj)

        return adj
    


def prepare_data(df, target, parameters):

    target_df = df.xs(target, level="name", axis=1)

    covariates = {}
    for col in parameters:
        if col != target:
            covariates[col] = df.xs(col, level="name", axis=1)

    return target_df, covariates

def add_increment_flag(df: pd.DataFrame,
                       cum_col: str = "incrementing_ostrinia",
                       flag_col: str = "increment_flag") -> pd.DataFrame:
    """
    Append a binary flag indicating where the cumulative count increases.

    Parameters
    ----------
    df : pd.DataFrame
        Input data‐frame with nodes as columns and time as index.
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

    # Detect only positive changes (increases) over time for each node
    changes = out.diff().fillna(0) > 0
    out = changes.astype(int)

    return out


def drop_and_fill_vectorized(df, adj_orig, drop_prob=0.7, seed=None):
    """Optimized version using vectorized operations where possible."""
    if seed is not None:
        np.random.seed(seed)
    
    df_filled = df.copy()
    mask = np.random.random(df.shape) < drop_prob
    mask_df = pd.DataFrame(mask, index=df.index, columns=df.columns)
    df_filled[mask_df] = np.nan
    
    # Precompute nearest neighbors (excluding self)
    adj_matrix = adj_orig.copy()
    np.fill_diagonal(adj_matrix, np.inf)
    nearest_neighbors = np.argmin(adj_matrix, axis=1)
    
    # Fill row by row
    for i, date in enumerate(df_filled.index):
        row = df_filled.iloc[i].values
        missing_mask = np.isnan(row)
        
        for j in np.where(missing_mask)[0]:
            nn_idx = nearest_neighbors[j]
            if not np.isnan(row[nn_idx]):
                df_filled.iloc[i, j] = row[nn_idx]
    
    return df_filled, mask_df