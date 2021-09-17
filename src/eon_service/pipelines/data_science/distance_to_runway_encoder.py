from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DistanceToRunwayEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        positions_at_landing: pd.DataFrame,
    ):
        """
        Args:
            positions_at_landing: data frame with these columns
                gufi
                arrival_runway_actual_via_surveillance
                position_latitude_at_landing
                position_longitude_at_landing
            this only needs to be provided for training data gufis
            it is used to learn a lookup table of median lat & long
            at landing per arrival runway (during .fit()), then discarded
        """
        self.positions_at_landing = positions_at_landing

    def fit(
        self,
        data,
        y
    ):
        data['actual_arrival_ramp_taxi_time']=y
        self.avg_ramp_taxi_time = (
                data.groupby(
                    ["arrival_runway_actual","arrival_stand_actual"])
                ['actual_arrival_ramp_taxi_time'].mean())

        self.overall_avg = data['actual_arrival_ramp_taxi_time'].mean()

        delattr(self, 'positions_at_landing')

        return self

    def transform(
        self,
        data,
    ) -> pd.DataFrame:

        transformed = pd.DataFrame(
            index=data.index,
            columns=[
                'avg_ramp_taxi_time'
            ],
        )
        transformed['avg_ramp_taxi_time'] = data.apply(self.get_avg_taxi_time, axis=1)

        return transformed

    def get_avg_taxi_time(
        self,
        row
    ):
        runway = row['arrival_runway_actual']
        stand = row['arrival_stand_actual']

        if runway not in self.avg_ramp_taxi_time or stand not in self.avg_ramp_taxi_time[runway]:
            return self.overall_avg

        return self.avg_ramp_taxi_time[runway][stand]

    def get_feature_names(
        self
    ):
        return [self.avg_ramp_taxi_time.name]