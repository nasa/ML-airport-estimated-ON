select 
	gufi,
	arrival_runway_actual_time,
	arrival_runway_actual,
	distance_from_runway,
	points_on_runway
from runways
where
	arrival_aerodrome_iata_name = :arrival_airport_iata
	and arrival_runway_actual_time between :start_time and :end_time
	and (points_on_runway = :surf_surv_avail)
order by arrival_runway_actual_time
