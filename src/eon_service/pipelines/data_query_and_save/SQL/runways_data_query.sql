select 
	gufi,
	arrival_runway_actual_time,
	arrival_runway_actual,
	distance_from_runway,
	points_on_runway
from runways
where
	airport_id = :arrival_airport_icao
	and arrival_runway_actual_time between :start_time and :end_time
order by arrival_runway_actual_time