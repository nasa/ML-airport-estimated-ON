select
	gufi,
	"timestamp",
	arrival_runway_estimated_time as tfms_eta
from matm_flight
where "timestamp" between :start_time and :end_time
	and arrival_aerodrome_icao_name = :arrival_airport_icao
	and last_update_source = 'TFM'
	and arrival_runway_estimated_time is not null