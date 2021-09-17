select
	tea.gufi,
	tea.last_tfm_position_timestamp,
	tea.last_tfm_position_altitude,
	tea.last_tfm_position_heading,
	tea.last_tfm_position_latitude,
	tea.last_tfm_position_longitude,
	tea.last_tfm_position_speed,
	tea."timestamp"
from tfm_extension_all tea
left join matm_flight_summary mfs on tea.gufi = mfs.gufi
where
	tea."timestamp" between :start_time and :end_time
	and mfs.arrival_aerodrome_icao_name = :arrival_airport_icao
